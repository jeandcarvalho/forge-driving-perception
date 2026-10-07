"""Tabela comparavel das 4 configuracoes + recorte car/truck.

Base YOLO · VLM single (v1) · VLM single + class correction (v3) · VLM decomposed (v2)

Mesma definicao operacional de "correto" nas quatro linhas, a de score.py:
  TP = a caixa SOBREVIVE a curadoria E
       (veredito humano in {kept, corrected-minor} e o rotulo final == o proposto)
       OU (veredito humano == class-changed e o rotulo final == o que o humano pos)
Sem matching por IoU contra o gabarito: o vinculo e ref_det_id.

Escreve pillar23-audit-reanalysis.csv e pillar3-single-classfix.json.
So leitura das fontes.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from forge.api.models import Frame
from forge.core.classes import normalize_class
from forge.vlm.phases import Phase1Row
from forge.vlm.prepare import box_tasks, human_added_boxes
from forge.vlm.score import human_corrected_class

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent

ARM_RUNS = {
    "vlm_single_v1": RUNS / "20260820T031131Z-vlm-audit",
    "vlm_decomposed_v2": RUNS / "20260821T002801Z-vlm-audit",
}
for d in sorted(RUNS.glob("*-vlm-audit-v3-single-classfix")):
    ARM_RUNS["vlm_single_classfix_v3"] = d

ORDER = ["base_yolo", "vlm_single_v1", "vlm_single_classfix_v3", "vlm_decomposed_v2"]
LABEL = {
    "base_yolo": "Base YOLO11x (sem VLM)",
    "vlm_single_v1": "VLM single verdict (v1)",
    "vlm_single_classfix_v3": "VLM single + class correction (v3)",
    "vlm_decomposed_v2": "VLM decomposed (v2)",
}
_TRUTH_OK = frozenset({"kept", "corrected-minor"})
FOCUS = ("car", "truck")


def load():
    frames = [Frame.model_validate(f) for f in
              json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))["frames"]]
    reviewed = [f for f in frames if f.reviewed]
    return frames, reviewed


def score(rows, corrected_cls, by_key):
    """score_phase1 reimplementado com contabilidade por classe PROPOSTA."""
    boxes = tp = 0
    detail = Counter()
    pc = defaultdict(Counter)
    cls_try = cls_ok = cls_bad = 0
    for r in rows:
        v = r.vlm or "kept"
        if r.vlm is None:
            detail["falha_chamada_tratada_como_kept"] += 1
        c = by_key[r.key].proposed_class
        pc[c]["proposals"] += 1
        if v == "class-changed":
            cls_try += 1
        if v == "rejected":
            if r.truth == "rejected":
                detail["removeu_fp_de_fundo"] += 1
            elif r.truth in _TRUTH_OK:
                detail["removeu_caixa_boa"] += 1
            else:
                detail["removeu_fp_de_erro"] += 1
            continue
        lab = r.vlm_class if v == "class-changed" and r.vlm_class else r.proposed_class
        boxes += 1
        pc[c]["boxes_retained"] += 1
        if r.truth == "rejected":
            detail["manteve_fp_de_fundo"] += 1
        elif r.truth == "corrected-redrawn":
            detail["manteve_caixa_mal_posicionada"] += 1
        elif r.truth == "class-changed":
            want = corrected_cls.get(r.key)
            if want is not None and lab == want:
                tp += 1; pc[c]["tp"] += 1
                detail["corrigiu_rotulo_certo"] += 1
                if v == "class-changed":
                    cls_ok += 1
            else:
                detail["rotulo_continua_errado"] += 1
                if v == "class-changed":
                    cls_bad += 1
        else:
            if lab == r.proposed_class:
                tp += 1; pc[c]["tp"] += 1
                detail["manteve_caixa_boa"] += 1
            else:
                detail["estragou_caixa_boa"] += 1
                if v == "class-changed":
                    cls_bad += 1
    return boxes, tp, dict(sorted(detail.items())), pc, (cls_try, cls_ok, cls_bad)


def main():
    frames, reviewed = load()
    tasks = box_tasks(reviewed)
    by_key = {t.key: t for t in tasks}
    corrected_cls = human_corrected_class(frames)

    v = Counter(t.truth for t in tasks)
    added = sum(len(human_added_boxes(f)) for f in reviewed)
    dets = sum(v.values())
    GT = dets - v["rejected"] + added
    RECOVERABLE = v["kept"] + v["corrected-minor"] + v["corrected-redrawn"] + v["class-changed"]

    gt_by_class = Counter()
    added_by_class = Counter()
    for f in reviewed:
        for g in (f.ground_truth or []):
            if isinstance(g.cls_name, str):
                gt_by_class[normalize_class(g.cls_name)] += 1
        for c, _ in human_added_boxes(f):
            added_by_class[c] += 1
    # GT por classe que TINHA proposta = GT da classe menos as desenhadas na mao
    recoverable_by_class = {c: gt_by_class[c] - added_by_class.get(c, 0) for c in gt_by_class}

    arms = {}
    # base
    pc = defaultdict(Counter)
    for t in tasks:
        pc[t.proposed_class]["proposals"] += 1
        pc[t.proposed_class]["boxes_retained"] += 1
        if t.truth in _TRUTH_OK:
            pc[t.proposed_class]["tp"] += 1
    arms["base_yolo"] = {
        "boxes_retained": dets, "tp": v["kept"] + v["corrected-minor"],
        "detail": {}, "per_class": pc,
        "class_corrections": (0, 0, 0),
        "run_dir": None, "rows_in_file": dets, "rows_used": dets,
    }
    for name, run in ARM_RUNS.items():
        p1 = run / "phase1.json"
        if not p1.is_file():
            print(f"[aviso] {name}: phase1.json ainda nao existe em {run}")
            continue
        allr = [Phase1Row(**r) for r in json.loads(p1.read_text(encoding="utf-8"))]
        rows = [r for r in allr if r.key in by_key]
        b, tp, det, pcx, cc = score(rows, corrected_cls, by_key)
        arms[name] = {"boxes_retained": b, "tp": tp, "detail": det, "per_class": pcx,
                      "class_corrections": cc, "run_dir": str(run),
                      "rows_in_file": len(allr), "rows_used": len(rows)}

    # ── tabela principal ──
    table = []
    for k in ORDER:
        if k not in arms:
            continue
        a = arms[k]
        b, tp = a["boxes_retained"], a["tp"]
        cc_try, cc_ok, cc_bad = a["class_corrections"]
        table.append({
            "arm": k, "label": LABEL[k],
            "boxes_retained": b, "correct_tp": tp, "incorrect_fp": b - tp,
            "not_recovered_fn": GT - tp, "ground_truth": GT,
            "precision_pct": round(100 * tp / b, 1) if b else None,
            "reference_coverage_over_731_pct": round(100 * tp / GT, 1),
            "recovery_over_recoverable_477_pct": round(100 * tp / RECOVERABLE, 1),
            "class_corrections_attempted": cc_try,
            "class_corrections_right": cc_ok,
            "class_corrections_wrong": cc_bad,
            "detail": a["detail"],
            "run_dir": a["run_dir"],
            "rows_in_phase1_file": a["rows_in_file"],
            "rows_scored_on_501": a["rows_used"],
        })

    # ── car / truck ──
    focus = []
    for k in ORDER:
        if k not in arms:
            continue
        for c in FOCUS:
            d = arms[k]["per_class"].get(c, Counter())
            ret, tp = d.get("boxes_retained", 0), d.get("tp", 0)
            focus.append({
                "arm": k, "label": LABEL[k], "class": c,
                "gt_count": gt_by_class.get(c, 0),
                "gt_never_proposed_added": added_by_class.get(c, 0),
                "gt_with_a_proposal": recoverable_by_class.get(c, 0),
                "proposals_of_this_class": d.get("proposals", 0),
                "boxes_retained": ret, "correct": tp, "incorrect": ret - tp,
                "precision_pct": round(100 * tp / ret, 1) if ret else None,
                "coverage_over_class_gt_pct": round(100 * tp / gt_by_class[c], 1)
                                              if gt_by_class.get(c) else None,
                "recovery_over_class_recoverable_pct":
                    round(100 * tp / recoverable_by_class[c], 1)
                    if recoverable_by_class.get(c) else None,
                "caveat": ("`class` e a classe PROPOSTA pelo detector, nao a do gabarito. "
                           "Uma correcao de classe move o acerto para a linha da classe "
                           "proposta, nao da classe verdadeira."),
            })

    out = {
        "_meta": {
            "definition_of_correct": (
                "TP = caixa sobrevive a curadoria E (veredito humano in {kept, "
                "corrected-minor} com rotulo preservado) OU (veredito humano == "
                "class-changed e o rotulo final == o do humano). Sem IoU contra o GT."),
            "ground_truth": GT, "detector_proposals": dets,
            "gt_not_proposed_by_detector": added,
            "gt_with_a_detector_proposal": RECOVERABLE,
            "human_verdicts_501": dict(v),
            "gt_by_class": dict(gt_by_class.most_common()),
            "added_by_class": dict(added_by_class.most_common()),
            "arm_runs": {k: str(x) for k, x in ARM_RUNS.items()},
        },
        "main_table": table,
        "car_truck": focus,
    }
    (OUT / "pillar3-single-classfix.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    with (OUT / "pillar23-audit-reanalysis.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter=";")
        w.writerow(["bloco", "arm", "classe", "metrica", "valor", "observacao"])
        for r in table:
            for m in ("boxes_retained", "correct_tp", "incorrect_fp", "not_recovered_fn",
                      "ground_truth", "precision_pct", "reference_coverage_over_731_pct",
                      "recovery_over_recoverable_477_pct", "class_corrections_attempted",
                      "class_corrections_right", "class_corrections_wrong"):
                w.writerow(["principal", r["arm"], "", m, r[m], r["label"]])
        for r in focus:
            for m in ("gt_count", "gt_never_proposed_added", "gt_with_a_proposal",
                      "proposals_of_this_class", "boxes_retained", "correct", "incorrect",
                      "precision_pct", "coverage_over_class_gt_pct",
                      "recovery_over_class_recoverable_pct"):
                w.writerow(["car_truck", r["arm"], r["class"], m, r[m], r["label"]])

    hdr = (f"{'configuracao':<38}{'caixas':>7}{'ok':>6}{'erro':>6}{'nao rec.':>9}"
           f"{'precisao':>10}{'cob/731':>9}{'rec/477':>9}")
    print(hdr); print("-" * len(hdr))
    for r in table:
        print(f"{r['label']:<38}{r['boxes_retained']:>7}{r['correct_tp']:>6}"
              f"{r['incorrect_fp']:>6}{r['not_recovered_fn']:>9}"
              f"{str(r['precision_pct'])+'%':>10}"
              f"{str(r['reference_coverage_over_731_pct'])+'%':>9}"
              f"{str(r['recovery_over_recoverable_477_pct'])+'%':>9}")
    print("\ncorrecao de classe (tentadas / certas / erradas):")
    for r in table:
        print(f"  {r['label']:<38}{r['class_corrections_attempted']:>4}"
              f"{r['class_corrections_right']:>6}{r['class_corrections_wrong']:>6}")
    print("\ncar / truck:")
    h2 = (f"{'configuracao':<38}{'cls':<7}{'gtGT':>6}{'s/prop':>8}{'c/prop':>8}"
          f"{'ret':>6}{'ok':>5}{'prec':>8}{'cob':>7}{'rec':>7}")
    print(h2); print("-" * len(h2))
    for r in focus:
        print(f"{r['label']:<38}{r['class']:<7}{r['gt_count']:>6}"
              f"{r['gt_never_proposed_added']:>8}{r['gt_with_a_proposal']:>8}"
              f"{r['boxes_retained']:>6}{r['correct']:>5}"
              f"{str(r['precision_pct'])+'%':>8}"
              f"{str(r['coverage_over_class_gt_pct'])+'%':>7}"
              f"{str(r['recovery_over_class_recoverable_pct'])+'%':>7}")


if __name__ == "__main__":
    main()
