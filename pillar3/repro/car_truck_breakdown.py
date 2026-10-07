"""Recorte por classe PROPOSTA nas 4 configuracoes, com as colunas que faltavam
no artefato de 04/09: not_recovered e correcao de classe por classe.

So leitura. Nao sobrescreve nada em paper/pillar23/.
Mesma definicao de correto de compare_arms.py / vlm/score.py.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

FORGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(FORGE / "src"))

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
    if (d / "phase1.json").is_file():
        ARM_RUNS["vlm_single_classfix_v3"] = d

ORDER = ["base_yolo", "vlm_single_v1", "vlm_single_classfix_v3", "vlm_decomposed_v2"]
LABEL = {
    "base_yolo": "Base YOLO11x (sem VLM)",
    "vlm_single_v1": "VLM single verdict (v1)",
    "vlm_single_classfix_v3": "VLM single + class correction (v3)",
    "vlm_decomposed_v2": "VLM decomposed (v2)",
}
_TRUTH_OK = frozenset({"kept", "corrected-minor"})

man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
frames = [Frame.model_validate(f) for f in man["frames"]]
reviewed = [f for f in frames if f.reviewed]
tasks = box_tasks(reviewed)
by_key = {t.key: t for t in tasks}
corrected_cls = human_corrected_class(frames)

gt_by_class, added_by_class = Counter(), Counter()
for f in reviewed:
    for g in (f.ground_truth or []):
        if isinstance(g.cls_name, str):
            gt_by_class[normalize_class(g.cls_name)] += 1
    for c, _ in human_added_boxes(f):
        added_by_class[c] += 1
recoverable_by_class = {c: gt_by_class[c] - added_by_class.get(c, 0) for c in gt_by_class}


def per_class(rows_by_key):
    """rows_by_key: key -> (verdict, vlm_class). None = braco base (tudo fica)."""
    pc = defaultdict(Counter)
    for t in tasks:
        c = t.proposed_class
        pc[c]["proposals"] += 1
        if rows_by_key is None:
            pc[c]["boxes_retained"] += 1
            if t.truth in _TRUTH_OK:
                pc[c]["correct"] += 1
            continue
        v, vcls = rows_by_key.get(t.key, ("kept", None))
        v = v or "kept"
        if v == "class-changed":
            pc[c]["cls_try"] += 1
        if v == "rejected":
            continue
        lab = vcls if (v == "class-changed" and vcls) else t.proposed_class
        pc[c]["boxes_retained"] += 1
        if t.truth == "rejected" or t.truth == "corrected-redrawn":
            pass
        elif t.truth == "class-changed":
            want = corrected_cls.get(t.key)
            if want is not None and lab == want:
                pc[c]["correct"] += 1
                if v == "class-changed":
                    pc[c]["cls_ok"] += 1
            elif v == "class-changed":
                pc[c]["cls_bad"] += 1
        else:
            if lab == t.proposed_class:
                pc[c]["correct"] += 1
            elif v == "class-changed":
                pc[c]["cls_bad"] += 1
    return pc


arms = {"base_yolo": per_class(None)}
for name, run in ARM_RUNS.items():
    rows = [Phase1Row(**r) for r in json.loads((run / "phase1.json").read_text(encoding="utf-8"))]
    arms[name] = per_class({r.key: (r.vlm, r.vlm_class) for r in rows if r.key in by_key})

CLASSES = [c for c, _ in gt_by_class.most_common()]
rows_out = []
for k in ORDER:
    if k not in arms:
        continue
    for c in CLASSES:
        d = arms[k].get(c, Counter())
        gt = gt_by_class.get(c, 0)
        rec = recoverable_by_class.get(c, 0)
        ret, ok = d.get("boxes_retained", 0), d.get("correct", 0)
        rows_out.append({
            "arm": k,
            "label": LABEL[k],
            "proposed_class": c,
            "gt_count": gt,
            "gt_never_proposed_added": added_by_class.get(c, 0),
            "gt_with_a_proposal": rec,
            "proposals_of_this_class": d.get("proposals", 0),
            "boxes_retained": ret,
            "correct": ok,
            "incorrect": ret - ok,
            "not_recovered": gt - ok,
            "precision_pct": round(100 * ok / ret, 1) if ret else None,
            "coverage_over_class_gt_pct": round(100 * ok / gt, 1) if gt else None,
            "reachable_recovery_over_class_with_proposal_pct": round(100 * ok / rec, 1) if rec else None,
            "class_corrections_attempted": d.get("cls_try", 0),
            "class_corrections_correct": d.get("cls_ok", 0),
            "class_corrections_incorrect": d.get("cls_bad", 0),
            # Denominador consistente? `correct` e contado na linha da classe
            # PROPOSTA; gt_count/gt_with_a_proposal sao da classe do GABARITO.
            # Quando correcao de classe move massa para ca, os dois deixam de
            # ser a mesma particao e cobertura/recuperacao passam de 100%.
            "denominator_ok": bool(gt and rec and ok <= rec),
        })

cols = list(rows_out[0].keys())
with (OUT / "car-truck-breakdown.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, delimiter=";")
    w.writeheader()
    w.writerows(rows_out)

meta = {
    "_fonte_manifest": str(ROOT / "manifest.json"),
    "_arm_runs": {k: str(v) for k, v in ARM_RUNS.items()},
    "_definicao_correto": ("TP = a caixa sobrevive a curadoria do braco E (veredito humano in "
                           "{kept, corrected-minor} com rotulo preservado) OU (veredito humano == "
                           "class-changed e o rotulo final == o do humano). Sem IoU contra o GT."),
    "_aviso_denominador": ("`proposed_class` e a classe PROPOSTA pelo detector. gt_count e "
                           "added/gt_with_a_proposal sao do GABARITO daquela classe. As duas "
                           "familias nao sao a mesma particao: uma correcao de classe move o "
                           "acerto para a linha da classe PROPOSTA."),
    "gt_by_class": dict(gt_by_class.most_common()),
    "added_by_class": dict(added_by_class.most_common()),
    "recoverable_by_class": recoverable_by_class,
    "_classes_fora_da_regua_731": ["airplane", "train"],
    "_nota_regua": ("gt_by_class soma 735 e recoverable soma 481 porque incluem airplane e train. A regua publicada (731 GT / 477 recuperaveis) exclui as duas — GT_EXCLUDED_CLASSES."),
    "checks": {
        "sum_gt": sum(gt_by_class.values()),
        "sum_added": sum(added_by_class.values()),
        "sum_recoverable": sum(recoverable_by_class.values()),
        "sum_proposals_por_classe": {k: sum(v.get("proposals", 0) for v in arms[k].values()) for k in arms},
    },
    "rows": rows_out,
}
(OUT / "car-truck-breakdown.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")

print(json.dumps(meta["checks"], indent=1))
print()
h = f"{'configuracao':<36}{'cls':<11}{'GT':>5}{'s/prop':>7}{'c/prop':>7}{'prop':>6}{'ret':>5}{'ok':>5}{'err':>5}{'naoRec':>7}{'prec':>7}{'cob':>7}{'rec':>7}{'cc t/o/x':>10}"
print(h); print('-' * len(h))
for r in rows_out:
    print(f"{r['label']:<36}{r['proposed_class']:<11}{r['gt_count']:>5}{r['gt_never_proposed_added']:>7}"
          f"{r['gt_with_a_proposal']:>7}{r['proposals_of_this_class']:>6}{r['boxes_retained']:>5}"
          f"{r['correct']:>5}{r['incorrect']:>5}{r['not_recovered']:>7}"
          f"{str(r['precision_pct']):>7}{str(r['coverage_over_class_gt_pct']):>7}"
          f"{str(r['reachable_recovery_over_class_with_proposal_pct']):>7}"
          f"{str(r['class_corrections_attempted'])+'/'+str(r['class_corrections_correct'])+'/'+str(r['class_corrections_incorrect']):>10}")
