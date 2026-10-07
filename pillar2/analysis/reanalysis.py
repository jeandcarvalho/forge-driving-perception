"""Reanalise dos Pilares 2 e 3 - SO LEITURA, arquivos de saida NOVOS.

Responde, a partir do manifest local e dos runs ja gravados:
  1. decomposicao 254 / 402 / 148
  2. contagem dos 5 vereditos humanos nas 501 propostas
  3. definicao operacional de "correto" em cada braco
  6. denominadores de recuperacao (recall condicionado)
  7. semantica x geometria
  9. recorte car/truck

Nao chama VLM, nao escreve no banco, nao toca run antigo.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from forge.api.models import Frame
from forge.config import GT_EXCLUDED_CLASSES, PARITY_IOU
from forge.evaluate.parity import curation_parity
from forge.vlm.phases import Phase1Row
from forge.vlm.prepare import box_tasks, human_added_boxes, human_verdict
from forge.vlm.score import human_corrected_class

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent
V1_RUN = RUNS / "20260820T031131Z-vlm-audit"   # single / veredito holistico
V2_RUN = RUNS / "20260821T002801Z-vlm-audit"   # decomposed
P2_RUN = RUNS / "20260820T173145Z-vlm-audit"   # fase 2 (enumeracao)

_TRUTH_OK = frozenset({"kept", "corrected-minor"})

PUBLISHED_735 = {
    "reviewed_frames": 313, "detections": 505, "ground_truth": 735, "tp": 333,
    "fp": 172, "fn": 402, "kept": 249, "corrected_box_minor": 84,
    "corrected_box_major": 93, "corrected_class": 55, "rejected": 24, "added": 254,
}


def load_frames():
    doc = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    return [Frame.model_validate(f) for f in doc["frames"]]


def load_rows(run: Path):
    raw = json.loads((run / "phase1.json").read_text(encoding="utf-8"))
    return [Phase1Row(**r) for r in raw]


def analyse_arm(rows, corrected_cls, by_key, gt, recoverable, gt_by_class):
    """Reaplica score_phase1 e abre em semantica x geometria."""
    boxes = tp = 0
    detail = Counter()
    geom = defaultdict(Counter)
    presence = Counter()
    cls_attempts = cls_ok = cls_bad = 0
    per_class = defaultdict(Counter)
    surviving_truth = Counter()

    for r in rows:
        t = by_key[r.key]
        v = r.vlm
        if v is None:
            v = "kept"
            detail["falha_chamada_tratada_como_kept"] += 1

        vlm_present = v != "rejected"
        truth_real = r.truth != "rejected"
        presence[("real" if truth_real else "background") + "__vlm_" +
                 ("present" if vlm_present else "absent")] += 1

        geom[r.truth][v] += 1
        if v == "class-changed":
            cls_attempts += 1

        cls_key = t.proposed_class
        per_class[cls_key]["proposals"] += 1

        if v == "rejected":
            if r.truth == "rejected":
                detail["removeu_fp_de_fundo"] += 1
            elif r.truth in _TRUTH_OK:
                detail["removeu_caixa_boa"] += 1
            else:
                detail["removeu_fp_de_erro"] += 1
            continue

        label_cls = r.vlm_class if v == "class-changed" and r.vlm_class else r.proposed_class
        boxes += 1
        per_class[cls_key]["boxes_retained"] += 1

        if r.truth == "rejected":
            detail["manteve_fp_de_fundo"] += 1
        elif r.truth == "corrected-redrawn":
            detail["manteve_caixa_mal_posicionada"] += 1
        elif r.truth == "class-changed":
            want = corrected_cls.get(r.key)
            if want is not None and label_cls == want:
                tp += 1
                per_class[cls_key]["tp"] += 1
                detail["corrigiu_rotulo_certo"] += 1
                if v == "class-changed":
                    cls_ok += 1
            else:
                detail["rotulo_continua_errado"] += 1
                if v == "class-changed":
                    cls_bad += 1
        else:
            if label_cls == r.proposed_class:
                tp += 1
                per_class[cls_key]["tp"] += 1
                detail["manteve_caixa_boa"] += 1
            else:
                detail["estragou_caixa_boa"] += 1
                if v == "class-changed":
                    cls_bad += 1
        surviving_truth[r.truth] += 1

    geom_binary = {}
    for truth in ("kept", "corrected-minor", "corrected-redrawn"):
        row = geom.get(truth, Counter())
        n = sum(row.values())
        said_good = row.get("kept", 0)
        geom_binary[truth] = {
            "n": n,
            "vlm_said_tight_kept": said_good,
            "vlm_said_needs_change_or_reject": n - said_good,
            "pct_said_tight": round(100 * said_good / n, 1) if n else None,
        }

    pc = {}
    for k, v in sorted(per_class.items()):
        d = dict(v)
        d["gt_total_for_class"] = gt_by_class.get(k, 0)
        d["precision"] = (round(100 * d.get("tp", 0) / d["boxes_retained"], 1)
                          if d.get("boxes_retained") else None)
        d["recovery_over_gt_class"] = (round(100 * d.get("tp", 0) / d["gt_total_for_class"], 1)
                                       if d["gt_total_for_class"] else None)
        pc[k] = d

    return {
        "definition_of_correct": (
            "TP = caixa que SOBREVIVE a curadoria do VLM E cujo veredito HUMANO e "
            "kept|corrected-minor com rotulo preservado, OU cujo veredito humano e "
            "class-changed e o VLM emitiu exatamente a classe que o humano pos. "
            "Nao ha matching por IoU contra o GT: o vinculo e ref_det_id."
        ),
        "boxes_retained": boxes, "tp": tp, "fp": boxes - tp,
        "gt": gt, "fn": gt - tp,
        "precision": round(100 * tp / boxes, 1) if boxes else None,
        "recall_over_gt": round(100 * tp / gt, 1) if gt else None,
        "recall_over_recoverable": round(100 * tp / recoverable, 1) if recoverable else None,
        "recoverable_denominator": recoverable,
        "detail": dict(sorted(detail.items())),
        "semantics": {
            "presence_confusion": dict(sorted(presence.items())),
            "class_corrections_attempted": cls_attempts,
            "class_corrections_right": cls_ok,
            "class_corrections_wrong": cls_bad,
            "class_correction_success_rate":
                round(100 * cls_ok / cls_attempts, 1) if cls_attempts else None,
        },
        "geometry": {
            "human_verdict_x_vlm_verdict": {k: dict(v) for k, v in geom.items()},
            "binary_view": geom_binary,
        },
        "surviving_by_human_verdict": dict(surviving_truth),
        "per_proposed_class": pc,
    }


def main():
    out = {"_meta": {
        "manifest": str(ROOT / "manifest.json"),
        "v1_run": str(V1_RUN), "v2_run": str(V2_RUN), "phase2_run": str(P2_RUN),
        "parity_iou": PARITY_IOU,
        "gt_excluded_classes": sorted(GT_EXCLUDED_CLASSES),
    }}

    frames = load_frames()
    reviewed = [f for f in frames if f.reviewed]
    tasks = box_tasks(reviewed)
    by_key = {t.key: t for t in tasks}

    # ---- (2) cinco vereditos ----
    v501 = Counter(t.truth for t in tasks)
    v505 = Counter()
    for f in reviewed:
        for d in f.stored_detections:
            hv = human_verdict(d)
            if hv is not None and d.box_n is not None and isinstance(d.cls_name, str):
                v505[hv] += 1
    unreviewed_dets = sum(1 for f in reviewed for d in f.stored_detections
                          if d.verdict == "unreviewed")
    no_corr = sum(1 for f in reviewed for d in f.stored_detections
                  if d.verdict == "corrected" and d.correction is None)
    added_total = sum(len(human_added_boxes(f)) for f in reviewed)
    added_by_class = Counter(c for f in reviewed for c, _ in human_added_boxes(f))

    n501 = sum(v501.values())
    out["verdicts"] = {
        "ruler_501": {
            "counts": dict(v501), "total": n501,
            "pct": {k: round(100 * v / n501, 1) for k, v in v501.items()},
        },
        "ruler_505_before_class_exclusion": {
            "counts": dict(v505), "total": sum(v505.values()),
        },
        "other_states_in_data": {
            "unreviewed_detections_in_reviewed_frames": unreviewed_dets,
            "corrected_without_correction_payload": no_corr,
            "note": ("verdict bruto no banco tem 4 estados: kept|corrected|rejected|"
                     "unreviewed. As 5 categorias saem de `corrected` decomposto por "
                     "classe e por IoU(orig,correcao) >= 0.7."),
        },
        "human_added_never_proposed": {
            "total": added_total, "by_class": dict(added_by_class.most_common()),
        },
    }

    # ---- (1) decomposicao ----
    rep = curation_parity(frames)
    kept, minor = v501["kept"], v501["corrected-minor"]
    major, cls_ch, rej = v501["corrected-redrawn"], v501["class-changed"], v501["rejected"]
    dets = kept + minor + major + cls_ch + rej
    tp = kept + minor
    gt = dets - rej + added_total
    fn = added_total + major + cls_ch
    out["decomposition_254_402"] = {
        "detections_proposals": dets,
        "ground_truth": gt,
        "tp_base": tp,
        "fp_base": major + cls_ch + rej,
        "fn_base": fn,
        "identity_gt_minus_tp": {"gt_minus_tp": gt - tp, "fn": fn, "equal": gt - tp == fn},
        "added_254": {
            "value": added_total,
            "meaning": ("objetos do gabarito que o detector NUNCA propos - "
                        "desenhados a mao (origin == 'added')"),
        },
        "had_proposal_but_failed_148": {
            "value": major + cls_ch,
            "corrected_redrawn": major,
            "class_changed": cls_ch,
            "meaning": ("objetos do gabarito que TINHAM proposta do detector mas a "
                        "proposta nao foi aceita como correta: geometria refeita "
                        "(mesma classe, IoU(orig,corr) < 0.7) ou classe trocada pelo humano"),
        },
        "sum_check": {"added_plus_failed": added_total + major + cls_ch, "fn": fn,
                      "equal": added_total + major + cls_ch == fn},
        "parity_report_full": rep.as_dict(),
        "parity_diff_vs_published_735": rep.diff_against(PUBLISHED_735),
    }

    # ---- (6) denominadores ----
    recoverable = kept + minor + major + cls_ch
    out["recovery_denominators"] = {
        "gt_total": gt,
        "detector_proposals": dets,
        "gt_not_proposed_by_detector": added_total,
        "gt_with_a_detector_proposal": recoverable,
        "check": {"recoverable_plus_added": recoverable + added_total, "gt": gt,
                  "equal": recoverable + added_total == gt},
        "note": "as 24 `rejected` sao propostas sobre nada: entram em proposals, nao em GT.",
    }

    # GT por classe (classe RESOLVIDA do gabarito), para o recorte car/truck
    gt_by_class = Counter()
    for f in reviewed:
        for g in (f.ground_truth or []):
            if isinstance(g.cls_name, str):
                from forge.core.classes import normalize_class
                gt_by_class[normalize_class(g.cls_name)] += 1
    out["gt_by_class"] = dict(gt_by_class.most_common())

    # ---- base YOLO por classe ----
    base_pc = defaultdict(Counter)
    for t in tasks:
        base_pc[t.proposed_class]["proposals"] += 1
        base_pc[t.proposed_class]["boxes_retained"] += 1
        if t.truth in _TRUTH_OK:
            base_pc[t.proposed_class]["tp"] += 1
    base_class = {}
    for k, v in sorted(base_pc.items()):
        d = dict(v)
        d["gt_total_for_class"] = gt_by_class.get(k, 0)
        d["precision"] = round(100 * d.get("tp", 0) / d["boxes_retained"], 1)
        d["recovery_over_gt_class"] = (round(100 * d.get("tp", 0) / d["gt_total_for_class"], 1)
                                       if d["gt_total_for_class"] else None)
        base_class[k] = d

    out["arms"] = {
        "base_yolo": {
            "definition_of_correct": ("TP = veredito humano in {kept, corrected-minor}. "
                                      "Todas as 501 propostas ficam. Sem VLM."),
            "boxes_retained": dets, "tp": tp, "fp": dets - tp,
            "gt": gt, "fn": gt - tp,
            "precision": round(100 * tp / dets, 1),
            "recall_over_gt": round(100 * tp / gt, 1),
            "recall_over_recoverable": round(100 * tp / recoverable, 1),
            "recoverable_denominator": recoverable,
            "per_proposed_class": base_class,
        }
    }

    corrected_cls = human_corrected_class(frames)
    for label, run in (("vlm_single_v1", V1_RUN), ("vlm_decomposed_v2", V2_RUN)):
        all_rows = load_rows(run)
        rows = [r for r in all_rows if r.key in by_key]
        a = analyse_arm(rows, corrected_cls, by_key, gt, recoverable, gt_by_class)
        a["rows_in_phase1_file"] = len(all_rows)
        a["rows_used_on_501_ruler"] = len(rows)
        a["rows_dropped_off_ruler"] = len(all_rows) - len(rows)
        a["run_dir"] = str(run)
        out["arms"][label] = a

    # ---- (8) uma caixa util a cada onze ----
    p2 = json.loads((P2_RUN / "meta.json").read_text(encoding="utf-8"))["results"]["phase2"]
    out["one_in_eleven"] = {
        "source_run": str(P2_RUN),
        "proposals_total": p2["proposals"],
        "matched_by_presence": p2["matched_proposals"],
        "matched_presence_and_class": p2["class_correct"],
        "human_added_targets": p2["human_added"],
        "ratio_proposals_per_presence_hit": round(p2["proposals"] / p2["matched_proposals"], 2),
        "ratio_proposals_per_class_correct": round(p2["proposals"] / p2["class_correct"], 2),
        "presence_rule": "IoU >= 0.3 (VLM_RECALL_PRESENCE_IOU) OR centro dentro da caixa humana",
        "by_class": p2.get("by_class"),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "pillar23-audit-reanalysis.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    slim = {k: v for k, v in out.items() if k != "arms"}
    print(json.dumps(slim, indent=2, ensure_ascii=False))
    print("\n================ ARMS ================")
    for k, v in out["arms"].items():
        print("\n---- " + k + " ----")
        print(json.dumps(v, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
