"""Reproduz o confidence-threshold sweep da secao 6.2 do dossie.

Fonte: runs/20260816T220716Z-infer/predictions.json (yolo11x.pt, conf 0.001,
split test, 313 imagens) contra annotations/instances_test.json.

SO LEITURA. Nao cria run novo do forge, escreve so o JSON/CSV desta pasta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from forge.evaluate.coco_map import predictions_to_matcher_input
from forge.evaluate.matcher import EvalClassScope, match

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
PRED = ROOT / "runs" / "20260816T220716Z-infer" / "predictions.json"
GT = ROOT / "annotations" / "instances_test.json"
OUT = Path(__file__).resolve().parent

CONFS = (0.25, 0.50, 0.70, 0.80, 0.90)
PUBLISHED = {  # dossie.md secao 6.2 / numeros.csv p2_eval
    0.25: (62.0, 82.2, 130),
    0.50: (77.5, 75.0, 183),
    0.70: (88.0, 64.2, 262),
    0.80: (93.8, 50.1, 365),
    0.90: (98.0, 19.8, 586),
}


def main():
    rows = []
    for scope in (EvalClassScope.GT_PRESENT, EvalClassScope.ALL_COCO):
        for iou_t in (0.5, 0.7):
            preds_all, gts = predictions_to_matcher_input(PRED, GT)
            for c in CONFS:
                preds = [p for p in preds_all if p.conf >= c]
                res = match(preds, gts, iou_threshold=iou_t, class_scope=scope)
                k = res.counters
                det, gt = k["detections"], k["ground_truth"]
                tp = k["tp"]
                lost = k["fn_undetected"]
                row = {
                    "class_scope": scope.value, "iou": iou_t, "conf": c,
                    "detections": det, "ground_truth": gt, "tp": tp,
                    "fp": k["fp_localization"] + k["fp_classification"]
                          + k["fp_duplicate"] + k["fp_background"],
                    "fn": k["fn_from_errors"] + k["fn_undetected"],
                    "fn_undetected_objects_lost": lost,
                    "precision_tp_over_det": round(100 * tp / det, 1) if det else None,
                    "recall_tp_over_gt": round(100 * tp / gt, 1) if gt else None,
                    "recall_1_minus_lost_over_gt": round(100 * (1 - lost / gt), 1) if gt else None,
                    "counters": k,
                }
                pub = PUBLISHED[c]
                # "objetos perdidos" do dossie = GT - TP (FN total), NAO fn_undetected.
                row["objects_lost_gt_minus_tp"] = gt - tp
                row["published_precision"] = pub[0]
                row["published_recall"] = pub[1]
                row["published_objects_lost"] = pub[2]
                row["matches_published"] = (
                    row["precision_tp_over_det"] == pub[0]
                    and row["recall_tp_over_gt"] == pub[1]
                    and gt - tp == pub[2]
                )
                rows.append(row)

    out = {
        "_meta": {
            "predictions": str(PRED), "ground_truth": str(GT),
            "predictions_run_params": json.loads(
                (PRED.parent / "meta.json").read_text(encoding="utf-8"))["params"],
            "note": ("dossie 6.2 nao registra qual class_scope nem qual IoU usou; "
                     "por isso as 4 combinacoes sao testadas aqui."),
        },
        "sweep": rows,
    }
    (OUT / "pillar23-sweep-repro.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    hdr = (f"{'scope':<12}{'IoU':>5}{'conf':>6}{'det':>7}{'gt':>6}{'tp':>6}{'P%':>7}"
           f"{'R=tp/gt':>9}{'perdidos':>10}{'naodetect':>11}{'bate?':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['class_scope']:<12}{r['iou']:>5}{r['conf']:>6}{r['detections']:>7}"
              f"{r['ground_truth']:>6}{r['tp']:>6}{r['precision_tp_over_det']:>7}"
              f"{r['recall_tp_over_gt']:>9}{r['objects_lost_gt_minus_tp']:>10}"
              f"{r['fn_undetected_objects_lost']:>11}"
              f"{'SIM' if r['matches_published'] else '-':>7}")
    print("\npublicado no dossie 6.2:")
    for c, (p, rc, l) in PUBLISHED.items():
        print(f"  conf {c}: P={p}%  R={rc}%  perdidos={l}")


if __name__ == "__main__":
    main()
