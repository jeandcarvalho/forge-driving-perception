"""Audit nos dois ramos: com as 4 caixas (735) e sem elas (731)."""
import json
from pathlib import Path

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
test = [f for f in man["frames"] if f["split"] == "test"]
EXCL = {"airplane", "train"}


def iou(a, b):
    x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    i = (x2 - x1) * (y2 - y1)
    ar = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    br = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return i / (ar + br - i)


def audit(drop_excluded: bool, label: str):
    kept = rejected = minor = major = klass = dets = 0
    for f in test:
        # det_ids do gabarito que caem nas classes excluidas
        gone = {g["ref_det_id"] for g in f["ground_truth"]
                if drop_excluded and g["class"] in EXCL and g["ref_det_id"]}
        for d in f.get("stored_detections", []):
            if drop_excluded and d["det_id"] in gone:
                continue
            v = d["verdict"]
            # deteccao corrigida PARA uma classe excluida tambem sai
            if drop_excluded and v == "corrected" and d["correction"]["class"] in EXCL:
                continue
            dets += 1
            if v == "kept":
                kept += 1
            elif v == "rejected":
                rejected += 1
            else:
                cor = d["correction"]
                if cor["class"] != d["class"]:
                    klass += 1
                elif iou(d["box_n"], cor["box_n"]) >= 0.7:
                    minor += 1
                else:
                    major += 1
    added = sum(1 for f in test for g in f["ground_truth"]
                if g["origin"] == "added" and not (drop_excluded and g["class"] in EXCL))
    gt = dets - rejected + added
    tp, fp, fn = kept + minor, major + klass + rejected, added + major + klass
    print(f"\n--- {label} ---")
    print(f"deteccoes propostas {dets:5}   gabarito {gt:5}")
    print(f"  kept {kept:5} ({100*kept/dets:5.1f}%) | leve {minor:4} ({100*minor/dets:4.1f}%) | "
          f"refeita {major:4} ({100*major/dets:4.1f}%) | classe {klass:4} ({100*klass/dets:4.1f}%) | "
          f"rejeitada {rejected:4} ({100*rejected/dets:4.1f}%)")
    print(f"  precisou intervencao {dets-kept:5} ({100*(dets-kept)/dets:5.1f}%)")
    print(f"  adicionadas {added:5} ({100*added/gt:.1f}% do gabarito)")
    print(f"  tp={tp} fp={fp} fn={fn}  precisao={100*tp/(tp+fp):.1f}%  recall={100*tp/(tp+fn):.1f}%")


audit(False, "RAMO A — mantem as 4 caixas (regua 735)")
audit(True, "RAMO B — exclui airplane/train (regua 731)")

print("\n--- as 4 caixas em questao ---")
for f in test:
    for g in f["ground_truth"]:
        if g["class"] in EXCL:
            ref = next((d for d in f["stored_detections"] if d["det_id"] == g["ref_det_id"]), None)
            print(f"  {f['acq_id']} sec={f['sec']:4} classe={g['class']:9} origem={g['origin']:9} "
                  f"| deteccao original: classe={ref['class'] if ref else '-':9} conf={ref['conf'] if ref else '-'}")
