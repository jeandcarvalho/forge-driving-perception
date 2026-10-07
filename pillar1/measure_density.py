"""Densidade da COLECAO COMPLETA sob o mesmo rotulador.

MEDICAO, nao treino: nao escreve lista de split nenhuma, nao toca em
labels/auto/. So conta. A guarda de `discarded` no CLI continua valendo — nada
aqui vira material treinavel.

Reaproveita os rotulos ja existentes do split `train` (labels/auto/yolo11x_c025)
e roda o YOLO11x nos frames de `test` e `discarded`, com o MESMO corte.
"""
import json
from pathlib import Path

from ultralytics import YOLO

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
OUT = Path(os.environ.get("FORGE_OUT", "out"))
CONF, IOU, IMGSZ = 0.25, 0.7, 640

man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
by_split = {"test": [], "train": [], "discarded": []}
for f in man["frames"]:
    p = ROOT / "images" / f["acq_id"] / f"{f['sec']:06d}.jpg"
    if p.exists():
        by_split[f["split"]].append(p)
print({k: len(v) for k, v in by_split.items()})

# --- 1. train: reaproveita o que ja esta em disco -------------------------
train_dir = ROOT / "labels" / "auto" / "yolo11x_c025"
train_objs = train_empty = 0
for p in by_split["train"]:
    txt = train_dir / f"{p.parent.name}__{p.stem.lstrip('0') or '0'}.txt"
    if not txt.exists():
        cand = list(train_dir.glob(f"{p.parent.name}__*{int(p.stem)}.txt"))
        txt = cand[0] if cand else None
    if txt is None or not txt.exists():
        print(f"  SEM ROTULO: {p}")
        continue
    n = len([ln for ln in txt.read_text().splitlines() if ln.strip()])
    train_objs += n
    train_empty += n == 0
print(f"train reaproveitado: {len(by_split['train'])} frames, {train_objs} objetos, {train_empty} vazios")

# --- 2. test + discarded: roda o professor --------------------------------
model = YOLO(os.environ.get("FORGE_YOLO_WEIGHTS", "yolo11x.pt"))
results = {}
for split in ("test", "discarded"):
    paths = by_split[split]
    listfile = OUT / f"_measure_{split}.txt"
    listfile.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    objs = empty = 0
    for r in model.predict(source=str(listfile), conf=CONF, iou=IOU, imgsz=IMGSZ,
                           device=0, batch=8, stream=True, verbose=False):
        n = 0 if r.boxes is None else len(r.boxes)
        objs += n
        empty += n == 0
    results[split] = (len(paths), objs, empty)
    print(f"{split}: {len(paths)} frames, {objs} objetos, {empty} vazios")

n_tr = len(by_split["train"])
tot_frames = n_tr + sum(v[0] for v in results.values())
tot_objs = train_objs + sum(v[1] for v in results.values())
tot_empty = train_empty + sum(v[2] for v in results.values())
print("\n=== COLECAO COMPLETA ===")
print(f"frames={tot_frames}  objetos={tot_objs}  obj/frame={tot_objs/tot_frames:.2f}  "
      f"vazios={tot_empty} ({100*tot_empty/tot_frames:.1f}%)")
(OUT / "density_collection.json").write_text(json.dumps(
    {"frames": tot_frames, "objects": tot_objs, "empty": tot_empty,
     "per_split": {"train": [n_tr, train_objs, train_empty], **{k: list(v) for k, v in results.items()}}},
    indent=2), encoding="utf-8")
