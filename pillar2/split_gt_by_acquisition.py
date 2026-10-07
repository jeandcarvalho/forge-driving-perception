"""Divide o gabarito humano em treino/teste POR GRAVAÇÃO, sem vazamento.

Os 313 frames curados viram ~2/3 para treinar e ~1/3 para medir. O corte é por
aquisição (a mesma regra de antivazamento da plataforma): nenhuma gravação
aparece dos dois lados, senão o frame vizinho — mesma estrada, 2 segundos depois
— faria o teste medir memorização.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# Roda avulso: `python scripts/split_gt_by_acquisition.py`, sem instalar o pacote.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from forge.api.models import Frame
from forge.data.layout import DatasetLayout
from forge.data.writers.coco import write_coco

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
TAG = "gt_manual"
VAL_SPLIT = "gtval"
SEED = 42
TRAIN_FRACTION = 2 / 3


def mulberry32(seed: int):
    a = seed & 0xFFFFFFFF

    def rnd() -> float:
        nonlocal a
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = a
        t = (t ^ (t >> 15)) * (1 | t) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t) & 0xFFFFFFFF)) & 0xFFFFFFFF ^ t
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rnd


def seeded_shuffle(items: list, seed: int) -> list:
    rnd = mulberry32(seed)
    out = list(items)
    for i in range(len(out) - 1, 0, -1):
        j = int(rnd() * (i + 1))
        out[i], out[j] = out[j], out[i]
    return out


layout = DatasetLayout(root=ROOT)
doc = json.loads(layout.manifest_path.read_text(encoding="utf-8"))
frames = [Frame.model_validate(f) for f in doc["frames"] if f["split"] == "test"]

acqs = sorted({f.acq_id for f in frames})
shuffled = seeded_shuffle(acqs, SEED)
n_train = round(len(shuffled) * TRAIN_FRACTION)
train_acqs = set(shuffled[:n_train])

train = [f for f in frames if f.acq_id in train_acqs]
val = [f for f in frames if f.acq_id not in train_acqs]

print(f"gravações: {len(train_acqs)} treino · {len(acqs) - len(train_acqs)} teste")
print(f"frames:    {len(train)} treino · {len(val)} teste")
inst_t = sum(len(f.ground_truth or []) for f in train)
inst_v = sum(len(f.ground_truth or []) for f in val)
print(f"instâncias (antes da exclusão de classe): {inst_t} treino · {inst_v} teste")

# ── lado do treino: rótulos humanos entram como se fossem "auto", para o
#    `forge train --tag` consumir sem nenhum caminho especial ─────────────────
auto_dir = layout.auto_labels_dir(TAG)
auto_dir.mkdir(parents=True, exist_ok=True)
train_imgs = []
for f in train:
    src = layout.label_path("test", f.acq_id, f.sec)
    img = layout.image_path(f.acq_id, f.sec)
    if not src.is_file() or not img.is_file():
        continue
    shutil.copyfile(src, auto_dir / f"{f.acq_id}__{f.sec}.txt")
    train_imgs.append(img)
layout.auto_split_list_path(TAG).write_text(
    "\n".join(str(p.resolve()) for p in train_imgs) + "\n", encoding="utf-8"
)

# ── lado do teste: vira um split próprio, com rótulos, lista e COCO ──────────
val_lbl_dir = layout.labels_dir / VAL_SPLIT
val_lbl_dir.mkdir(parents=True, exist_ok=True)
val_imgs = []
for f in val:
    src = layout.label_path("test", f.acq_id, f.sec)
    img = layout.image_path(f.acq_id, f.sec)
    if not src.is_file() or not img.is_file():
        continue
    shutil.copyfile(src, layout.label_path(VAL_SPLIT, f.acq_id, f.sec))
    val_imgs.append(img)
layout.split_list_path(VAL_SPLIT, trainable=True).write_text(
    "\n".join(str(p.resolve()) for p in val_imgs) + "\n", encoding="utf-8"
)

# O COCO é escrito por `frame.split`, então renomeia o split das cópias.
res = write_coco(layout, [f.model_copy(update={"split": VAL_SPLIT}) for f in val], VAL_SPLIT)
print(f"COCO do teste: {res.images} imagens, {res.annotations} instancias "
      f"({res.excluded_class} tiradas por classe excluida)")
print(f"treino: {len(train_imgs)} imagens em {layout.auto_split_list_path(TAG).name}")
print(f"teste:  {len(val_imgs)} imagens em {layout.split_list_path(VAL_SPLIT, trainable=True).name}")
