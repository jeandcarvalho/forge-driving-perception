"""Writer COCO do gabarito humano (`annotations/instances_test.json`).

`category_id` == índice YOLO de 80 classes (0=person … 79=toothbrush), NÃO o
category_id 1..90 do COCO oficial. O Forge escreve tanto o gabarito quanto as
predições, então o requisito é consistência, e um espaço só evita a conversão
que erra sinal. Ver config.COCO_CATEGORY_ID_IS_YOLO_INDEX.

Só frame com gabarito entra. Frame não revisado não tem "zero anotações" — ele
não tem informação, e um `images[]` sem `annotations` correspondente diria ao
pycocotools que a imagem foi conferida e está vazia.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from forge.api.models import Frame
from forge.config import EXPECTED_IMAGE_SIZE, GT_EXCLUDED_CLASSES
from forge.core.classes import COCO_CLASSES, class_id, is_valid_class, normalize_class
from forge.core.geometry import is_valid_box_n
from forge.data.layout import DatasetLayout


@dataclass(slots=True)
class CocoWriteResult:
    path: Path
    images: int
    annotations: int
    empty_images: int
    excluded_unreviewed: int
    dropped_boxes: int
    #: Instâncias tiradas por GT_EXCLUDED_CLASSES (fora do escopo de classe do gabarito).
    excluded_class: int = 0


def frame_image_id(acq_id: str, sec: int) -> int:
    """id inteiro estável e reversível: acq_id (14 díg.) * 100000 + sec.

    Determinístico entre execuções — hash de string mudaria a cada run e as
    predições deixariam de casar com o gabarito.
    """
    return int(acq_id) * 100_000 + sec


def write_coco(
    layout: DatasetLayout,
    frames: list[Frame],
    split: str,
    *,
    dims_by_key: dict[str, tuple[int, int]] | None = None,
) -> CocoWriteResult:
    """`dims_by_key` são as dimensões MEDIDAS no arquivo (verify.py).

    Sem elas cai no declarado pelo manifest e, na falta, em 1920x1080 — mas a
    caixa em px sai errada se a imagem real for outra, então o `pull` sempre
    passa o medido.
    """
    dims_by_key = dims_by_key or {}

    images: list[dict] = []
    annotations: list[dict] = []
    empty_images = 0
    excluded = 0
    dropped = 0
    excluded_class = 0
    ann_id = 1

    for f in frames:
        if f.split != split:
            continue
        if not f.is_labeled:
            excluded += 1
            continue

        w, h = dims_by_key.get(f.key) or (
            (f.image.width, f.image.height) if f.image and f.image.width and f.image.height
            else EXPECTED_IMAGE_SIZE
        )
        img_id = frame_image_id(f.acq_id, f.sec)
        images.append(
            {
                "id": img_id,
                "file_name": f"{f.acq_id}/{f.sec:06d}.jpg",
                "width": w,
                "height": h,
                "acq_id": f.acq_id,
                "sec": f.sec,
                "split": f.split,
            }
        )

        n_before = len(annotations)
        for g in f.ground_truth or []:
            if not is_valid_class(g.cls_name):
                dropped += 1
                continue
            # Fora do escopo do gabarito, não classe rara — ver GT_EXCLUDED_CLASSES.
            if normalize_class(g.cls_name or "") in GT_EXCLUDED_CLASSES:
                excluded_class += 1
                continue
            box = g.box_n.to_box() if g.box_n is not None else None
            if not is_valid_box_n(box):
                dropped += 1
                continue
            bbox = box.to_coco_xywh(w, h)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id(g.cls_name),
                    "bbox": [round(v, 3) for v in bbox],
                    "area": round(bbox[2] * bbox[3], 3),
                    "iscrowd": 0,
                    # Proveniência da instância: veio da pré-anotação confirmada,
                    # corrigida, ou foi desenhada do zero pelo curador.
                    "origin": g.origin,
                    "det_id": g.det_id,
                }
            )
            ann_id += 1
        if len(annotations) == n_before:
            empty_images += 1

    doc = {
        "info": {
            "description": "CarCará — gabarito humano (curadoria manual assistida)",
            "split": split,
            "generated_by": "carcara-forge",
            "date_created": datetime.now(UTC).isoformat(),
            "category_id_space": "yolo80",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": i, "name": name, "supercategory": "coco80"}
            for i, name in enumerate(COCO_CLASSES)
        ],
    }

    path = layout.coco_path(split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")

    return CocoWriteResult(
        path=path,
        images=len(images),
        annotations=len(annotations),
        empty_images=empty_images,
        excluded_unreviewed=excluded,
        dropped_boxes=dropped,
        excluded_class=excluded_class,
    )
