"""Inferência local com ultralytics → predições em COCO results.

Por que sempre local, e nunca o `yolo[]` do banco: aquele array saiu de um passe
de tracking sobre vídeo, com a confiança já cortada em ~0.2 e sem registro de
imgsz, hash dos pesos, IoU de NMS ou versão. Sem a cauda de baixa confiança não
existe curva precision-recall e, portanto, não existe mAP.

Daí `conf=0.001`: não é exagero, é o que faz a curva existir.

O índice de classe é resolvido por NOME (`model.names` → `class_id`), nunca por
posição. Assim um modelo com cabeça diferente falha alto em vez de produzir
silenciosamente predições com a classe trocada.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from forge.config import INFER_CONF, INFER_IMGSZ, INFER_NMS_IOU
from forge.core.classes import class_id, is_valid_class
from forge.data.writers.coco import frame_image_id
from forge.detect.source import source_list


@dataclass(slots=True)
class InferResult:
    predictions_path: Path
    images: int
    detections: int
    classes_seen: dict[str, int]
    unknown_classes: list[str]


def _image_id_from_path(path: Path) -> int:
    """images/<acq_id>/<sec:06d>.jpg → o mesmo id do instances_*.json."""
    return frame_image_id(path.parent.name, int(path.stem))


def run_inference(
    weights: str | Path,
    images: list[Path],
    out_path: Path,
    *,
    conf: float = INFER_CONF,
    iou: float = INFER_NMS_IOU,
    imgsz: int = INFER_IMGSZ,
    device: str | int | None = None,
    batch: int = 16,
    progress: bool = True,
) -> InferResult:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names: dict[int, str] = model.names

    # Falha antes de gastar GPU: se a cabeça não for o espaço COCO de 80, as
    # predições não seriam comparáveis com o gabarito.
    unknown = sorted({n for n in names.values() if not is_valid_class(n)})
    if unknown:
        raise ValueError(
            f"os pesos {weights} têm classes fora do espaço COCO de 80: {unknown[:10]}. "
            "O Forge mantém indexação COCO de ponta a ponta (ver core/classes.py)."
        )

    results_json: list[dict] = []
    classes_seen: dict[str, int] = {}

    # `.txt` e não lista de caminhos: com uma lista o ultralytics ignora `batch`
    # e monta um batch único do conjunto inteiro. Ver detect/source.py.
    n_images = 0
    with source_list(images) as source:
        stream = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            stream=True,
            verbose=False,
            batch=batch,
        )

        for res in tqdm(
            stream, total=len(images), disable=not progress, unit="img", desc="inferindo"
        ):
            n_images += 1
            img_path = Path(res.path)
            image_id = _image_id_from_path(img_path)
            boxes = res.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy.cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            clss = boxes.cls.cpu().tolist()
            for (x1, y1, x2, y2), score, cls_idx in zip(xyxy, confs, clss, strict=True):
                name = names[int(cls_idx)]
                cid = class_id(name)  # por nome, não por posição
                classes_seen[name] = classes_seen.get(name, 0) + 1
                results_json.append(
                    {
                        "image_id": image_id,
                        "category_id": cid,
                        "bbox": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                        "score": round(float(score), 6),
                    }
                )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results_json, ensure_ascii=False), encoding="utf-8")

    return InferResult(
        predictions_path=out_path,
        images=n_images,
        detections=len(results_json),
        classes_seen=dict(sorted(classes_seen.items(), key=lambda kv: -kv[1])),
        unknown_classes=[],
    )
