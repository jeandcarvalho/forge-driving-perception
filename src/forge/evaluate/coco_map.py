"""mAP via pycocotools.

Nota sobre o mascaramento de classe (config.EvalClassScope)
──────────────────────────────────────────────────────────
mAP é uma média DE APs POR CATEGORIA. Categoria sem nenhuma instância no gabarito
recebe AP = -1 no pycocotools e já é excluída da média — ou seja, uma detecção de
`traffic light` num dataset sem `traffic light` anotado não derruba o mAP, ela
apenas some.

Então o escopo de classe quase não move o mAP: ele decide quais categorias entram
na média (aqui, explicitamente, via `params.catIds`), e é na decomposição de erro
do matcher que a escolha realmente aparece — lá a detecção fora do gabarito ou é
descartada ou vira FP de fundo.

Deixar isso explícito importa porque a leitura ingênua ("mascarei, logo inflei o
mAP") está errada, e a oposta ("não mascarei, logo fui justo") também.
"""

from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass, field
from pathlib import Path

from forge.config import EvalClassScope
from forge.core.classes import class_id, class_name


@dataclass(slots=True)
class MapResult:
    map_50_95: float
    map_50: float
    map_75: float
    map_small: float
    map_medium: float
    map_large: float
    per_class: dict[str, float] = field(default_factory=dict)
    categories_evaluated: list[str] = field(default_factory=list)
    n_predictions: int = 0

    def as_dict(self) -> dict:
        return {
            "mAP@50:95": round(self.map_50_95, 5),
            "mAP@50": round(self.map_50, 5),
            "mAP@75": round(self.map_75, 5),
            "mAP@50:95_small": round(self.map_small, 5),
            "mAP@50:95_medium": round(self.map_medium, 5),
            "mAP@50:95_large": round(self.map_large, 5),
            "per_class_AP@50:95": {k: round(v, 5) for k, v in self.per_class.items()},
            "categories_evaluated": self.categories_evaluated,
            "n_predictions": self.n_predictions,
        }


def evaluate_map(
    gt_json: Path,
    predictions_json: Path,
    *,
    class_scope: EvalClassScope = EvalClassScope.GT_PRESENT,
) -> MapResult:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    preds = json.loads(Path(predictions_json).read_text(encoding="utf-8"))
    if not preds:
        raise ValueError(f"{predictions_json} não tem nenhuma predição")

    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(gt_json))

    gt_cat_ids = sorted({a["category_id"] for a in coco_gt.dataset["annotations"]})

    if class_scope is EvalClassScope.GT_PRESENT:
        cat_ids = gt_cat_ids
        preds = [p for p in preds if p["category_id"] in set(gt_cat_ids)]
        if not preds:
            raise ValueError(
                "depois do mascaramento não sobrou predição nenhuma nas classes do "
                f"gabarito ({[class_name(c) for c in gt_cat_ids]})"
            )
    else:
        cat_ids = sorted({c["id"] for c in coco_gt.dataset["categories"]})

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.catIds = cat_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    # precision: [T, R, K, A, M] — T limiares de IoU, K categorias, A área, M maxDets
    precision = ev.eval["precision"]
    per_class: dict[str, float] = {}
    for k, cid in enumerate(cat_ids):
        p = precision[:, :, k, 0, 2]
        valid = p[p > -1]
        if valid.size:
            per_class[class_name(cid)] = float(valid.mean())

    return MapResult(
        map_50_95=float(ev.stats[0]),
        map_50=float(ev.stats[1]),
        map_75=float(ev.stats[2]),
        map_small=float(ev.stats[3]),
        map_medium=float(ev.stats[4]),
        map_large=float(ev.stats[5]),
        per_class=dict(sorted(per_class.items(), key=lambda kv: -kv[1])),
        categories_evaluated=[class_name(c) for c in cat_ids],
        n_predictions=len(preds),
    )


def predictions_to_matcher_input(
    predictions_json: Path, gt_json: Path
) -> tuple[list, list]:
    """Converte COCO results + COCO GT no par (predições, gabarito) do matcher."""
    from forge.core.geometry import BoxN
    from forge.evaluate.matcher import GtInstance, Prediction

    gt_doc = json.loads(Path(gt_json).read_text(encoding="utf-8"))
    dims: dict[int, tuple[int, int]] = {}
    key_by_id: dict[int, str] = {}
    for im in gt_doc["images"]:
        dims[im["id"]] = (im["width"], im["height"])
        key_by_id[im["id"]] = f"{im['acq_id']}:{im['sec']}"

    def to_box(image_id: int, bbox: list[float]) -> BoxN:
        w, h = dims[image_id]
        x, y, bw, bh = bbox
        return BoxN(x / w, y / h, (x + bw) / w, (y + bh) / h)

    gts = [
        GtInstance(
            key_by_id[a["image_id"]],
            class_name(a["category_id"]),
            to_box(a["image_id"], a["bbox"]),
            a.get("origin"),
            a.get("det_id"),
        )
        for a in gt_doc["annotations"]
    ]

    preds_doc = json.loads(Path(predictions_json).read_text(encoding="utf-8"))
    preds = [
        Prediction(
            key_by_id[p["image_id"]],
            class_name(p["category_id"]),
            float(p["score"]),
            to_box(p["image_id"], p["bbox"]),
        )
        for p in preds_doc
        if p["image_id"] in key_by_id
    ]
    return preds, gts


__all__ = ["MapResult", "class_id", "evaluate_map", "predictions_to_matcher_input"]
