"""Caixas e IoU.

Coordenada canônica é normalizada (0..1), igual ao banco: px é derivado, nunca o
contrário. Só convertemos para px na hora de escrever COCO, e usando a dimensão
*medida* na imagem — não a que veio no manifest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BoxN:
    """Caixa normalizada em 0..1, canto superior-esquerdo → inferior-direito."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def to_px(self, img_w: int, img_h: int) -> tuple[float, float, float, float]:
        """xyxy absoluto."""
        return (self.x1 * img_w, self.y1 * img_h, self.x2 * img_w, self.y2 * img_h)

    def to_coco_xywh(self, img_w: int, img_h: int) -> list[float]:
        """[x, y, w, h] absoluto — formato de bbox do COCO."""
        x1, y1, x2, y2 = self.to_px(img_w, img_h)
        return [x1, y1, x2 - x1, y2 - y1]

    def to_yolo_cxcywh(self) -> tuple[float, float, float, float]:
        """Centro + tamanho, normalizado — formato de label do ultralytics."""
        return (
            (self.x1 + self.x2) / 2.0,
            (self.y1 + self.y2) / 2.0,
            self.width,
            self.height,
        )

    @classmethod
    def from_yolo_cxcywh(cls, cx: float, cy: float, w: float, h: float) -> BoxN:
        return cls(cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)

    @classmethod
    def from_px_xyxy(cls, x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> BoxN:
        return cls(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h)


def is_valid_box_n(b: BoxN | None) -> bool:
    """Mesma validação do `isValidBoxN` em curation/shared.ts."""
    if b is None:
        return False
    vals = (b.x1, b.y1, b.x2, b.y2)
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        return False
    return 0 <= b.x1 < b.x2 <= 1 and 0 <= b.y1 < b.y2 <= 1


def iou(a: BoxN, b: BoxN) -> float:
    """Porte literal de `iou()` em curation/stats.ts.

    Mesma aritmética, mesma ordem de operações — a paridade numérica depende
    disso, então não "melhore" esta função.
    """
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
    return inter / union if union > 0 else 0.0
