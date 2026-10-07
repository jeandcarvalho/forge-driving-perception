"""Fluxo reordenado: YOLO → proponente → VLM classificador.

O desenho de hoje (20/08/2026) separa as peças por competência, depois que as
duas fraquezas do VLM ficaram medidas:

  * ele não desenha caixa       → quem propõe geometria é um detector
  * ele inunda quando enumera   → ele nunca enumera, só responde sim/não

**Por que o proponente é o YOLO11x e não um segmentador.** O bloco `semseg` do
CarCará são quatro escalares por frame (building/vegetation/sidewalk %), não
contorno — não há máscara nenhuma no schema nem em disco. Com o papel vago,
medi o candidato natural, e ele ganha por larga margem:

    fonte de candidato          recuperou das 254    candidatos por acerto
    VLM enumerando (Fase 2)         137 (53,9%)              11,2
    YOLO11x fresco @0.25            230 (90,6%)               2,4

O que o humano teve de desenhar na mão já era visível para o YOLO11x no MESMO
corte de confiança — as detecções guardadas no Hub vieram de um passe mais fraco
(tracking sobre vídeo, corte ~0.2). O buraco de recall era de limiar, não de
percepção. E máscara não compraria nada aqui: a Table 6 pontua caixa, e um
contorno viraria bounding box antes do IoU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge.api.models import Frame, StoredDetection
from forge.config import PARITY_IOU, VLM_RECALL_DEDUP_IOU
from forge.core.classes import normalize_class
from forge.core.geometry import BoxN, iou

#: As 6 classes do gabarito. Fora disto o gabarito não tem instância, então uma
#: proposta correta de `traffic light` seria contada como erro — mediria a
#: lacuna do gabarito, não o modelo.
FLOW_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")


@dataclass(slots=True)
class Candidate:
    """Região proposta pelo detector que nenhuma caixa do Hub já cobria."""

    acq_id: str
    sec: int
    index: int
    det_class: str
    conf: float
    box: BoxN
    #: Preenchido pela fila binária do VLM.
    vlm_present: bool | None = None
    vlm_class: str | None = None
    reason: str = ""
    error: str | None = None

    @property
    def key(self) -> str:
        return f"{self.acq_id}:{self.sec}:c{self.index}"


# ── gabarito resolvido, por detecção ─────────────────────────────────────────


def resolved_truth(det: StoredDetection) -> tuple[str, BoxN] | None:
    """A instância do gabarito que ESTA detecção representa, ou None.

    Espelha `resolveGroundTruth` da API: `kept` fica com a caixa original,
    `corrected` com a correção humana, `rejected` não vira instância nenhuma.
    """
    if det.verdict == "rejected":
        return None
    if det.correction is not None:
        cls, box = det.correction.cls_name, det.correction.box_n
    else:
        cls, box = det.cls_name, det.box_n
    if box is None or not isinstance(cls, str):
        return None
    return normalize_class(cls), box.to_box()


def truth_index(frames: list[Frame]) -> dict[str, tuple[str, BoxN]]:
    """`acq:sec:index` → (classe, caixa) do gabarito, para as detecções."""
    out: dict[str, tuple[str, BoxN]] = {}
    for f in frames:
        for i, det in enumerate(f.stored_detections):
            t = resolved_truth(det)
            if t is not None:
                out[f"{f.acq_id}:{f.sec}:{i}"] = t
    return out


# ── geração de candidatos ────────────────────────────────────────────────────


def existing_boxes(frame: Frame) -> list[BoxN]:
    """Caixas que o Hub já propôs — o que o candidato NÃO pode duplicar."""
    return [d.box_n.to_box() for d in frame.stored_detections if d.box_n is not None]


def build_candidates(
    detections: dict[str, list[tuple[str, float, BoxN]]],
    frames: list[Frame],
    *,
    dedup_iou: float = VLM_RECALL_DEDUP_IOU,
) -> list[Candidate]:
    """Detecções frescas menos o que o Hub já cobria."""
    out: list[Candidate] = []
    for f in frames:
        key = f"{f.acq_id}:{f.sec}"
        exist = existing_boxes(f)
        n = 0
        for cls, conf, box in detections.get(key, []):
            if cls not in FLOW_CLASSES:
                continue
            if any(iou(box, e) >= dedup_iou for e in exist):
                continue
            out.append(
                Candidate(
                    acq_id=f.acq_id, sec=f.sec, index=n,
                    det_class=cls, conf=conf, box=box,
                )
            )
            n += 1
    return out


# ── geometria corrigida ──────────────────────────────────────────────────────


def best_overlap(box: BoxN, pool: list[tuple[str, float, BoxN]]) -> BoxN | None:
    """A detecção fresca que mais sobrepõe a região, ou None se nenhuma toca.

    É a peça que faltava: quando o VLM diz que a caixa precisa de correção, ele
    não devolve coordenada — então a coordenada vem daqui. Exige sobreposição
    real (> 0), senão estaríamos colando uma caixa qualquer do outro lado da
    cena em cima de um veredito de correção.
    """
    best, best_iou = None, 0.0
    for _cls, _conf, b in pool:
        v = iou(box, b)
        if v > best_iou:
            best, best_iou = b, v
    return best


# ── pontuação ────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class FlowScore:
    label: str
    gt: int
    tp: int = 0
    boxes: int = 0
    detail: dict[str, int] = field(default_factory=dict)

    def bump(self, key: str, n: int = 1) -> None:
        self.detail[key] = self.detail.get(key, 0) + n

    @property
    def fp(self) -> int:
        return self.boxes - self.tp

    @property
    def fn(self) -> int:
        return self.gt - self.tp

    @property
    def precision(self) -> float | None:
        return round(100 * self.tp / self.boxes, 1) if self.boxes else None

    @property
    def recall(self) -> float | None:
        return round(100 * self.tp / self.gt, 1) if self.gt else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label, "boxes": self.boxes, "tp": self.tp,
            "fp": self.fp, "fn": self.fn, "ground_truth": self.gt,
            "precision": self.precision, "recall": self.recall,
            "detail": dict(sorted(self.detail.items())),
        }


def box_is_correct(label: str, box: BoxN, truth: tuple[str, BoxN] | None) -> bool:
    """Regra única de acerto, e ela GENERALIZA a Table 6 em vez de competir.

    Uma caixa entregue conta como TP quando o rótulo bate com o do gabarito E a
    geometria está a IoU ≥ 0.7 da caixa do gabarito. Confira nos cinco casos da
    régua original, com a caixa e o rótulo intactos:

      kept              → gabarito é a própria caixa, IoU 1.0     → TP  ✔
      corrected-minor   → IoU(orig, correção) ≥ 0.7 por definição → TP  ✔
      corrected-redrawn → IoU < 0.7 por definição                 → FP  ✔
      class-changed     → rótulo difere do corrigido              → FP  ✔
      rejected          → não há instância de gabarito            → FP  ✔

    Reproduz 329 TP / 172 FP. A vantagem é que, ao contrário da contagem por
    veredito, ela continua valendo quando a geometria MUDA — que é exatamente o
    que a substituição por caixa do YOLO11x faz.
    """
    if truth is None:
        return False
    cls, tbox = truth
    return label == cls and iou(box, tbox) >= PARITY_IOU


# ── schema e prompt da fila binária ──────────────────────────────────────────

#: Pergunta fechada. O VLM não desenha e não enumera: ele olha UMA região que o
#: detector já recortou e diz se há mesmo um objeto ali, e qual. É o único modo
#: em que ele foi medido como competente (kept 83,7% / rejected 87,5%).
BINARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "present": {"type": "boolean"},
        "class": {"type": "string", "enum": list(FLOW_CLASSES)},
        "reason": {"type": "string"},
    },
    # `class` é obrigatório: fora de `required` o modelo simplesmente o omite, e
    # o papel dele neste fluxo é justamente classificar. Quando `present` é
    # falso o rótulo é ignorado — custa um token e evita um campo vazio.
    "required": ["present", "class", "reason"],
    "additionalProperties": False,
}

_BINARY_TASK = """A detector proposed a region in this driving scene, outlined in pink. You get the full frame and a zoomed crop of the same region.

Answer one question: is there really an object of one of these classes inside that region?
{classes}

- present = true  → yes, one of those classes is really there. Also give which one.
- present = false → no. The region is background, road surface, a shadow, a reflection, part of a building, or an object of some other kind.

Be strict: the object must actually be visible inside the outlined region, not merely nearby. But do not reject an object just because it is small or distant — distant vehicles are still vehicles."""


def binary_prompt(cand: Candidate) -> str:
    b = cand.box
    return (
        f"{_BINARY_TASK.format(classes=', '.join(FLOW_CLASSES))}\n\n"
        f"Detector guess: {cand.det_class} (confidence {cand.conf:.2f})\n"
        f"Region (normalized x1,y1,x2,y2): "
        f"[{b.x1:.3f}, {b.y1:.3f}, {b.x2:.3f}, {b.y2:.3f}]"
    )
