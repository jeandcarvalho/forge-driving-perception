"""O que o VLM vê, e contra o que ele é medido.

Duas responsabilidades que andam juntas:

1. **Renderizar** a caixa proposta do jeito que o curador humano a viu — frame
   inteiro com a caixa desenhada, mais um recorte ampliado. A interface do Hub
   mostrava a cena e deixava aproximar; mandar só o recorte tiraria o contexto
   (é um carro ou o reflexo de um carro?) e mandar só o frame inteiro esconderia
   o objeto distante de 30 px.

2. **Derivar o veredito humano** das 5 categorias, que é o gabarito desta
   comparação. Isso NÃO é reimplementação: é a mesma aritmética de
   `evaluate/parity.py`, que por sua vez é porte 1:1 de `curation/stats.ts`.
   Se divergir, o `audit` denuncia.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from forge.api.models import Frame, StoredDetection
from forge.config import (
    GT_EXCLUDED_CLASSES,
    PARITY_IOU,
    VLM_CROP_MIN_PX,
    VLM_CROP_PAD,
    VLM_RECALL_MAX_ITEMS,
    VLM_VERDICTS,
)
from forge.core.classes import normalize_class
from forge.core.geometry import BoxN, iou

# ── veredito humano: a régua ─────────────────────────────────────────────────


def human_verdict(det: StoredDetection) -> str | None:
    """As 5 categorias do Hub para UMA detecção proposta.

    Devolve None para `unreviewed` — frame que ninguém olhou não tem veredito e
    não pode entrar na matriz.

    A separação de `corrected` em três é geométrica e sai do IoU entre a caixa
    ORIGINAL e a CORREÇÃO dela mesma (PARITY_IOU = 0.7): a pergunta é "o curador
    empurrou ou refez?", não "a predição achou o objeto?".
    """
    if det.verdict == "kept":
        return "kept"
    if det.verdict == "rejected":
        return "rejected"
    if det.verdict != "corrected" or det.correction is None:
        return None

    same_class = (
        isinstance(det.cls_name, str)
        and isinstance(det.correction.cls_name, str)
        and normalize_class(det.cls_name) == normalize_class(det.correction.cls_name)
    )
    if not same_class:
        return "class-changed"

    a, b = det.box_n, det.correction.box_n
    if a is not None and b is not None and iou(a.to_box(), b.to_box()) >= PARITY_IOU:
        return "corrected-minor"
    return "corrected-redrawn"


@dataclass(frozen=True, slots=True)
class BoxTask:
    """Uma caixa proposta a julgar, com o veredito humano já resolvido."""

    acq_id: str
    sec: int
    det_index: int
    det_id: str | None
    proposed_class: str
    conf: float | None
    box: BoxN
    truth: str

    @property
    def key(self) -> str:
        return f"{self.acq_id}:{self.sec}:{self.det_index}"


def box_tasks(frames: list[Frame]) -> list[BoxTask]:
    """Todas as caixas propostas em frames revisados, com veredito humano.

    São as 501 do pilar 2 — mesma população, mesma régua, para que a matriz de
    concordância seja comparável lado a lado com a tabela do `audit`.
    """
    out: list[BoxTask] = []
    for f in frames:
        if not f.reviewed:
            continue
        for i, det in enumerate(f.stored_detections):
            truth = human_verdict(det)
            if truth is None or det.box_n is None or not isinstance(det.cls_name, str):
                continue
            # Régua do paper: 731 instâncias / 501 detecções. As 4 caixas de
            # `airplane`/`train` saíram do gabarito por decisão humana em
            # 17/08/2026 (rótulo trocado, não classe rara) — GT_EXCLUDED_CLASSES.
            # Onde o veredito é `kept`, a classe resolvida É a proposta, então é
            # aqui que a exclusão morde. Sem isto a matriz sai em 505 e não
            # compara com a tabela publicada.
            if (
                truth == "kept"
                and normalize_class(det.cls_name) in GT_EXCLUDED_CLASSES
            ):
                continue
            out.append(
                BoxTask(
                    acq_id=f.acq_id,
                    sec=f.sec,
                    det_index=i,
                    det_id=det.det_id,
                    proposed_class=normalize_class(det.cls_name),
                    conf=det.conf,
                    box=det.box_n.to_box(),
                    truth=truth,
                )
            )
    return out


def human_added_boxes(frame: Frame) -> list[tuple[str, BoxN]]:
    """Caixas que o humano desenhou na mão (`origin == "added"`).

    É o alvo da Fase 2: são exatamente os objetos que a pré-anotação não propôs
    e que só existem porque uma pessoa os viu.
    """
    out: list[tuple[str, BoxN]] = []
    for g in frame.ground_truth or []:
        if g.origin != "added" or g.box_n is None or not isinstance(g.cls_name, str):
            continue
        out.append((normalize_class(g.cls_name), g.box_n.to_box()))
    return out


# ── renderização ─────────────────────────────────────────────────────────────

_OUTLINE = (255, 0, 128)
_SHADOW = (0, 0, 0)


def _px(box: BoxN, w: int, h: int) -> tuple[int, int, int, int]:
    return (
        max(0, round(box.x1 * w)),
        max(0, round(box.y1 * h)),
        min(w, round(box.x2 * w)),
        min(h, round(box.y2 * h)),
    )


def draw_box(img: Image.Image, box: BoxN, label: str) -> Image.Image:
    """Frame inteiro com a caixa em destaque. Não mutila o original."""
    out = img.copy()
    d = ImageDraw.Draw(out)
    x1, y1, x2, y2 = _px(box, out.width, out.height)
    # Contorno duplo: preto por fora, cor por dentro. Numa cena noturna ou num
    # asfalto claro uma cor só some, e aí o modelo julga a caixa errada.
    d.rectangle((x1 - 2, y1 - 2, x2 + 2, y2 + 2), outline=_SHADOW, width=5)
    d.rectangle((x1, y1, x2, y2), outline=_OUTLINE, width=3)
    ty = max(0, y1 - 22)
    d.rectangle((x1, ty, x1 + 9 * len(label) + 10, ty + 20), fill=_SHADOW)
    d.text((x1 + 5, ty + 4), label, fill=_OUTLINE)
    return out


def crop_around(img: Image.Image, box: BoxN) -> Image.Image:
    """Recorte ampliado ao redor da caixa, com contexto.

    O padding é proporcional ao lado maior da caixa (`VLM_CROP_PAD`), então
    caixa pequena ganha proporcionalmente mais entorno — que é justamente onde
    o contexto decide (uma mancha escura na calçada é um pedestre ou um poste?).
    """
    w, h = img.width, img.height
    x1, y1, x2, y2 = _px(box, w, h)
    side = max(x2 - x1, y2 - y1, 1)
    pad = int(side * VLM_CROP_PAD)
    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)

    # Recorte minúsculo vira mosaico ilegível — força um mínimo e reamostra.
    if (cx2 - cx1) < VLM_CROP_MIN_PX or (cy2 - cy1) < VLM_CROP_MIN_PX:
        need = VLM_CROP_MIN_PX
        mx, my = (cx1 + cx2) // 2, (cy1 + cy2) // 2
        cx1, cx2 = max(0, mx - need // 2), min(w, mx + need // 2)
        cy1, cy2 = max(0, my - need // 2), min(h, my + need // 2)

    cw, ch = max(cx2 - cx1, 1), max(cy2 - cy1, 1)
    # Posição da caixa DENTRO do recorte, normalizada. Calculada antes de
    # qualquer reamostragem: normalizado é invariante a resize, pixel não é.
    rel = BoxN((x1 - cx1) / cw, (y1 - cy1) / ch, (x2 - cx1) / cw, (y2 - cy1) / ch)

    crop = img.crop((cx1, cy1, cx2, cy2))
    if crop.width < VLM_CROP_MIN_PX:
        scale = VLM_CROP_MIN_PX / max(crop.width, 1)
        crop = crop.resize(
            (int(crop.width * scale), int(crop.height * scale)), Image.LANCZOS
        )
    # A caixa também vai desenhada no recorte: sem isso, com dois objetos juntos
    # o modelo julga o vizinho.
    return draw_box(crop, rel, "")


# ── schemas: a restrição que vale mais que qualquer instrução ────────────────

PHASE1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": list(VLM_VERDICTS)},
        "correct_class": {
            "type": "string",
            "description": "Só quando verdict == class-changed: a classe certa.",
        },
        "reason": {"type": "string"},
    },
    "required": ["verdict", "reason"],
    "additionalProperties": False,
}

#: ── Fase 1 v2 (20/08/2026) ──────────────────────────────────────────────────
#:
#: Duas mudanças, as duas por defeito medido na v1:
#:
#: 1. **Classe travada nas 6 e OBRIGATÓRIA.** Na v1 `correct_class` era string
#:    livre e ficava fora de `required` — resultado: as 7 vezes que o modelo
#:    disse `class-changed`, ele OMITIU a classe. O scoring caía no rótulo
#:    original e contava 0 acertos de correção. Era bug de schema, não do
#:    modelo: 6 dessas 7 caixas o humano também tinha relabelado.
#:
#: 2. **Pergunta decomposta em campos ordenados.** Na v1 havia um enum único de
#:    5 vereditos, e queixa de GEOMETRIA vazava para `rejected`: das 17 caixas
#:    boas que ele descartou, 4 diziam "a caixa cobre só um pedaço do veículo"
#:    — ou seja, ele via o objeto e mesmo assim removia. Aqui `object_present`
#:    vem primeiro e, uma vez afirmado, fecha a porta da rejeição; a qualidade
#:    da caixa é uma pergunta separada, depois.
#:
#: O `format` do llama.cpp gera a gramática na ORDEM declarada das propriedades,
#: e todas são obrigatórias — então a sequência é imposta pelo decodificador, não
#: pedida no texto. É a mesma ideia de duas chamadas, por 1x o custo.
PHASE1_SCHEMA_V2: dict[str, Any] = {
    "type": "object",
    "properties": {
        "object_present": {
            "type": "boolean",
            "description": "Há mesmo um objeto das 6 classes dentro da caixa?",
        },
        "object_class": {
            "type": "string",
            "enum": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
        },
        "box_quality": {
            "type": "string",
            "enum": ["tight", "needs_nudge", "badly_placed"],
        },
        "reason": {"type": "string"},
    },
    "required": ["object_present", "object_class", "box_quality", "reason"],
    "additionalProperties": False,
}


def verdict_from_v2(answer: dict[str, Any], proposed_class: str) -> tuple[str, str | None]:
    """Resposta decomposta → (veredito das 5 categorias, classe corrigida).

    A ordem de precedência é a de `parity.py`: classe primeiro, geometria depois.
    Lá, `corrected` só vira erro de caixa quando a classe bate; se a classe mudou,
    é `class-changed` independentemente do que aconteceu com a geometria.
    """
    if not answer.get("object_present"):
        return "rejected", None
    cls = answer.get("object_class")
    cls = normalize_class(cls) if isinstance(cls, str) else None
    if cls is not None and cls != proposed_class:
        return "class-changed", cls
    quality = answer.get("box_quality")
    if quality == "needs_nudge":
        return "corrected-minor", None
    if quality == "badly_placed":
        return "corrected-redrawn", None
    return "kept", None


PHASE2_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "missed": {
            "type": "array",
            # Teto duro no decodificador, não pedido no prompt: pedir "no máximo
            # N" em texto não segura o modelo, o schema segura.
            "maxItems": VLM_RECALL_MAX_ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "class": {"type": "string"},
                    "box": {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                },
                "required": ["class", "box", "confidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["missed"],
    "additionalProperties": False,
}
