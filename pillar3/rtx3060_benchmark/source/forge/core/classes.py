"""Espaço de classes — 80 classes COCO, indexação única de ponta a ponta.

REGRA DURA: o índice é o índice COCO/YOLO de 80 classes (0=person … 79=toothbrush),
o mesmo dos pesos pré-treinados do ultralytics. Nunca remapeie para o subconjunto
que aparece no gabarito (car, truck, bus, motorcycle, person, bicycle, train,
airplane), por mais tentador que seja um 0..7 contíguo.

Motivo: o Experimento 2 compara YOLO11n cru (pesos COCO) com YOLO11n treinado.
Remapear troca a cabeça do modelo e os dois deixam de ser comparáveis — some o
resultado do paper. O custo de manter 80 é uma cabeça com 72 classes que nunca
disparam; o custo de remapear é o experimento.

A ordem desta lista É a indexação. Não reordene.
"""

from __future__ import annotations

COCO_CLASSES: tuple[str, ...] = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)

assert len(COCO_CLASSES) == 80, "espaço de classes corrompido"

_INDEX_BY_NAME: dict[str, int] = {name: i for i, name in enumerate(COCO_CLASSES)}

# Classes efetivamente presentes no gabarito da coleção-alvo (735 instâncias).
# NÃO é o espaço de treino — é o escopo de *avaliação* quando o mascaramento
# está ligado (ver EvalClassScope em config.py). Deriva do dado, não é chutada:
# `forge audit` recalcula e avisa se divergir.
GT_PRESENT_CLASSES: frozenset[str] = frozenset(
    {"car", "truck", "bus", "motorcycle", "person", "bicycle", "train", "airplane"}
)


def normalize_class(name: str) -> str:
    """Porte de `normalizeClass` em curation/shared.ts."""
    return name.strip().lower()


def is_valid_class(name: object) -> bool:
    """Porte de `isValidClass` em curation/shared.ts."""
    return isinstance(name, str) and normalize_class(name) in _INDEX_BY_NAME


def class_id(name: str) -> int:
    """Nome → índice COCO/YOLO de 80 classes."""
    key = normalize_class(name)
    try:
        return _INDEX_BY_NAME[key]
    except KeyError:
        raise ValueError(f"classe fora do espaço COCO de 80: {name!r}") from None


def class_name(idx: int) -> str:
    if not 0 <= idx < 80:
        raise ValueError(f"índice de classe fora de 0..79: {idx}")
    return COCO_CLASSES[idx]
