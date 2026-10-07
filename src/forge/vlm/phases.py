"""As duas fases do A1.

Fase 1 — veredito por caixa existente. Para cada caixa que o YOLO11x propôs num
frame revisado, o VLM escolhe dentro das 5 categorias do Hub. Ele NÃO devolve
coordenada: geometria é a fraqueza dele, e a decisão de método é usá-lo onde ele
é forte (semântica) e não onde não é.

Fase 2 — recall. Frame inteiro, intacto, mais a lista em texto das regiões já
confirmadas; a pergunta é o que ficou de fora. Aqui ele PRECISA propor caixa, e
por isso o resultado é medido como **presença** (achou o objeto?) e não como
geometria (desenhou bem?). O teto de iterações é duro: "até não achar mais nada"
não termina, porque o modelo sempre acha mais uma coisa se você continuar
perguntando.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from forge.api.models import Frame
from forge.config import (
    VLM_RECALL_DEDUP_IOU,
    VLM_VERDICTS,
)
from forge.core.classes import COCO_CLASSES, normalize_class
from forge.core.geometry import BoxN, iou
from forge.vlm.client import VlmClient, VlmError, load_frame_image
from forge.vlm.prepare import (
    PHASE1_SCHEMA,
    PHASE1_SCHEMA_V2,
    PHASE2_SCHEMA,
    BoxTask,
    crop_around,
    draw_box,
    verdict_from_v2,
)

# ── Fase 1 ───────────────────────────────────────────────────────────────────

_TAXONOMY = """\
You are auditing an automatic object detector's output, exactly as a human \
annotator would. A detector proposed ONE box. Judge only that box.

You get two images: the full driving scene with the proposed box outlined in \
pink, and a zoomed crop around the same box.

Choose exactly one verdict:

- "kept": the class is right AND the box is tight around the object. No change \
needed.
- "corrected-minor": the class is right and the box is on the right object, but \
the edges need a small nudge (slightly loose, slightly cropped).
- "corrected-redrawn": the class is right and the object is there, but the box \
is badly placed or badly sized — it would have to be drawn again from scratch.
- "class-changed": there IS an object there and the box is reasonable, but the \
label is wrong (e.g. a truck labelled as a car). Give the correct class.
- "rejected": there is no such object there at all — the detector fired on \
background, a reflection, a shadow, or a duplicate of another box.

Be decisive. "kept" is a real answer and is common: if the box looks right, say \
kept rather than hedging toward a correction."""


_TAXONOMY_V2 = """You are auditing an automatic object detector, exactly as a human annotator would. A detector proposed ONE box, outlined in pink. You get the full driving scene and a zoomed crop of the same box.

Answer three things, in this order:

1. object_present — Is there really an object of one of these classes inside the outlined box?
   {classes}
   Answer false ONLY if there is no such object there at all: the detector fired on background, road surface, a reflection, glare, or a shadow.
   If a vehicle IS there, answer true — even if the box is a poor fit. A badly drawn box around a real object is still a real object; you will say so in the next field.

2. object_class — Which of those classes it actually is. Answer this whenever object_present is true. If the detector's guess is right, repeat it; if it is wrong, give the right one.

3. box_quality — How well the box fits that object:
   - tight        : the box hugs the object. No change needed.
   - needs_nudge  : right object, but the edges are slightly loose or slightly cropped.
   - badly_placed : right object, but the box is badly sized or badly positioned and would have to be drawn again.

Be decisive. "tight" is a real answer and is common."""


def phase1_prompt_v2(task: BoxTask) -> str:
    b = task.box
    conf = f"{task.conf:.2f}" if task.conf is not None else "unknown"
    return (
        f"{_TAXONOMY_V2.format(classes=', '.join(RECALL_CLASSES))}\n\n"
        f"Detector guess: {task.proposed_class} (confidence {conf})\n"
        f"Proposed box (normalized x1,y1,x2,y2): "
        f"[{b.x1:.3f}, {b.y1:.3f}, {b.x2:.3f}, {b.y2:.3f}]"
    )


def phase1_prompt(task: BoxTask) -> str:
    b = task.box
    conf = f"{task.conf:.2f}" if task.conf is not None else "unknown"
    return (
        f"{_TAXONOMY}\n\n"
        f"Proposed class: {task.proposed_class}\n"
        f"Detector confidence: {conf}\n"
        f"Proposed box (normalized x1,y1,x2,y2): "
        f"[{b.x1:.3f}, {b.y1:.3f}, {b.x2:.3f}, {b.y2:.3f}]\n\n"
        f"Give your verdict and a one-sentence reason."
    )


@dataclass(slots=True)
class Phase1Row:
    key: str
    acq_id: str
    sec: int
    det_index: int
    proposed_class: str
    conf: float | None
    truth: str
    vlm: str | None
    vlm_class: str | None
    reason: str
    error: str | None = None


def run_phase1(
    client: VlmClient,
    tasks: list[BoxTask],
    image_of: Callable[[str, int], Path],
    *,
    on_step: Callable[[int, int], None] | None = None,
    v2: bool = False,
) -> list[Phase1Row]:
    """Um veredito por caixa. Falha em uma caixa não derruba a rodada.

    `v2` usa o schema decomposto (existe objeto? → qual classe? → a caixa está
    boa?) com a classe travada nas 6. Ver PHASE1_SCHEMA_V2 para o porquê.
    """
    rows: list[Phase1Row] = []
    # Agrupa por frame para abrir cada JPEG uma vez só: são 501 caixas em 313
    # frames, então quase metade dos `open` seria repetida.
    by_frame: dict[tuple[str, int], list[BoxTask]] = {}
    for t in tasks:
        by_frame.setdefault((t.acq_id, t.sec), []).append(t)

    done = 0
    for (acq_id, sec), group in by_frame.items():
        img = load_frame_image(image_of(acq_id, sec))
        for task in group:
            scene = draw_box(img, task.box, task.proposed_class)
            crop = crop_around(img, task.box)
            row = Phase1Row(
                key=task.key,
                acq_id=acq_id,
                sec=sec,
                det_index=task.det_index,
                proposed_class=task.proposed_class,
                conf=task.conf,
                truth=task.truth,
                vlm=None,
                vlm_class=None,
                reason="",
            )
            try:
                if v2:
                    ans = client.ask(
                        phase1_prompt_v2(task), [scene, crop], PHASE1_SCHEMA_V2
                    )
                    verdict, cls = verdict_from_v2(ans, task.proposed_class)
                    row.vlm, row.vlm_class = verdict, cls
                    row.reason = str(ans.get("reason", ""))[:400]
                else:
                    ans = client.ask(phase1_prompt(task), [scene, crop], PHASE1_SCHEMA)
                    verdict = str(ans.get("verdict", "")).strip()
                    if verdict not in VLM_VERDICTS:
                        raise VlmError(f"veredito fora do enum: {verdict!r}")
                    row.vlm = verdict
                    row.reason = str(ans.get("reason", ""))[:400]
                    cc = ans.get("correct_class")
                    if verdict == "class-changed" and isinstance(cc, str):
                        row.vlm_class = normalize_class(cc)
            except VlmError as exc:
                row.error = str(exc)[:200]
            rows.append(row)
            done += 1
            if on_step:
                on_step(done, len(tasks))
    return rows


# ── Fase 2 ───────────────────────────────────────────────────────────────────

#: DESENHO A, adotado em 20/08/2026 depois que o desenho original falhou.
#:
#: O original mandava a lista de caixas já confirmadas em texto e pedia "ache o
#: que NÃO está nesta lista". Medido, com o mesmo modelo e a mesma imagem:
#:
#:   com a lista de confirmados ............ 0 propostas (em todos os frames)
#:   sem a lista ........................... inunda: 144 itens, JSON truncado
#:   sem a regra "não chute" ............... 0 propostas
#:   com maxItems=10 ....................... 0 propostas
#:   pedido cru ("liste o que você vê") .... 3 propostas coerentes
#:
#: A variável decisiva é a lista. O modelo ENUMERA bem e SUBTRAI mal: pedir uma
#: operação de conjunto contra coordenadas em texto faz ele responder "nada".
#: Então a subtração sai do prompt e vai para o código, que é onde ela já era
#: feita por IoU de qualquer forma.
#:
#: Consequência honesta, que vai no método: sem a lista crescendo a cada volta,
#: iterar com `temperature=0` e entrada idêntica devolve a mesma resposta. A
#: Fase 2 é um PASSE ÚNICO, e o que ela mede é recall de um passe — não recall
#: iterativo. `VLM_RECALL_MAX_ITERS` fica no config como registro do que foi
#: tentado, mas o padrão aqui é 1.
_RECALL_TASK = """\
List every object of these classes that is clearly visible in this driving \
scene image:
{classes}

Rules:
- Boxes are normalized [x1, y1, x2, y2] with 0,0 at the top-left corner and \
1,1 at the bottom-right.
- Include every instance you can see, including small and distant ones.
- Do not report an object you are guessing at. If you cannot actually see it, \
leave it out.
- Give a confidence of high, medium or low for each."""

#: Classes que a Fase 2 pode propor. É o escopo do gabarito — pedir as 80 do
#: COCO faria o modelo reportar mochila e semáforo de pedestre, que ninguém
#: anotou, e inflaria o falso positivo por lacuna do gabarito e não por erro.
RECALL_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")


def phase2_prompt() -> str:
    """Enumeração pura. Não recebe `confirmed` de propósito — ver _RECALL_TASK."""
    return _RECALL_TASK.format(classes=", ".join(RECALL_CLASSES))


@dataclass(slots=True)
class Proposal:
    cls_name: str
    box: BoxN
    confidence: str
    iteration: int


@dataclass(slots=True)
class Phase2Row:
    acq_id: str
    sec: int
    iterations: int
    proposals: list[Proposal] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    #: Instrumentação da subtração. Sem isto, "0 propostas" é ambíguo: não dá
    #: para distinguir "o modelo não viu nada" de "o modelo só viu o que a
    #: pré-anotação já cobria". São diagnósticos opostos.
    raw_listed: int = 0
    dropped_dedup: int = 0
    dropped_invalid: int = 0

    @property
    def key(self) -> str:
        return f"{self.acq_id}:{self.sec}"


#: Denominador da grade de coordenadas do Qwen3-VL. O modelo ignora o pedido de
#: coordenada normalizada em 0..1 e responde em inteiros 0..1000 — convenção da
#: família Qwen, não defeito do prompt (testei pedir normalizado de três formas).
#:
#: O denominador foi determinado por medição, não por suposição. Sobre 276
#: coordenadas em 6 frames: máximo global 1000, nada acima de 1080, e o teste
#: decisivo é o alinhamento contra as caixas do YOLO — que são verdade
#: POSICIONAL conhecida, independente do gabarito humano:
#:
#:   dividindo por 1000 ............ soma dos melhores IoU = 0,74  ✔
#:   dividindo por (1920, 1080) .... soma dos melhores IoU = 0,00  �’
#:
#: Inspeção caixa a caixa confirma: VLM [0.573 0.488 0.783 0.692] contra YOLO
#: [0.591 0.518 0.813 0.710] é IoU 0,64 — mesmo carro, leve deslocamento.
QWEN_COORD_GRID = 1000.0


def _valid_box(raw: Any) -> BoxN | None:
    """Caixa do modelo → BoxN normalizada, ou None se não der para confiar.

    A conversão é por item e condicional: só divide o que está claramente fora
    de 0..1. Assim, se uma versão futura do modelo passar a responder
    normalizado de verdade, este código continua correto sem mudança.
    """
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        vals = [float(v) for v in raw]
    except (TypeError, ValueError):
        return None
    if any(v > 1.0 for v in vals):
        if any(v > QWEN_COORD_GRID for v in vals):
            return None  # nem 0..1 nem 0..1000: não adivinha, descarta
        vals = [v / QWEN_COORD_GRID for v in vals]
    x1, y1, x2, y2 = vals
    if not all(0.0 <= v <= 1.0 for v in vals):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return BoxN(x1, y1, x2, y2)


def run_phase2(
    client: VlmClient,
    frames: list[Frame],
    image_of: Callable[[str, int], Path],
    *,
    max_iters: int = 1,
    on_step: Callable[[int, int], None] | None = None,
) -> list[Phase2Row]:
    """Passe único de enumeração; a subtração é feita aqui, no código.

    `max_iters > 1` é aceito mas não faz sentido neste desenho: com
    `temperature=0` e entrada idêntica, a volta 2 devolve exatamente a volta 1.
    O parâmetro sobrevive para reproduzir a rodada antiga, não para uso normal.
    """
    rows: list[Phase2Row] = []
    for n, f in enumerate(frames, start=1):
        if not f.reviewed:
            continue
        img = load_frame_image(image_of(f.acq_id, f.sec))

        # O que a pré-anotação propôs e o humano não rejeitou. NÃO vai no prompt
        # — serve só de referência para a subtração por IoU depois da resposta.
        confirmed: list[tuple[str, BoxN]] = [
            (normalize_class(d.cls_name), d.box_n.to_box())
            for d in f.stored_detections
            if d.verdict != "rejected"
            and d.box_n is not None
            and isinstance(d.cls_name, str)
        ]
        row = Phase2Row(acq_id=f.acq_id, sec=f.sec, iterations=0)
        seen: list[BoxN] = [b for _, b in confirmed]

        for it in range(1, max_iters + 1):
            try:
                ans = client.ask(phase2_prompt(), [img], PHASE2_SCHEMA)
            except VlmError as exc:
                row.errors.append(f"iter{it}: {exc}"[:200])
                break
            row.iterations = it

            for item in ans.get("missed") or []:
                if not isinstance(item, dict):
                    continue
                row.raw_listed += 1
                box = _valid_box(item.get("box"))
                cls = item.get("class")
                if box is None or not isinstance(cls, str):
                    row.dropped_invalid += 1
                    continue
                cls_n = normalize_class(cls)
                if cls_n not in COCO_CLASSES:
                    row.dropped_invalid += 1
                    continue
                # Subtração: contra o que a pré-anotação já cobria E contra o que
                # o próprio modelo já listou nesta resposta (ele repete objeto).
                if any(iou(box, b) >= VLM_RECALL_DEDUP_IOU for b in seen):
                    row.dropped_dedup += 1
                    continue
                row.proposals.append(
                    Proposal(cls_n, box, str(item.get("confidence", "medium")), it)
                )
                seen.append(box)

        rows.append(row)
        if on_step:
            on_step(n, len(frames))
    return rows


def dump_rows(path: Path, rows: list[Any]) -> None:
    """Grava as linhas cruas. `asdict` já recursiona nos dataclasses aninhados,
    então `BoxN` sai como {"x1":..,"y1":..} — legível e sem ordem implícita."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in rows]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
