"""O que sai do A1: matriz de concordância, precisão/recall, e o custo.

Três entregas, nesta ordem de importância:

1. **Matriz VLM × humano** nas 5 categorias — onde bateram e onde discordaram.
2. **Precisão/recall do VLM contra o gabarito final**, no mesmo formato da
   tabela que já existe para o YOLO base, para dar comparação lado a lado.
3. **Custo** — aqui é tempo de máquina e nº de chamadas, não dólar.

O gabarito humano é referência absoluta: quando o VLM discorda, quem errou é o
VLM. Esta é a regra do projeto e não se reabre por conveniência de resultado.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge.api.models import Frame
from forge.config import VLM_RECALL_PRESENCE_IOU, VLM_VERDICTS
from forge.core.geometry import BoxN, iou
from forge.vlm.phases import Phase1Row, Phase2Row
from forge.vlm.prepare import human_added_boxes

#: Colapso das 5 categorias em 3, para o número que não depende de geometria.
#: A distinção minor × redrawn É geométrica (IoU 0.7 entre original e correção),
#: e cobrar isso de um VLM mede a fraqueza dele, não a utilidade dele. O número
#: honesto de manchete é o de 3 vias; o de 5 vias é o detalhe.
COARSE = {
    "kept": "accept",
    "corrected-minor": "fix-box",
    "corrected-redrawn": "fix-box",
    "class-changed": "fix-class",
    "rejected": "reject",
}
COARSE_LABELS = ("accept", "fix-box", "fix-class", "reject")


@dataclass(slots=True)
class Agreement:
    labels: tuple[str, ...]
    matrix: dict[str, dict[str, int]]
    total: int
    agreed: int
    errors: int

    @property
    def accuracy(self) -> float | None:
        return round(100 * self.agreed / self.total, 1) if self.total else None

    def per_truth(self) -> dict[str, dict[str, Any]]:
        """Recall por categoria humana: das N que o humano chamou de X, quantas
        o VLM também chamou de X."""
        out: dict[str, dict[str, Any]] = {}
        for t in self.labels:
            row = self.matrix[t]
            n = sum(row.values())
            out[t] = {
                "n": n,
                "hit": row[t],
                "pct": round(100 * row[t] / n, 1) if n else None,
            }
        return out


def agreement(rows: list[Phase1Row], *, coarse: bool = False) -> Agreement:
    labels = COARSE_LABELS if coarse else VLM_VERDICTS
    matrix = {t: {v: 0 for v in labels} for t in labels}
    total = agreed = errors = 0
    for r in rows:
        if r.vlm is None:
            errors += 1
            continue
        t = COARSE[r.truth] if coarse else r.truth
        v = COARSE[r.vlm] if coarse else r.vlm
        matrix[t][v] += 1
        total += 1
        agreed += t == v
    return Agreement(tuple(labels), matrix, total, agreed, errors)


# ── Pilar 2 na régua do VLM ──────────────────────────────────────────────────


@dataclass(slots=True)
class CurationMetrics:
    """Precisão/recall implícitos na curadoria do VLM.

    Mesma aritmética de `evaluate/parity.py`, aplicada aos vereditos do VLM em
    vez dos humanos. É isso que permite pôr a linha do VLM embaixo da linha do
    YOLO base na mesma tabela: `tp = kept + corrected-minor`, e por aí.
    """

    counts: dict[str, int]
    added: int
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def finish(self) -> CurationMetrics:
        c = self.counts
        self.tp = c["kept"] + c["corrected-minor"]
        self.fp = c["corrected-redrawn"] + c["class-changed"] + c["rejected"]
        self.fn = self.added + c["corrected-redrawn"] + c["class-changed"]
        return self

    @property
    def detections(self) -> int:
        return sum(self.counts.values())

    @property
    def ground_truth(self) -> int:
        return self.detections - self.counts["rejected"] + self.added

    @property
    def precision(self) -> float | None:
        d = self.detections
        return round(100 * self.tp / d, 1) if d else None

    @property
    def recall(self) -> float | None:
        g = self.ground_truth
        return round(100 * self.tp / g, 1) if g else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "detections": self.detections,
            "ground_truth": self.ground_truth,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            **{k.replace("-", "_"): v for k, v in self.counts.items()},
            "added": self.added,
        }


def curation_metrics(
    rows: list[Phase1Row], *, added: int, source: str
) -> CurationMetrics:
    """`source` é "truth" (humano) ou "vlm"."""
    counts = {v: 0 for v in VLM_VERDICTS}
    for r in rows:
        label = r.truth if source == "truth" else r.vlm
        if label in counts:
            counts[label] += 1
    return CurationMetrics(counts=counts, added=added).finish()


# ── Fase 2: recall de PRESENÇA ───────────────────────────────────────────────


def _center_in(box: BoxN, target: BoxN) -> bool:
    cx, cy = (box.x1 + box.x2) / 2, (box.y1 + box.y2) / 2
    return target.x1 <= cx <= target.x2 and target.y1 <= cy <= target.y2


def presence_hit(proposal: BoxN, truth: BoxN) -> bool:
    """Casamento frouxo, deliberadamente.

    IoU >= 0.3 OU centro da proposta dentro da caixa humana. A pergunta é se o
    VLM PERCEBEU que faltava um objeto naquele lugar — não se ele desenhou bem.
    Cobrar geometria fina aqui mediria a fraqueza conhecida do modelo e diria
    pouco sobre a utilidade dele como triador.
    """
    return iou(proposal, truth) >= VLM_RECALL_PRESENCE_IOU or _center_in(proposal, truth)


@dataclass(slots=True)
class RecallReport:
    frames: int
    human_added: int
    found: int
    proposals: int
    matched_proposals: int
    class_correct: int
    iterations_used: dict[int, int] = field(default_factory=dict)
    by_class: dict[str, dict[str, int]] = field(default_factory=dict)

    @property
    def recall(self) -> float | None:
        return round(100 * self.found / self.human_added, 1) if self.human_added else None

    @property
    def precision(self) -> float | None:
        """Proporção das propostas que caíram em cima de caixa humana.

        Cuidado ao ler: uma proposta "errada" pode ser um objeto real que o
        humano não anotou. Isto é precisão CONTRA O GABARITO, e o gabarito é a
        verdade absoluta do projeto — mas a limitação vale nota de rodapé.
        """
        return (
            round(100 * self.matched_proposals / self.proposals, 1)
            if self.proposals
            else None
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "frames": self.frames,
            "human_added": self.human_added,
            "found": self.found,
            "recall": self.recall,
            "proposals": self.proposals,
            "matched_proposals": self.matched_proposals,
            "precision": self.precision,
            "class_correct": self.class_correct,
            "iterations_used": self.iterations_used,
            "by_class": self.by_class,
        }


def recall_report(rows: list[Phase2Row], frames: list[Frame]) -> RecallReport:
    by_key = {f"{f.acq_id}:{f.sec}": f for f in frames}
    rep = RecallReport(
        frames=len(rows), human_added=0, found=0, proposals=0,
        matched_proposals=0, class_correct=0,
    )
    for row in rows:
        frame = by_key.get(row.key)
        if frame is None:
            continue
        truth = human_added_boxes(frame)
        rep.human_added += len(truth)
        rep.proposals += len(row.proposals)
        rep.iterations_used[row.iterations] = rep.iterations_used.get(row.iterations, 0) + 1

        for cls, _ in truth:
            slot = rep.by_class.setdefault(cls, {"human": 0, "found": 0})
            slot["human"] += 1

        # Casamento guloso: cada caixa humana conta no máximo uma vez, e cada
        # proposta casa no máximo uma caixa. Sem isso, três propostas em cima do
        # mesmo pedestre virariam "recall 300%".
        unused = list(range(len(truth)))
        for prop in row.proposals:
            hit = next(
                (i for i in unused if presence_hit(prop.box, truth[i][1])), None
            )
            if hit is None:
                continue
            unused.remove(hit)
            rep.matched_proposals += 1
            rep.found += 1
            cls_t = truth[hit][0]
            rep.by_class[cls_t]["found"] += 1
            if prop.cls_name == cls_t:
                rep.class_correct += 1
    return rep


# ── impressão ────────────────────────────────────────────────────────────────


#: Cabeçalhos curtos — o nome inteiro estoura 80 colunas e a matriz quebra
#: linha, que é justamente o formato em que ninguém consegue lê-la.
_SHORT = {
    "kept": "kept",
    "corrected-minor": "minor",
    "corrected-redrawn": "redrawn",
    "class-changed": "class",
    "rejected": "reject",
    "accept": "accept",
    "fix-box": "fix-box",
    "fix-class": "fix-cls",
    "reject": "reject",
}


def format_matrix(ag: Agreement, title: str) -> str:
    w = max(len(x) for x in ag.labels) + 2
    cell = 9
    head = "humano \\ vlm".ljust(w) + "".join(
        _SHORT.get(x, x)[:cell - 1].rjust(cell) for x in ag.labels
    )
    lines = [title, "=" * (len(head) + 9), head]
    for t in ag.labels:
        row = ag.matrix[t]
        # A diagonal (concordância) vai entre colchetes — é onde o olho procura.
        cells = "".join(
            (f"[{row[v]}]" if v == t else str(row[v])).rjust(cell) for v in ag.labels
        )
        lines.append(t.ljust(w) + cells + f"  n={sum(row.values())}")
    lines.append("")
    lines.append(
        f"concordância: {ag.agreed}/{ag.total} = {ag.accuracy}%"
        + (f"   ·   {ag.errors} falhas de chamada" if ag.errors else "")
    )
    return "\n".join(lines)
