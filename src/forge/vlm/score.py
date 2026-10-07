"""Placar do conjunto YOLO+VLM contra o gabarito, na aritmética da Table 6.

A pergunta: se a curadoria do VLM fosse aplicada de verdade ao que o YOLO
propôs, o conjunto resultante seria melhor ou pior que o YOLO cru?

**O gabarito decide, não o VLM.** A ação do VLM só determina se a caixa
SOBREVIVE e com que rótulo; se a caixa sobrevivente está certa ou errada é o
veredito humano que diz. Essa separação é o coração do módulo — misturar as duas
coisas faria o VLM se avaliar.

Aritmética, idêntica à do `parity.py` mas escrita na forma mínima:

    GT   = 731            (o gabarito não muda com a curadoria)
    TP   = caixas sobreviventes que estão corretas
    FP   = sobreviventes − TP
    FN   = GT − TP
    P    = TP / sobreviventes        R = TP / GT

Confere no braço base: 501 caixas, TP 329 → FP 172, FN 402, P 65,7%, R 45,0%.
São os números publicados.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from forge.api.models import Frame
from forge.core.classes import normalize_class
from forge.vlm.phases import Phase1Row, Phase2Row
from forge.vlm.prepare import human_added_boxes
from forge.vlm.report import presence_hit

#: Vereditos do VLM que preservam a caixa e o rótulo original.
_KEEPS_BOX = frozenset({"kept", "corrected-minor", "corrected-redrawn"})

#: Vereditos humanos em que a caixa proposta já está correta como está — é a
#: definição de `tp` da Table 6: `kept + corrected(mesma classe, IoU ≥ 0.7)`.
_TRUTH_OK = frozenset({"kept", "corrected-minor"})


@dataclass(slots=True)
class Score:
    label: str
    gt: int
    tp: int = 0
    boxes: int = 0
    #: De onde veio cada acerto e cada erro, para a leitura não virar caixa-preta.
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
            "label": self.label,
            "boxes": self.boxes,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "ground_truth": self.gt,
            "precision": self.precision,
            "recall": self.recall,
            "detail": dict(sorted(self.detail.items())),
        }


def human_corrected_class(frames: list[Frame]) -> dict[str, str]:
    """`acq:sec:index` → classe que o HUMANO pôs no lugar, em `class-changed`.

    Sem isto não dá para saber se o VLM acertou a correção de rótulo: ele
    trocaria `car` por `bus` e nós contaríamos como acerto só porque trocou.
    """
    out: dict[str, str] = {}
    for f in frames:
        for i, det in enumerate(f.stored_detections):
            corr = det.correction
            if corr is None or not isinstance(corr.cls_name, str):
                continue
            out[f"{f.acq_id}:{f.sec}:{i}"] = normalize_class(corr.cls_name)
    return out


def score_phase1(
    rows: list[Phase1Row],
    corrected: dict[str, str],
    score: Score,
    *,
    drop_corrections: bool,
) -> None:
    """Aplica a curadoria do VLM às caixas propostas e pontua o que sobra.

    `drop_corrections` é a variante de sensibilidade: trata os vereditos
    `corrected-minor`/`corrected-redrawn` do VLM como remoção, já que ele não
    devolve coordenada nova e portanto não "corrige" nada de fato.
    """
    for r in rows:
        v = r.vlm
        # Chamada que falhou não pode virar decisão. A caixa segue como o YOLO
        # a entregou — é o comportamento conservador e é o que um pipeline real
        # faria diante de um curador que não respondeu.
        if v is None:
            v = "kept"
            score.bump("falha_chamada_tratada_como_kept")

        removed = v == "rejected" or (drop_corrections and v in {"corrected-minor", "corrected-redrawn"})

        if removed:
            # Caixa sai. O que isso vale depende do que ela era no gabarito.
            if r.truth == "rejected":
                score.bump("removeu_fp_de_fundo")  # acerto: era lixo mesmo
            elif r.truth in _TRUTH_OK:
                score.bump("removeu_caixa_boa")  # dano: perdeu um TP
            else:
                score.bump("removeu_fp_de_erro")  # acerto parcial: FP some, FN fica
            continue

        # Caixa fica. Qual rótulo ela carrega agora?
        label = r.vlm_class if v == "class-changed" and r.vlm_class else r.proposed_class
        score.boxes += 1

        if r.truth == "rejected":
            score.bump("manteve_fp_de_fundo")
            continue
        if r.truth == "corrected-redrawn":
            # Geometria ruim e o VLM não tem como consertar: continua FP.
            score.bump("manteve_caixa_mal_posicionada")
            continue
        if r.truth == "class-changed":
            want = corrected.get(r.key)
            if want is not None and label == want:
                score.tp += 1
                score.bump("corrigiu_rotulo_certo")
            else:
                score.bump("rotulo_continua_errado")
            continue
        # truth ∈ {kept, corrected-minor}: a caixa estava certa.
        if label == r.proposed_class:
            score.tp += 1
            score.bump("manteve_caixa_boa")
        else:
            score.bump("estragou_caixa_boa")  # trocou o rótulo de uma caixa certa


def score_phase2(
    rows: list[Phase2Row],
    frames: list[Frame],
    score: Score,
    *,
    require_class: bool = True,
) -> None:
    """Soma as adições da Fase 2, casadas contra as caixas desenhadas na mão.

    ⚠️ O casamento aqui é POR PRESENÇA (IoU frouxo ou centro dentro), régua mais
    permissiva que a das caixas da Fase 1 — que foram julgadas pelo humano com
    IoU 0.7 entre original e correção. Isso favorece o braço VLM e precisa estar
    escrito no método. `require_class=False` afrouxa mais ainda, ignorando o
    rótulo; serve de cota superior, não de resultado.
    """
    by_key = {f"{f.acq_id}:{f.sec}": f for f in frames}
    for row in rows:
        frame = by_key.get(row.key)
        if frame is None:
            continue
        truth = human_added_boxes(frame)
        unused = list(range(len(truth)))
        for prop in row.proposals:
            score.boxes += 1
            hit = next((i for i in unused if presence_hit(prop.box, truth[i][1])), None)
            if hit is None:
                score.bump("adicao_sem_alvo")
                continue
            unused.remove(hit)
            if not require_class or prop.cls_name == truth[hit][0]:
                score.tp += 1
                score.bump("recuperou_objeto_perdido")
            else:
                score.bump("achou_objeto_rotulo_errado")


def baseline_score(rows: list[Phase1Row], gt: int) -> Score:
    """YOLO cru: todas as caixas ficam, nada muda. Reproduz a Table 6."""
    s = Score(label="YOLO base", gt=gt)
    for r in rows:
        s.boxes += 1
        if r.truth in _TRUTH_OK:
            s.tp += 1
    return s


def format_table(scores: list[Score]) -> str:
    head = (
        f"{'conjunto':<34}{'caixas':>8}{'TP':>6}{'FP':>6}{'FN':>6}"
        f"{'precisão':>11}{'recall':>9}"
    )
    lines = [head, "─" * len(head)]
    for s in scores:
        lines.append(
            f"{s.label:<34}{s.boxes:>8}{s.tp:>6}{s.fp:>6}{s.fn:>6}"
            f"{str(s.precision) + '%':>11}{str(s.recall) + '%':>9}"
        )
    return "\n".join(lines)
