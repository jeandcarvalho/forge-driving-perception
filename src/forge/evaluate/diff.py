"""Relatório de divergência entre `audit` e `eval`.

Isto é ENTREGÁVEL, não log. A lista de casos é o material do parágrafo de método
do paper; a contagem sozinha não diz nada a um revisor.

O que está sendo comparado
──────────────────────────
As mesmas detecções armazenadas, julgadas por dois algoritmos:

  audit (parity.py)   segue o vínculo `ref_det_id`: esta detecção recebeu qual
                      ação humana? A identidade decide.
  eval  (matcher.py)  ignora o vínculo: esta caixa sobrepõe alguma instância do
                      gabarito? A geometria decide.

Onde eles discordam há sempre uma história concreta. As três típicas:

  rejected → tp        o curador rejeitou a caixa como falso positivo, mas ela
                       cai em cima de uma instância `added` que ele desenhou
                       depois. Para a auditoria é FP de fundo; para a geometria
                       é acerto. Costuma ser reanotação: a mesma coisa, remarcada
                       do zero em vez de corrigida.
  corrected → tp       a caixa foi refeita (IoU < 0.7 com a correção dela), mas
                       ainda encosta o suficiente em ALGUMA instância do gabarito
                       — às vezes na de outro objeto vizinho.
  kept → fp_duplicate  duas pré-anotações confirmadas sobre a mesma instância; a
                       geometria só deixa uma acertar.

Para ser comparação justa o matcher roda com `EvalClassScope.ALL_COCO`: a
contagem por veredito não mascara classe nenhuma, então mascarar de um lado só
inventaria divergência.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forge.api.models import Frame
from forge.config import PARITY_IOU, EvalClassScope
from forge.evaluate.matcher import (
    GtLabel,
    gt_from_frames,
    match,
    stored_detections_as_predictions,
)
from forge.evaluate.parity import curation_parity, parity_detection_labels


@dataclass(slots=True)
class Divergence:
    frame_key: str
    det_id: str
    cls_name: str
    conf: float | None
    audit_label: str
    eval_label: str
    best_iou: float
    matched_gt_class: str | None
    matched_gt_origin: str | None

    @property
    def transition(self) -> str:
        return f"{self.audit_label} → {self.eval_label}"


@dataclass(slots=True)
class DiffReport:
    iou_threshold: float
    compared_detections: int
    audit_counters: dict[str, int]
    eval_counters: dict[str, int]
    divergences: list[Divergence] = field(default_factory=list)
    #: Instâncias `added` que a geometria casou com alguma pré-anotação.
    added_gt_matched_geometrically: list[str] = field(default_factory=list)

    @property
    def agree(self) -> bool:
        return not self.divergences and not self.added_gt_matched_geometrically

    @property
    def by_transition(self) -> dict[str, int]:
        return dict(Counter(d.transition for d in self.divergences).most_common())

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "iou_threshold": self.iou_threshold,
                    "compared_detections": self.compared_detections,
                    "audit_counters": self.audit_counters,
                    "eval_counters": self.eval_counters,
                    "agree": self.agree,
                    "by_transition": self.by_transition,
                    "divergences": [asdict(d) for d in self.divergences],
                    "added_gt_matched_geometrically": self.added_gt_matched_geometrically,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return path


def audit_vs_eval(frames: list[Frame], *, iou_threshold: float = PARITY_IOU) -> DiffReport:
    """Julga as mesmas detecções pelos dois algoritmos e lista onde discordam."""
    audit_labels = parity_detection_labels(frames)
    parity = curation_parity(frames)

    preds = stored_detections_as_predictions(frames)
    gts = gt_from_frames(frames)
    # ALL_COCO de propósito: comparação justa com uma contagem que não mascara.
    result = match(
        preds, gts, iou_threshold=iou_threshold, class_scope=EvalClassScope.ALL_COCO
    )

    divergences: list[Divergence] = []
    compared = 0

    for m in result.predictions:
        det_id = m.pred.det_id
        if det_id is None:
            continue
        key = f"{m.pred.frame_key}|{det_id}"
        audit_label = audit_labels.get(key)
        if audit_label is None:
            continue  # detecção sem veredito: fora da auditoria
        compared += 1

        eval_label = m.label.value
        if eval_label == audit_label:
            continue
        divergences.append(
            Divergence(
                frame_key=m.pred.frame_key,
                det_id=det_id,
                cls_name=m.pred.cls_name,
                conf=m.pred.conf,
                audit_label=audit_label,
                eval_label=eval_label,
                best_iou=round(m.best_iou, 4),
                matched_gt_class=m.matched_gt.cls_name if m.matched_gt else None,
                matched_gt_origin=m.matched_gt.origin if m.matched_gt else None,
            )
        )

    # Lado do gabarito: a auditoria conta TODA instância `added` como FN não
    # detectada, por construção. Se a geometria casou alguma, é o mesmo fenômeno
    # visto do outro lado.
    added_matched = [
        f"{g.gt.frame_key}|{g.gt.det_id} ({g.gt.cls_name})"
        for g in result.ground_truth
        if g.gt.origin == "added" and g.label is GtLabel.MATCHED
    ]

    return DiffReport(
        iou_threshold=iou_threshold,
        compared_detections=compared,
        audit_counters={
            "detections": parity.detections,
            "ground_truth": parity.ground_truth,
            "tp": parity.tp,
            "fp": parity.fp,
            "fn": parity.fn,
            "fp_localization": parity.fp_localization,
            "fp_classification": parity.fp_classification,
            "fp_background": parity.fp_background,
            "fn_from_errors": parity.fn_from_errors,
            "fn_undetected": parity.fn_undetected,
        },
        eval_counters=result.counters,
        divergences=sorted(divergences, key=lambda d: (d.transition, d.frame_key)),
        added_gt_matched_geometrically=sorted(added_matched),
    )
