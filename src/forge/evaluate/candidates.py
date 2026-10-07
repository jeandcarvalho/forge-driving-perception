"""Candidatos para a segunda passada de curadoria.

O gabarito nasceu **editando** a saída de um detector: o curador viu caixas
pré-desenhadas e as manteve, corrigiu ou rejeitou. Isso torna a anotação viável e
cobra um preço — o gabarito pode ter herdado o ponto cego do pré-anotador. Objeto
que nenhuma caixa propôs é objeto que o curador provavelmente não procurou.

Este módulo produz a lista do que olhar de novo: detecções de uma inferência
local, com confiança alta, que **não casam com nada** no gabarito. A inferência
local é um observador diferente do pré-anotador (por imagem em vez de tracking
sobre vídeo, outro corte, outro imgsz), então o que ela vê e o gabarito não tem é
exatamente o candidato a ponto cego.

Não é medida de nada. É uma fila de trabalho — a curadoria em si acontece na web,
e só ela decide se o candidato é objeto ou alucinação do detector.

Escopo de classe (o default importa)
────────────────────────────────────
`gt_present` (padrão): só classes que o gabarito já contém. É a fila útil — o
`car` que ninguém anotou é ponto cego de verdade.

`all_coco`: inclui classes que o gabarito nunca teve (`traffic light`, `bench`…).
Isso não é ponto cego, é **escopo**: aquelas classes nunca estiveram no contrato
da anotação. Sai separado, contado, e não misturado com o resto — decidir ampliar
o escopo é decisão de projeto, não de curadoria frame a frame.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from forge.config import EvalClassScope
from forge.core.classes import normalize_class
from forge.evaluate.matcher import Label, Prediction, gt_classes_present, match


@dataclass(slots=True)
class Candidate:
    frame_key: str
    acq_id: str
    sec: int
    cls_name: str
    conf: float
    #: Normalizada, que é a coordenada canônica do banco — a curadoria na web
    #: consome exatamente esta forma.
    box_n: dict[str, float]
    #: Melhor sobreposição com QUALQUER instância do gabarito. Perto de zero é
    #: objeto sem nada em volta; subindo, é caixa duplicada de algo já anotado.
    best_iou: float
    in_scope: bool


@dataclass(slots=True)
class CandidateReport:
    iou_threshold: float
    min_conf: float
    class_scope: str
    predictions_considered: int
    ground_truth: int
    candidates: list[Candidate] = field(default_factory=list)
    #: Candidatos de classe fora do gabarito, contados mas não listados junto.
    out_of_scope_by_class: dict[str, int] = field(default_factory=dict)

    @property
    def frames_touched(self) -> int:
        return len({c.frame_key for c in self.candidates})

    @property
    def by_class(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.candidates:
            out[c.cls_name] = out.get(c.cls_name, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def to_json(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "iou_threshold": self.iou_threshold,
            "min_conf": self.min_conf,
            "class_scope": self.class_scope,
            "predictions_considered": self.predictions_considered,
            "ground_truth": self.ground_truth,
            "frames": self.frames_touched,
            "by_class": self.by_class,
            "out_of_scope_by_class": self.out_of_scope_by_class,
            "note": (
                "Fila de trabalho da segunda passada de curadoria, não medida. "
                "Detecção com confiança alta sem nada correspondente no gabarito: "
                "candidata a objeto que o pré-anotador não propôs e o curador não "
                "procurou. Quem decide se é objeto é a curadoria na web."
            ),
            "candidates": [asdict(c) for c in self.candidates],
        }
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def find_candidates(
    predictions: list[Prediction],
    ground_truth: list,
    *,
    min_conf: float = 0.5,
    iou_threshold: float = 0.5,
    class_scope: EvalClassScope = EvalClassScope.GT_PRESENT,
) -> CandidateReport:
    """Detecções de confiança ≥ `min_conf` que o matcher classificou como fundo.

    `fp_background` é a categoria certa e as outras não servem: uma predição
    `fp_localization` cobre uma instância que JÁ existe no gabarito (o curador
    viu o objeto, a caixa é que ficou torta) e `fp_classification` idem. Só o
    "fundo" significa *não há nada anotado aqui* — e é isso que a segunda passada
    vai conferir.
    """
    scope = gt_classes_present(ground_truth)
    strong = [p for p in predictions if p.conf >= min_conf]

    out_of_scope: dict[str, int] = {}
    if class_scope is EvalClassScope.GT_PRESENT:
        considered = []
        for p in strong:
            c = normalize_class(p.cls_name)
            if c in scope:
                considered.append(p)
            else:
                out_of_scope[c] = out_of_scope.get(c, 0) + 1
        strong = considered

    res = match(strong, ground_truth, iou_threshold=iou_threshold, class_scope=class_scope)

    report = CandidateReport(
        iou_threshold=iou_threshold,
        min_conf=min_conf,
        class_scope=class_scope.value,
        predictions_considered=len(strong),
        ground_truth=len(ground_truth),
        out_of_scope_by_class=dict(sorted(out_of_scope.items(), key=lambda kv: -kv[1])),
    )

    for m in res.predictions:
        if m.label is not Label.FP_BACKGROUND:
            continue
        acq, _, sec = m.pred.frame_key.partition(":")
        b = m.pred.box
        report.candidates.append(
            Candidate(
                frame_key=m.pred.frame_key,
                acq_id=acq,
                sec=int(sec),
                cls_name=normalize_class(m.pred.cls_name),
                conf=round(m.pred.conf, 4),
                box_n={
                    "x1": round(b.x1, 6), "y1": round(b.y1, 6),
                    "x2": round(b.x2, 6), "y2": round(b.y2, 6),
                },
                best_iou=round(m.best_iou, 4),
                in_scope=normalize_class(m.pred.cls_name) in scope,
            )
        )

    # Confiança decrescente: a fila é para gente olhar, e o mais provável primeiro.
    report.candidates.sort(key=lambda c: -c.conf)
    return report
