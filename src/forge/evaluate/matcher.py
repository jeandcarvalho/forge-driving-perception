"""Matching geométrico predição × gabarito, com decomposição de erro estilo TIDE.

ISTO É AVALIAÇÃO DE DETECTOR. Não confunda com evaluate/parity.py, que audita a
curadoria contando por veredito e vínculo `ref_det_id`.

Aqui não existe vínculo: uma predição é só uma caixa com classe e confiança, e o
que decide se ela acertou é a sobreposição. É por isso que os dois algoritmos
podem discordar no mesmo input — ver evaluate/diff.py.

Decomposição do falso positivo (TIDE, Bolya et al. 2020). Com `tf` = limiar de
acerto e `tb` = limiar de fundo, e olhando o melhor IoU contra o gabarito da
MESMA classe e contra o de OUTRA classe:

  fp_localization   mesma classe, tb ≤ IoU < tf — achou, errou onde
  fp_classification mesma classe < tb, outra classe ≥ tf — errou o quê
  fp_both           mesma classe < tb, outra classe entre tb e tf — errou o quê
                    E errou onde, ao mesmo tempo
  fp_duplicate      mesma classe ≥ tf, mas o objeto já foi levado por uma
                    predição de confiança maior
  fp_background     nada em lugar nenhum passa de tb

As cinco são mutuamente exclusivas por construção. `fp_both` não é preciosismo
de taxonomia: sem ela, uma caixa que errou classe e posição junto seria contada
como "fundo", e a leitura viraria "o modelo alucina objeto do nada" quando o que
houve foi um objeto real, mal classificado e mal localizado.

E do falso negativo:
  fn_from_errors    instância coberta por predição que virou FP de loc/cls
  fn_undetected     instância que nenhuma predição chegou perto
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from forge.api.models import Frame
from forge.config import EvalClassScope
from forge.core.classes import normalize_class
from forge.core.geometry import BoxN, iou

#: Abaixo disto a predição não está "errando a caixa", está vendo fundo.
BACKGROUND_IOU = 0.1


class Label(str, Enum):
    TP = "tp"
    FP_LOCALIZATION = "fp_localization"
    FP_CLASSIFICATION = "fp_classification"
    FP_BOTH = "fp_both"
    FP_DUPLICATE = "fp_duplicate"
    FP_BACKGROUND = "fp_background"


class GtLabel(str, Enum):
    MATCHED = "matched"
    FN_FROM_ERRORS = "fn_from_errors"
    FN_UNDETECTED = "fn_undetected"


@dataclass(frozen=True, slots=True)
class Prediction:
    frame_key: str
    cls_name: str
    conf: float
    box: BoxN
    det_id: str | None = None


@dataclass(frozen=True, slots=True)
class GtInstance:
    frame_key: str
    cls_name: str
    box: BoxN
    origin: str | None = None
    det_id: str | None = None


@dataclass(slots=True)
class PredMatch:
    pred: Prediction
    label: Label
    best_iou: float
    matched_gt: GtInstance | None = None


@dataclass(slots=True)
class GtMatch:
    gt: GtInstance
    label: GtLabel
    matched_by: Prediction | None = None


@dataclass(slots=True)
class MatchResult:
    iou_threshold: float
    class_scope: EvalClassScope
    scope_classes: frozenset[str]
    predictions: list[PredMatch] = field(default_factory=list)
    ground_truth: list[GtMatch] = field(default_factory=list)
    #: Predições descartadas antes do matching por classe fora do escopo.
    masked_predictions: int = 0
    masked_classes: dict[str, int] = field(default_factory=dict)

    def count(self, label: Label) -> int:
        return sum(1 for m in self.predictions if m.label is label)

    @property
    def tp(self) -> int:
        return self.count(Label.TP)

    @property
    def fp(self) -> int:
        return len(self.predictions) - self.tp

    @property
    def fn(self) -> int:
        return sum(1 for g in self.ground_truth if g.label is not GtLabel.MATCHED)

    @property
    def counters(self) -> dict[str, int]:
        return {
            "detections": len(self.predictions),
            "ground_truth": len(self.ground_truth),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "fp_localization": self.count(Label.FP_LOCALIZATION),
            "fp_classification": self.count(Label.FP_CLASSIFICATION),
            "fp_both": self.count(Label.FP_BOTH),
            "fp_duplicate": self.count(Label.FP_DUPLICATE),
            "fp_background": self.count(Label.FP_BACKGROUND),
            "fn_from_errors": sum(
                1 for g in self.ground_truth if g.label is GtLabel.FN_FROM_ERRORS
            ),
            "fn_undetected": sum(
                1 for g in self.ground_truth if g.label is GtLabel.FN_UNDETECTED
            ),
            "masked_predictions": self.masked_predictions,
        }


def gt_classes_present(gts: list[GtInstance]) -> frozenset[str]:
    """Escopo de avaliação derivado do DADO, não de constante."""
    return frozenset(normalize_class(g.cls_name) for g in gts)


def match(
    predictions: list[Prediction],
    ground_truth: list[GtInstance],
    *,
    iou_threshold: float = 0.5,
    class_scope: EvalClassScope = EvalClassScope.GT_PRESENT,
) -> MatchResult:
    """Casa predições com gabarito, frame a frame, por confiança decrescente.

    `class_scope` NÃO tem default implícito escondido aqui — quem chama decide, e
    o padrão do projeto está em config.EVAL_CLASS_SCOPE_DEFAULT. Ver a docstring
    de EvalClassScope para por que mascarar é o honesto.
    """
    scope = gt_classes_present(ground_truth)
    result = MatchResult(
        iou_threshold=iou_threshold, class_scope=class_scope, scope_classes=scope
    )

    # ── mascaramento ─────────────────────────────────────────────────────────
    if class_scope is EvalClassScope.GT_PRESENT:
        kept: list[Prediction] = []
        for p in predictions:
            if normalize_class(p.cls_name) in scope:
                kept.append(p)
            else:
                result.masked_predictions += 1
                c = normalize_class(p.cls_name)
                result.masked_classes[c] = result.masked_classes.get(c, 0) + 1
        predictions = kept

    gt_by_frame: dict[str, list[GtInstance]] = {}
    for g in ground_truth:
        gt_by_frame.setdefault(g.frame_key, []).append(g)

    pred_by_frame: dict[str, list[Prediction]] = {}
    for p in predictions:
        pred_by_frame.setdefault(p.frame_key, []).append(p)

    gt_state: dict[int, GtMatch] = {
        id(g): GtMatch(gt=g, label=GtLabel.FN_UNDETECTED) for g in ground_truth
    }
    # Instâncias que uma predição chegou a cobrir, mas com erro (loc ou cls).
    covered_with_error: set[int] = set()

    for frame_key in set(pred_by_frame) | set(gt_by_frame):
        frame_gts = gt_by_frame.get(frame_key, [])
        frame_preds = sorted(
            pred_by_frame.get(frame_key, []), key=lambda p: p.conf, reverse=True
        )
        taken: set[int] = set()

        for p in frame_preds:
            p_cls = normalize_class(p.cls_name)

            best_same: tuple[float, GtInstance | None] = (0.0, None)
            best_same_free: tuple[float, GtInstance | None] = (0.0, None)
            best_diff: tuple[float, GtInstance | None] = (0.0, None)

            for g in frame_gts:
                ov = iou(p.box, g.box)
                if normalize_class(g.cls_name) == p_cls:
                    if ov > best_same[0]:
                        best_same = (ov, g)
                    if id(g) not in taken and ov > best_same_free[0]:
                        best_same_free = (ov, g)
                elif ov > best_diff[0]:
                    best_diff = (ov, g)

            # 1) acerto: mesma classe, IoU suficiente, instância ainda livre
            if best_same_free[0] >= iou_threshold and best_same_free[1] is not None:
                g = best_same_free[1]
                taken.add(id(g))
                gt_state[id(g)] = GtMatch(gt=g, label=GtLabel.MATCHED, matched_by=p)
                result.predictions.append(
                    PredMatch(p, Label.TP, best_same_free[0], g)
                )
                continue

            # 2) duplicata: acertaria, mas alguém de confiança maior já levou
            if best_same[0] >= iou_threshold:
                result.predictions.append(
                    PredMatch(p, Label.FP_DUPLICATE, best_same[0], best_same[1])
                )
                continue

            # 3) erro de localização: classe CERTA, caixa no meio do caminho.
            #    Testado antes dos erros de classe porque as categorias do TIDE
            #    são definidas a partir do IoU com a própria classe: só quando
            #    ele fica abaixo de tb é que olhamos as outras classes.
            if best_same[0] >= BACKGROUND_IOU and best_same[1] is not None:
                covered_with_error.add(id(best_same[1]))
                result.predictions.append(
                    PredMatch(p, Label.FP_LOCALIZATION, best_same[0], best_same[1])
                )
                continue

            # 4) erro de classe: nada da própria classe por perto, mas há um
            #    objeto de OUTRA classe bem debaixo da caixa
            if best_diff[0] >= iou_threshold and best_diff[1] is not None:
                covered_with_error.add(id(best_diff[1]))
                result.predictions.append(
                    PredMatch(p, Label.FP_CLASSIFICATION, best_diff[0], best_diff[1])
                )
                continue

            # 5) os dois de uma vez: classe errada E caixa no meio do caminho
            if best_diff[0] >= BACKGROUND_IOU and best_diff[1] is not None:
                covered_with_error.add(id(best_diff[1]))
                result.predictions.append(
                    PredMatch(p, Label.FP_BOTH, best_diff[0], best_diff[1])
                )
                continue

            # 6) fundo: nada em lugar nenhum passa de tb
            best_overall = max(best_same[0], best_diff[0])
            result.predictions.append(
                PredMatch(p, Label.FP_BACKGROUND, best_overall, None)
            )

    for gid, gm in gt_state.items():
        if gm.label is GtLabel.FN_UNDETECTED and gid in covered_with_error:
            gm.label = GtLabel.FN_FROM_ERRORS
    result.ground_truth = list(gt_state.values())

    return result


# ── extração a partir do manifest ────────────────────────────────────────────


def gt_from_frames(frames: list[Frame]) -> list[GtInstance]:
    """Só frames com gabarito. Não revisado não vira "zero instâncias"."""
    out: list[GtInstance] = []
    for f in frames:
        if not f.is_labeled:
            continue
        for g in f.ground_truth or []:
            if g.cls_name is None or g.box_n is None:
                continue
            out.append(
                GtInstance(f.key, g.cls_name, g.box_n.to_box(), g.origin, g.det_id)
            )
    return out


def stored_detections_as_predictions(frames: list[Frame]) -> list[Prediction]:
    """As pré-anotações do manifest, tratadas como se fossem predições.

    Serve ao relatório de diff (evaluate/diff.py) e a nada mais: elas vieram de
    tracking sobre vídeo com corte de confiança em ~0.2, então não têm cauda de
    baixa confiança e não produzem curva precision-recall — logo, não dá mAP.
    """
    out: list[Prediction] = []
    for f in frames:
        if not f.is_labeled:
            continue
        for d in f.stored_detections:
            if d.cls_name is None or d.box_n is None:
                continue
            out.append(
                Prediction(f.key, d.cls_name, d.conf if d.conf is not None else 0.0,
                           d.box_n.to_box(), d.det_id)
            )
    return out
