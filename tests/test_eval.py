"""Matcher geométrico, mascaramento de classe e o diff audit × eval."""

from __future__ import annotations

import pytest

from forge.api.models import Frame
from forge.config import EvalClassScope
from forge.core.geometry import BoxN
from forge.evaluate.diff import audit_vs_eval
from forge.evaluate.matcher import (
    GtInstance,
    GtLabel,
    Label,
    Prediction,
    gt_from_frames,
    match,
    stored_detections_as_predictions,
)


def P(cls: str, conf: float, box: BoxN, key: str = "a:1") -> Prediction:
    return Prediction(key, cls, conf, box)


def G(cls: str, box: BoxN, key: str = "a:1", origin: str = "kept") -> GtInstance:
    return GtInstance(key, cls, box, origin)


BOX = BoxN(0.1, 0.1, 0.3, 0.3)
BOX_NEAR = BoxN(0.12, 0.12, 0.32, 0.32)   # IoU ~0.68 com BOX
BOX_FAR = BoxN(0.7, 0.7, 0.9, 0.9)


# ── rótulos do matcher ───────────────────────────────────────────────────────


def test_exact_match_is_tp() -> None:
    r = match([P("car", 0.9, BOX)], [G("car", BOX)], iou_threshold=0.5)
    assert r.predictions[0].label is Label.TP
    assert r.ground_truth[0].label is GtLabel.MATCHED


def test_wrong_class_over_an_object_is_classification_error() -> None:
    # ALL_COCO explícito: sob GT_PRESENT a predição `truck` seria mascarada,
    # porque `truck` não existe neste gabarito de uma classe só.
    r = match(
        [P("truck", 0.9, BOX)], [G("car", BOX)],
        iou_threshold=0.5, class_scope=EvalClassScope.ALL_COCO,
    )
    assert r.predictions[0].label is Label.FP_CLASSIFICATION
    assert r.ground_truth[0].label is GtLabel.FN_FROM_ERRORS


def test_masking_applies_even_when_the_prediction_sits_on_an_object() -> None:
    """A regra do mascaramento não olha geometria, e isso é deliberado.

    Uma predição de classe ausente do gabarito é descartada mesmo caindo em cima
    de uma instância anotada. É a consequência de restringir a avaliação às
    classes que o gabarito cobre — e é por isso que a escolha precisa estar
    escrita no método, não escondida no matcher.
    """
    r = match(
        [P("truck", 0.9, BOX)], [G("car", BOX)],
        iou_threshold=0.5, class_scope=EvalClassScope.GT_PRESENT,
    )
    assert r.predictions == []
    assert r.masked_predictions == 1
    assert r.fp == 0
    # A instância do gabarito continua não detectada — o mascaramento perdoa o
    # falso positivo, nunca inventa um acerto.
    assert r.fn == 1


def test_right_class_poor_overlap_is_localization_error() -> None:
    r = match([P("car", 0.9, BOX_NEAR)], [G("car", BOX)], iou_threshold=0.9)
    assert r.predictions[0].label is Label.FP_LOCALIZATION
    assert r.ground_truth[0].label is GtLabel.FN_FROM_ERRORS


def test_wrong_class_and_poor_overlap_is_both() -> None:
    # Categoria "both" do TIDE: errou o quê E errou onde. Sem ela isto cairia
    # em "fundo" e a leitura viraria "alucinou objeto do nada".
    r = match(
        [P("truck", 0.9, BOX_NEAR)], [G("car", BOX)],
        iou_threshold=0.9, class_scope=EvalClassScope.ALL_COCO,
    )
    assert r.predictions[0].label is Label.FP_BOTH
    assert r.predictions[0].best_iou > 0.1
    assert r.ground_truth[0].label is GtLabel.FN_FROM_ERRORS


def test_localization_is_checked_before_class_errors() -> None:
    """Um objeto da própria classe por perto ganha de um de outra classe embaixo.

    As categorias do TIDE partem do IoU com a PRÓPRIA classe: só quando ele fica
    abaixo do limiar de fundo é que as outras classes entram na conversa.
    """
    r = match(
        [P("car", 0.9, BOX_NEAR)],
        [G("car", BOX), G("truck", BOX_NEAR)],
        iou_threshold=0.9,
        class_scope=EvalClassScope.ALL_COCO,
    )
    assert r.predictions[0].label is Label.FP_LOCALIZATION


def test_nothing_nearby_is_background() -> None:
    r = match([P("car", 0.9, BOX_FAR)], [G("car", BOX)], iou_threshold=0.5)
    assert r.predictions[0].label is Label.FP_BACKGROUND
    assert r.ground_truth[0].label is GtLabel.FN_UNDETECTED


def test_second_detection_on_same_object_is_duplicate() -> None:
    r = match(
        [P("car", 0.9, BOX), P("car", 0.4, BOX)], [G("car", BOX)], iou_threshold=0.5
    )
    labels = [m.label for m in sorted(r.predictions, key=lambda m: -m.pred.conf)]
    assert labels == [Label.TP, Label.FP_DUPLICATE]
    assert r.tp == 1 and r.fp == 1


def test_higher_confidence_wins_the_instance() -> None:
    # A de confiança menor chega primeiro na lista, mas o matching é por conf.
    r = match(
        [P("car", 0.2, BOX), P("car", 0.95, BOX)], [G("car", BOX)], iou_threshold=0.5
    )
    winner = next(m for m in r.predictions if m.label is Label.TP)
    assert winner.pred.conf == 0.95


def test_frames_do_not_leak_into_each_other() -> None:
    r = match(
        [P("car", 0.9, BOX, key="a:1")],
        [G("car", BOX, key="b:2")],
        iou_threshold=0.5,
    )
    assert r.predictions[0].label is Label.FP_BACKGROUND
    assert r.fn == 1


# ── mascaramento de classe ───────────────────────────────────────────────────


def test_gt_present_scope_masks_absent_classes() -> None:
    preds = [P("car", 0.9, BOX), P("traffic light", 0.8, BOX_FAR)]
    r = match(preds, [G("car", BOX)], iou_threshold=0.5,
              class_scope=EvalClassScope.GT_PRESENT)
    assert r.masked_predictions == 1
    assert r.masked_classes == {"traffic light": 1}
    assert len(r.predictions) == 1
    assert r.fp == 0


def test_all_coco_scope_turns_absent_classes_into_background_fp() -> None:
    preds = [P("car", 0.9, BOX), P("traffic light", 0.8, BOX_FAR)]
    r = match(preds, [G("car", BOX)], iou_threshold=0.5,
              class_scope=EvalClassScope.ALL_COCO)
    assert r.masked_predictions == 0
    assert r.fp == 1
    assert r.count(Label.FP_BACKGROUND) == 1


def test_masking_changes_only_false_positives() -> None:
    preds = [P("car", 0.9, BOX), P("stop sign", 0.8, BOX_FAR)]
    gts = [G("car", BOX)]
    masked = match(preds, gts, iou_threshold=0.5, class_scope=EvalClassScope.GT_PRESENT)
    full = match(preds, gts, iou_threshold=0.5, class_scope=EvalClassScope.ALL_COCO)
    assert masked.tp == full.tp
    assert masked.fn == full.fn
    assert full.fp - masked.fp == masked.masked_predictions == 1


# ── extração a partir do manifest ────────────────────────────────────────────


def test_gt_extraction_ignores_unreviewed_frames(
    test_frames: list[Frame], train_frames: list[Frame]
) -> None:
    assert len(gt_from_frames(test_frames)) == 735
    assert gt_from_frames(train_frames) == []
    assert len(gt_from_frames(test_frames + train_frames)) == 735


def test_stored_detections_extraction(test_frames: list[Frame]) -> None:
    assert len(stored_detections_as_predictions(test_frames)) == 505


# ── diff audit × eval ────────────────────────────────────────────────────────


def test_diff_compares_every_stored_detection(test_frames: list[Frame]) -> None:
    rep = audit_vs_eval(test_frames, iou_threshold=0.7)
    assert rep.compared_detections == 505


def test_diff_totals_agree_even_where_the_decomposition_does_not(
    test_frames: list[Frame],
) -> None:
    """Achado do dado real, congelado como teste.

    Os dois algoritmos concordam em tp/fp/fn e discordam em POR QUE cada erro é
    erro. Nenhuma detecção cruza a fronteira TP↔FP: as 42 divergências são todas
    entre categorias de falso positivo.
    """
    rep = audit_vs_eval(test_frames, iou_threshold=0.7)
    for k in ("detections", "ground_truth", "tp", "fp", "fn"):
        assert rep.audit_counters[k] == rep.eval_counters[k], k
    assert rep.audit_counters["fp_background"] != rep.eval_counters["fp_background"]
    assert not rep.agree

    for d in rep.divergences:
        assert d.audit_label != "tp" and d.eval_label != "tp", (
            f"{d.frame_key}: divergência cruzou a fronteira TP↔FP — o achado mudou"
        )


def test_diff_report_is_serializable(test_frames: list[Frame], tmp_path) -> None:
    import json

    rep = audit_vs_eval(test_frames, iou_threshold=0.7)
    path = rep.to_json(tmp_path / "diff.json")
    doc = json.loads(path.read_text(encoding="utf-8"))
    # A lista de casos é o entregável: precisa estar no arquivo, não só contada.
    assert len(doc["divergences"]) == len(rep.divergences)
    assert doc["divergences"][0]["frame_key"]
    assert doc["divergences"][0]["audit_label"]
    assert doc["by_transition"]


def test_diff_iou_threshold_matters(test_frames: list[Frame]) -> None:
    strict = audit_vs_eval(test_frames, iou_threshold=0.9)
    loose = audit_vs_eval(test_frames, iou_threshold=0.3)
    assert len(strict.divergences) != len(loose.divergences)


# ── view por hardlink ────────────────────────────────────────────────────────


def test_view_uses_hardlinks_not_copies(tmp_path) -> None:
    from forge.data.view import materialize_view

    src_img = tmp_path / "images" / "20240101120000" / "000042.jpg"
    src_img.parent.mkdir(parents=True)
    src_img.write_bytes(b"\xff\xd8\xff" + b"x" * 1000)
    src_lbl = tmp_path / "labels" / "20240101120000__42.txt"
    src_lbl.parent.mkdir(parents=True)
    src_lbl.write_text("2 0.5 0.5 0.1 0.1\n", encoding="utf-8")

    res = materialize_view(tmp_path / "view", "train", [src_img], {src_img: src_lbl})
    assert res.linked == 1

    linked = res.images_dir / "20240101120000__42.jpg"
    assert linked.is_file()
    # Hardlink de verdade: mesmo inode, um só bloco de dados no disco.
    assert linked.stat().st_ino == src_img.stat().st_ino
    assert (res.labels_dir / "20240101120000__42.txt").is_file()

    # O pareamento do ultralytics: trocar /images/ por /labels/ tem que achar.
    assert str(linked).replace("images", "labels").replace(".jpg", ".txt") == str(
        res.labels_dir / "20240101120000__42.txt"
    )


def test_view_skips_images_without_label(tmp_path) -> None:
    from forge.data.view import materialize_view

    img = tmp_path / "images" / "20240101120000" / "000007.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xff")

    res = materialize_view(tmp_path / "view", "train", [img], {})
    assert res.linked == 0
    assert res.skipped_missing_label == [str(img)]
    assert list(res.images_dir.iterdir()) == []


def test_view_is_idempotent(tmp_path) -> None:
    from forge.data.view import materialize_view

    img = tmp_path / "images" / "20240101120000" / "000042.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xff")
    lbl = tmp_path / "labels" / "x.txt"
    lbl.parent.mkdir(parents=True)
    lbl.write_text("", encoding="utf-8")

    first = materialize_view(tmp_path / "view", "train", [img], {img: lbl})
    second = materialize_view(tmp_path / "view", "train", [img], {img: lbl})
    assert (first.linked, first.reused) == (1, 0)
    assert (second.linked, second.reused) == (0, 1)


def test_view_refuses_cross_volume(tmp_path, monkeypatch) -> None:
    from forge.data import view as view_mod

    img = tmp_path / "images" / "20240101120000" / "000042.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xff")

    monkeypatch.setattr(view_mod, "_same_volume", lambda a, b: False)
    with pytest.raises(view_mod.ViewError, match="volumes diferentes"):
        view_mod.materialize_view(tmp_path / "view", "train", [img], {img: img})


def test_view_never_falls_back_to_copy(tmp_path, monkeypatch) -> None:
    """Se o hardlink falhar, o erro sobe — não vira cópia silenciosa."""
    import os

    from forge.data import view as view_mod

    img = tmp_path / "images" / "20240101120000" / "000042.jpg"
    img.parent.mkdir(parents=True)
    img.write_bytes(b"\xff\xd8\xff")
    lbl = tmp_path / "labels" / "x.txt"
    lbl.parent.mkdir(parents=True)
    lbl.write_text("", encoding="utf-8")

    def boom(*_args, **_kwargs):
        raise OSError("sem permissão")

    monkeypatch.setattr(os, "link", boom)
    with pytest.raises(view_mod.ViewError, match="hardlink"):
        view_mod.materialize_view(tmp_path / "view", "train", [img], {img: lbl})


# ── candidatos da segunda passada de curadoria ───────────────────────────────


def test_candidates_are_only_the_ones_with_nothing_annotated_under_them() -> None:
    """Só `fp_background` entra na fila.

    Uma predição que erra a caixa (ou a classe) sobre uma instância anotada não é
    ponto cego: o curador VIU aquele objeto. Misturá-las encheria a fila de
    trabalho com casos já resolvidos.
    """
    from forge.evaluate.candidates import find_candidates

    gts = [G("car", BoxN(0.1, 0.1, 0.3, 0.3))]
    preds = [
        P("car", 0.9, BoxN(0.6, 0.6, 0.8, 0.8)),   # nada por perto → candidato
        P("car", 0.8, BoxN(0.12, 0.12, 0.28, 0.3)),  # a mesma instância, torta
        P("car", 0.2, BoxN(0.0, 0.7, 0.1, 0.9)),   # confiança baixa → fora
    ]
    rep = find_candidates(preds, gts, min_conf=0.5, iou_threshold=0.5)

    assert len(rep.candidates) == 1
    assert rep.candidates[0].conf == 0.9
    assert rep.candidates[0].best_iou == 0.0
    assert rep.candidates[0].frame_key == "a:1"
    assert rep.predictions_considered == 2  # a de conf 0.2 nem foi considerada


def test_out_of_scope_classes_are_counted_apart_not_queued() -> None:
    """Classe que o gabarito nunca teve é escopo, não ponto cego."""
    from forge.evaluate.candidates import find_candidates

    gts = [G("car", BoxN(0.1, 0.1, 0.3, 0.3))]
    preds = [
        P("traffic light", 0.95, BoxN(0.6, 0.1, 0.65, 0.2)),
        P("car", 0.9, BoxN(0.6, 0.6, 0.8, 0.8)),
    ]
    rep = find_candidates(preds, gts, min_conf=0.5)
    assert [c.cls_name for c in rep.candidates] == ["car"]
    assert rep.out_of_scope_by_class == {"traffic light": 1}

    wide = find_candidates(preds, gts, min_conf=0.5, class_scope=EvalClassScope.ALL_COCO)
    assert {c.cls_name for c in wide.candidates} == {"car", "traffic light"}
    assert wide.out_of_scope_by_class == {}
    assert not next(c for c in wide.candidates if c.cls_name == "traffic light").in_scope
