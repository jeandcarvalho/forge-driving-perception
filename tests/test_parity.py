"""O critério de aceite: a dobra do gabarito no Forge == a da API.

Roda contra fixture congelada, então não depende da API estar de pé nem de
alguém abrir o curador e mexer no dado.
"""

from __future__ import annotations

from forge.api.models import Frame
from forge.evaluate.parity import (
    EXPECTED_TARGET_COLLECTION,
    curation_parity,
    gt_class_histogram,
    js_round,
)

#: PROMPT.md §3 — 735 instâncias.
EXPECTED_HISTOGRAM = {
    "car": 427, "truck": 255, "bus": 23, "motorcycle": 13,
    "person": 10, "bicycle": 3, "train": 3, "airplane": 1,
}


def test_parity_reproduces_api_stats(test_frames: list[Frame]) -> None:
    rep = curation_parity(test_frames)
    assert rep.diff_against(EXPECTED_TARGET_COLLECTION) == {}


def test_parity_core_counters(test_frames: list[Frame]) -> None:
    rep = curation_parity(test_frames)
    assert (rep.tp, rep.fp, rep.fn, rep.ground_truth) == (333, 172, 402, 735)


def test_parity_ratios_match_js_rounding(test_frames: list[Frame]) -> None:
    # A API arredonda com Math.round (meio para cima); Python arredonda para par.
    # Se alguém trocar js_round por round(), estes números mudam.
    rep = curation_parity(test_frames)
    assert (rep.precision, rep.recall, rep.f1) == (65.9, 45.3, 53.7)


def test_js_round_is_half_up_not_banker() -> None:
    assert js_round(0.5) == 1  # round() do Python daria 0
    assert js_round(1.5) == 2
    assert js_round(2.5) == 3  # round() do Python daria 2


def test_derivation_agrees_with_api_curation_counts(test_frames: list[Frame]) -> None:
    # Deriva de stored_detections + ground_truth e confere contra o
    # curation_counts que a API mandou. Dois caminhos, um resultado.
    rep = curation_parity(test_frames)
    assert rep.count_mismatches == []


def test_ground_truth_class_histogram(test_frames: list[Frame]) -> None:
    assert gt_class_histogram(test_frames) == EXPECTED_HISTOGRAM
    assert sum(EXPECTED_HISTOGRAM.values()) == 735


def test_identities_hold(test_frames: list[Frame]) -> None:
    rep = curation_parity(test_frames)
    assert rep.tp + rep.fp == rep.detections
    assert rep.tp + rep.fn == rep.ground_truth
    assert rep.fp == rep.fp_localization + rep.fp_classification + rep.fp_background
    assert rep.fn == rep.fn_undetected + rep.fn_from_errors


def test_unreviewed_frames_do_not_enter_parity(
    test_frames: list[Frame], train_frames: list[Frame]
) -> None:
    # stats.ts conta só sobre frames revisados. Misturar train (100% não
    # revisado) não pode mover um único contador.
    a = curation_parity(test_frames)
    b = curation_parity(test_frames + train_frames)
    assert a.detections == b.detections
    assert a.ground_truth == b.ground_truth
    assert (a.tp, a.fp, a.fn) == (b.tp, b.fp, b.fn)
    assert b.reviewed_frames == a.reviewed_frames
