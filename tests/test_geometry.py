from __future__ import annotations

import pytest

from forge.core.geometry import BoxN, iou, is_valid_box_n


def test_iou_identical_is_one() -> None:
    b = BoxN(0.1, 0.1, 0.5, 0.5)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_disjoint_is_zero() -> None:
    assert iou(BoxN(0.0, 0.0, 0.1, 0.1), BoxN(0.5, 0.5, 0.6, 0.6)) == 0.0


def test_iou_touching_edges_is_zero() -> None:
    assert iou(BoxN(0.0, 0.0, 0.5, 0.5), BoxN(0.5, 0.0, 1.0, 0.5)) == 0.0


def test_iou_half_overlap() -> None:
    a = BoxN(0.0, 0.0, 0.2, 0.1)
    b = BoxN(0.1, 0.0, 0.3, 0.1)
    # interseção 0.1x0.1, união 0.02+0.02-0.01
    assert iou(a, b) == pytest.approx(1 / 3)


def test_iou_is_symmetric() -> None:
    a, b = BoxN(0.1, 0.2, 0.6, 0.7), BoxN(0.3, 0.1, 0.8, 0.5)
    assert iou(a, b) == pytest.approx(iou(b, a))


def test_yolo_roundtrip() -> None:
    b = BoxN(0.2, 0.3, 0.6, 0.9)
    back = BoxN.from_yolo_cxcywh(*b.to_yolo_cxcywh())
    for got, want in zip((back.x1, back.y1, back.x2, back.y2), (b.x1, b.y1, b.x2, b.y2)):
        assert got == pytest.approx(want)


def test_coco_xywh_uses_measured_dims() -> None:
    b = BoxN(0.5, 0.5, 1.0, 1.0)
    assert b.to_coco_xywh(1920, 1080) == [960.0, 540.0, 960.0, 540.0]


def test_invalid_boxes_are_rejected() -> None:
    assert not is_valid_box_n(None)
    assert not is_valid_box_n(BoxN(0.5, 0.1, 0.2, 0.9))  # x2 < x1
    assert not is_valid_box_n(BoxN(0.1, 0.1, 0.1, 0.9))  # largura zero
    assert not is_valid_box_n(BoxN(-0.1, 0.1, 0.5, 0.9))  # fora de 0..1
    assert not is_valid_box_n(BoxN(0.1, 0.1, 1.5, 0.9))
    assert is_valid_box_n(BoxN(0.0, 0.0, 1.0, 1.0))
