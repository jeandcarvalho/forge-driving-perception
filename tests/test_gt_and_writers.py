"""As três regras que o revisor mandou tratar como requisito, não detalhe.

1. ground_truth null ≠ []  — e a exclusão é DA LISTA DE IMAGENS, não do .txt
2. espaço de classes COCO de 80, sem remapeamento
3. split `discarded` nunca entra em lista treinável
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from forge.api.models import Frame, ManifestPage
from forge.core.classes import COCO_CLASSES, class_id, class_name
from forge.data.layout import DatasetLayout
from forge.data.writers.coco import write_coco
from forge.data.writers.yolo import write_data_yaml, write_split_list, write_yolo_labels


@pytest.fixture
def layout(tmp_path: Path) -> DatasetLayout:
    lay = DatasetLayout(root=tmp_path / "ds")
    lay.ensure_dirs()
    return lay


def _fake_images(layout: DatasetLayout, frames: list[Frame]) -> None:
    """1920x1080 de verdade — o writer exige o JPEG em disco para listar."""
    for f in frames:
        p = layout.image_path(f.acq_id, f.sec)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080), (0, 0, 0)).save(p, "JPEG", quality=20)


# ── 1. null ≠ [] ─────────────────────────────────────────────────────────────


def test_null_and_empty_are_different_states(
    test_frames: list[Frame], train_frames: list[Frame]
) -> None:
    empty = [f for f in test_frames if f.ground_truth == []]
    assert empty, "a fixture precisa conter frames revisados-e-vazios"
    assert all(f.reviewed and f.is_labeled and f.is_empty_labeled for f in empty)

    assert all(f.ground_truth is None for f in train_frames)
    assert not any(f.is_labeled for f in train_frames)
    # O ponto: [] é falsy e None também. Quem escrever `if not gt` colapsa os dois
    # estados; a distinção só sobrevive testando `gt is None`.
    unlabeled, empty_labeled = None, []
    assert not unlabeled and not empty_labeled  # indistinguíveis por verdade
    assert (unlabeled is None) is not (empty_labeled is None)  # distinguíveis por identidade


def test_model_rejects_reviewed_without_ground_truth() -> None:
    with pytest.raises(ValueError, match="reviewed=true com ground_truth=null"):
        Frame.model_validate(
            {
                "acq_id": "20240103201734", "sec": 4, "split": "test", "reviewed": True,
                "image": None, "ground_truth": None, "curation_counts": None,
                "stored_detections": [],
            }
        )


def test_model_rejects_unreviewed_with_ground_truth() -> None:
    with pytest.raises(ValueError, match="reviewed=false com ground_truth não-null"):
        Frame.model_validate(
            {
                "acq_id": "20240103201734", "sec": 4, "split": "train", "reviewed": False,
                "image": None, "ground_truth": [], "curation_counts": None,
                "stored_detections": [],
            }
        )


def test_unreviewed_frame_is_excluded_from_the_image_list(
    layout: DatasetLayout, train_frames: list[Frame]
) -> None:
    """A asserção que importa.

    Não basta não escrever o .txt: no ultralytics, imagem sem label é contada
    como background. O frame não revisado tem que sumir da LISTA.
    """
    _fake_images(layout, train_frames)
    res = write_yolo_labels(layout, train_frames, "train")

    assert res.listed_images == []
    assert res.labels_written == 0
    assert res.excluded_unreviewed == len(train_frames)

    # E nenhum .txt órfão foi deixado para trás.
    assert list((layout.labels_dir / "train").glob("*.txt")) == []

    path = write_split_list(layout, res.listed_images, "train", trainable=False)
    assert path.name == "train_pool.txt"  # nome que nenhum data.yaml aponta
    assert path.read_text(encoding="utf-8") == ""


def test_reviewed_and_empty_gets_an_empty_txt_and_stays_in_the_list(
    layout: DatasetLayout, test_frames: list[Frame]
) -> None:
    _fake_images(layout, test_frames)
    res = write_yolo_labels(layout, test_frames, "test")

    empty = [f for f in test_frames if f.is_empty_labeled]
    assert len(empty) == 8

    for f in empty:
        p = layout.label_path("test", f.acq_id, f.sec)
        assert p.is_file(), "revisado-e-vazio precisa do .txt vazio"
        assert p.read_text(encoding="utf-8") == ""
        assert layout.image_path(f.acq_id, f.sec) in res.listed_images

    assert res.labels_written == 313
    assert res.empty_labels == 8
    # 731, e não 735: `airplane` (1) e `train` (3) saem por GT_EXCLUDED_CLASSES —
    # são rótulo trocado, não classe rara. Nenhum frame fica vazio por isso.
    assert res.instances == 731
    assert res.excluded_class == 4
    assert res.excluded_unreviewed == 0


def test_split_list_only_contains_frames_that_have_labels(
    layout: DatasetLayout, test_frames: list[Frame], train_frames: list[Frame]
) -> None:
    everything = test_frames + train_frames
    _fake_images(layout, everything)

    res_test = write_yolo_labels(layout, everything, "test")
    listed = {p.parent.name + ":" + str(int(p.stem)) for p in res_test.listed_images}
    labeled = {f.key for f in everything if f.is_labeled and f.split == "test"}
    assert listed == labeled


# ── 2. espaço de classes ─────────────────────────────────────────────────────


def test_class_space_is_coco80_not_remapped() -> None:
    assert len(COCO_CLASSES) == 80
    # Índices dos pesos COCO. Um remapeamento para o subconjunto do gabarito
    # daria car=1, truck=7 → e mataria a comparação do Experimento 2.
    assert class_id("person") == 0
    assert class_id("bicycle") == 1
    assert class_id("car") == 2
    assert class_id("motorcycle") == 3
    assert class_id("airplane") == 4
    assert class_id("bus") == 5
    assert class_id("train") == 6
    assert class_id("truck") == 7
    assert class_name(79) == "toothbrush"


def test_labels_use_coco80_indices(layout: DatasetLayout, test_frames: list[Frame]) -> None:
    _fake_images(layout, test_frames)
    write_yolo_labels(layout, test_frames, "test")

    seen: set[int] = set()
    for p in (layout.labels_dir / "test").glob("*.txt"):
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(int(line.split()[0]))
    # As classes do gabarito, nos índices COCO — nada em 0..7 contíguo por acaso.
    assert seen == {class_id(c) for c in
                    ("car", "truck", "bus", "motorcycle", "person", "bicycle")}
    assert max(seen) == class_id("truck") == 7
    assert 0 in seen  # person, e não "a primeira classe do subconjunto"
    # E as duas excluídas não voltam pela porta dos fundos.
    assert class_id("airplane") not in seen
    assert class_id("train") not in seen


def test_coco_categories_cover_all_80(layout: DatasetLayout, test_frames: list[Frame]) -> None:
    res = write_coco(layout, test_frames, "test")
    doc = json.loads(res.path.read_text(encoding="utf-8"))
    assert len(doc["categories"]) == 80
    assert doc["categories"][2] == {"id": 2, "name": "car", "supercategory": "coco80"}
    assert res.images == 313
    assert res.annotations == 731
    assert res.excluded_class == 4
    assert res.empty_images == 8


def test_coco_excludes_unreviewed_frames(
    layout: DatasetLayout, test_frames: list[Frame], train_frames: list[Frame]
) -> None:
    res = write_coco(layout, test_frames + train_frames, "train")
    assert res.images == 0
    assert res.excluded_unreviewed == len(train_frames)


# ── 3. discarded nunca treina ────────────────────────────────────────────────


def test_discarded_cannot_become_a_trainable_list(
    layout: DatasetLayout, discarded_frames: list[Frame]
) -> None:
    _fake_images(layout, discarded_frames)
    res = write_yolo_labels(layout, discarded_frames, "discarded")
    assert res.listed_images == []  # 100% não revisado, já sai vazio

    with pytest.raises(ValueError, match="nunca é treinável"):
        write_split_list(layout, [Path("qualquer.jpg")], "discarded", trainable=True)


def test_discarded_frames_are_labeled_as_such(discarded_frames: list[Frame]) -> None:
    assert all(f.split == "discarded" for f in discarded_frames)
    assert not any(f.is_labeled for f in discarded_frames)


def test_data_yaml_refuses_pool_lists(layout: DatasetLayout) -> None:
    pool = write_split_list(layout, [], "train", trainable=False)
    real = write_split_list(layout, [], "test", trainable=True)
    with pytest.raises(ValueError, match="lista de POOL"):
        write_data_yaml(layout, train_list=pool, val_list=real)


def test_data_yaml_lists_80_names(layout: DatasetLayout) -> None:
    a = write_split_list(layout, [], "test", trainable=True)
    path = write_data_yaml(layout, train_list=a, val_list=a)
    text = path.read_text(encoding="utf-8")
    assert "  0: person" in text
    assert "  79: toothbrush" in text
    assert "  2: car" in text


# ── sanidade da fixture ──────────────────────────────────────────────────────


def test_fixture_shape(test_page: ManifestPage) -> None:
    assert test_page.total == 313
    assert test_page.totals.model_dump() == {"test": 313, "train": 1406, "discarded": 2645}
    assert all(f.split == "test" for f in test_page.frames)
    assert all(f.image is not None for f in test_page.frames)
