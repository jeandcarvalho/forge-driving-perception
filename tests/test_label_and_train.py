"""Rotulagem automática e montagem do treino.

O que estes testes defendem, e por que cada um existe:

1. O `_pool.txt` do `pull` precisa ter conteúdo — ele é a lista de trabalho do
   `forge label`. Listar só os frames COM rótulo (que é o certo para a lista
   treinável) deixaria o pool de treino vazio, porque nenhum frame de treino é
   revisado, e o `label` não teria o que rotular.
2. Frame em que o professor não vê nada vira `.txt` vazio e FICA na lista. É o
   inverso da regra do gabarito, e de propósito: ver `detect/label.py`.
3. As duas pernas da ablação (0.25 e 0.5) coexistem em disco.
4. Nada aponta um treino para imagem sem rótulo pareável.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from forge.api.manifest import merge_frames
from forge.api.models import Frame, SampleResponse
from forge.api.sample import sample_to_frames
from forge.core.classes import class_id
from forge.data.download import DownloadReport, Outcome, read_checksums, write_checksums
from forge.data.layout import DatasetLayout, frame_stem
from forge.data.view import materialize_view
from forge.data.writers.yolo import (
    write_auto_split_list,
    write_data_yaml,
    write_split_list,
    write_yolo_labels,
)
from forge.detect.label import default_tag, format_label_lines, write_label_file


@pytest.fixture
def layout(tmp_path: Path) -> DatasetLayout:
    lay = DatasetLayout(root=tmp_path / "ds")
    lay.ensure_dirs()
    return lay


def _fake_images(layout: DatasetLayout, frames: list[Frame]) -> None:
    for f in frames:
        p = layout.image_path(f.acq_id, f.sec)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080), (0, 0, 0)).save(p, "JPEG", quality=20)


# ── 1. o pool precisa ter conteúdo ───────────────────────────────────────────


def test_pool_list_inventories_unreviewed_frames(
    layout: DatasetLayout, train_frames: list[Frame]
) -> None:
    """A lista treinável fica vazia (nenhum frame revisado); o inventário, não."""
    _fake_images(layout, train_frames)
    res = write_yolo_labels(layout, train_frames, "train")

    assert res.listed_images == []
    assert len(res.inventory_images) == len(train_frames)
    assert res.excluded_unreviewed == len(train_frames)

    pool = write_split_list(layout, res.inventory_images, "train", trainable=False)
    assert pool.name == "train_pool.txt"
    assert len(pool.read_text(encoding="utf-8").split()) == len(train_frames)


def test_inventory_skips_frames_without_image_on_disk(
    layout: DatasetLayout, train_frames: list[Frame]
) -> None:
    _fake_images(layout, train_frames[:3])
    res = write_yolo_labels(layout, train_frames, "train")
    assert len(res.inventory_images) == 3
    assert res.excluded_no_image == len(train_frames) - 3


def test_labeled_frames_stay_out_of_the_pool_but_in_the_trainable_list(
    layout: DatasetLayout, test_frames: list[Frame]
) -> None:
    _fake_images(layout, test_frames)
    res = write_yolo_labels(layout, test_frames, "test")
    # No split de teste todo frame é revisado: os dois conjuntos coincidem, e é
    # a lista treinável que o `pull` grava.
    assert res.listed_images == res.inventory_images
    assert len(res.listed_images) == 313


# ── 2. frame sem detecção é background, não abstenção ────────────────────────


def test_empty_teacher_output_is_a_written_empty_label() -> None:
    lines, dropped = format_label_lines([])
    assert lines == []
    assert dropped == []


def test_lines_use_coco80_indices_and_center_format() -> None:
    lines, dropped = format_label_lines(
        [("car", (0.5, 0.5, 0.2, 0.1)), ("person", (0.1, 0.2, 0.05, 0.4))]
    )
    assert dropped == []
    assert lines[0].split()[0] == str(class_id("car")) == "2"
    assert lines[1].split()[0] == str(class_id("person")) == "0"
    cx, cy, w, h = (float(v) for v in lines[0].split()[1:])
    assert (cx, cy, w, h) == (0.5, 0.5, 0.2, 0.1)


def test_boxes_that_overflow_the_border_are_clamped_not_dropped() -> None:
    # Objeto cortado pela borda: o detector extrapola, e o ultralytics recusa
    # label fora de 0..1. Grampear preserva a instância; descartar a perderia.
    lines, dropped = format_label_lines([("truck", (0.05, 0.5, 0.4, 0.2))])
    assert dropped == []
    cx, _, w, _ = (float(v) for v in lines[0].split()[1:])
    assert cx - w / 2 == pytest.approx(0.0)
    assert cx + w / 2 == pytest.approx(0.25)


def test_degenerate_and_unknown_boxes_are_dropped_with_a_reason() -> None:
    lines, dropped = format_label_lines(
        [
            ("car", (0.5, 0.5, 0.0, 0.1)),          # largura zero
            ("hovercraft", (0.5, 0.5, 0.2, 0.2)),   # fora do espaço COCO
            ("car", (-1.0, 0.5, 0.2, 0.2)),         # inteiramente fora
        ]
    )
    assert lines == []
    assert len(dropped) == 3
    assert "degenerada" in dropped[0]
    assert "COCO" in dropped[1]
    assert "fora da imagem" in dropped[2]


# ── 3. a ablação precisa de dois conjuntos vivos ao mesmo tempo ──────────────


def test_tag_carries_teacher_and_confidence() -> None:
    assert default_tag("yolo11x.pt", 0.25) == "yolo11x_c025"
    assert default_tag("yolo11x.pt", 0.5) == "yolo11x_c050"
    assert default_tag(Path("C:/pesos/yolo11x.pt"), 0.25) == "yolo11x_c025"


def test_both_ablation_legs_coexist(layout: DatasetLayout) -> None:
    img = layout.image_path("20240103201734", 4)
    img.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (1920, 1080)).save(img, "JPEG", quality=20)

    paths = []
    for conf, dets in ((0.25, [("car", (0.5, 0.5, 0.2, 0.1))]), (0.5, [])):
        tag = default_tag("yolo11x.pt", conf)
        lines, _ = format_label_lines(dets)
        write_label_file(layout.auto_label_path(tag, "20240103201734", 4), lines)
        paths.append(write_auto_split_list(layout, [img], tag, source_split="train"))

    assert paths[0] != paths[1]
    assert {p.name for p in paths} == {"auto_yolo11x_c025.txt", "auto_yolo11x_c050.txt"}
    assert layout.auto_label_path("yolo11x_c025", "20240103201734", 4).read_text().strip()
    # Corte alto e nenhuma detecção: .txt vazio, mas o frame CONTINUA na lista.
    assert layout.auto_label_path("yolo11x_c050", "20240103201734", 4).read_text() == ""
    assert img.resolve() == Path(paths[1].read_text(encoding="utf-8").strip())


def test_auto_list_refuses_discarded(layout: DatasetLayout) -> None:
    with pytest.raises(ValueError, match="nunca é treinável"):
        write_auto_split_list(layout, [], "yolo11x_c025", source_split="discarded")


# ── 4. nada treina em imagem sem rótulo pareável ─────────────────────────────


def test_view_pairs_by_explicit_map_and_skips_unlabeled(layout: DatasetLayout) -> None:
    tag = "yolo11x_c025"
    imgs = []
    for sec in (4, 9):
        p = layout.image_path("20240103201734", sec)
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (1920, 1080)).save(p, "JPEG", quality=20)
        imgs.append(p)
    # Só o primeiro tem rótulo.
    write_label_file(layout.auto_label_path(tag, "20240103201734", 4), ["2 0.5 0.5 0.2 0.1"])

    res = materialize_view(
        layout.view_dir(tag), "train", imgs,
        {p: layout.auto_labels_dir(tag) / f"{frame_stem(p)}.txt" for p in imgs},
    )
    assert res.linked == 1
    assert len(res.skipped_missing_label) == 1
    assert (res.images_dir / "20240103201734__4.jpg").is_file()
    assert (res.labels_dir / "20240103201734__4.txt").is_file()
    assert not (res.images_dir / "20240103201734__9.jpg").exists()


def test_data_yaml_refuses_a_view_without_labels(layout: DatasetLayout, tmp_path: Path) -> None:
    view = tmp_path / "view" / "train"
    (view / "images").mkdir(parents=True)
    (view / "labels").mkdir(parents=True)
    with pytest.raises(ValueError, match="não é erro"):
        write_data_yaml(layout, train_list=view / "images", val_list=view / "images")

    (view / "labels" / "a.txt").write_text("", encoding="utf-8")
    path = write_data_yaml(
        layout, train_list=view / "images", val_list=view / "images",
        path=tmp_path / "view" / "data.yaml",
    )
    assert path.is_file()
    assert "  2: car" in path.read_text(encoding="utf-8")


def test_data_yaml_still_refuses_pool_lists(layout: DatasetLayout) -> None:
    pool = write_split_list(layout, [], "train", trainable=False)
    ok = write_split_list(layout, [], "test", trainable=True)
    with pytest.raises(ValueError, match="lista de POOL"):
        write_data_yaml(layout, train_list=pool, val_list=ok)


# ── pull por split não pode apagar o que já estava ───────────────────────────


def test_partial_pull_keeps_frames_of_other_splits(
    test_frames: list[Frame], train_frames: list[Frame]
) -> None:
    """`--split train` não pode levar embora o gabarito.

    Sem isto, o pull do treino reescreveria o manifest só com os 1.406 frames
    não revisados, e o `instances_test.json` sairia com 0 imagens — o `eval`
    passaria a medir contra um gabarito vazio.
    """
    merged = merge_frames(train_frames, test_frames)
    assert len(merged) == len(train_frames) + len(test_frames)
    assert sum(1 for f in merged if f.split == "test") == len(test_frames)
    assert sum(1 for f in merged if f.is_labeled) == len(test_frames)
    # Ordem canônica: o arquivo não muda por causa da ordem dos pulls.
    assert merged == sorted(merged, key=lambda f: (f.acq_id, f.sec))
    assert merge_frames(test_frames, train_frames) == merged


def test_repulling_a_split_replaces_it_instead_of_duplicating(
    test_frames: list[Frame],
) -> None:
    merged = merge_frames(test_frames, test_frames)
    assert len(merged) == len(test_frames)
    assert len({f.key for f in merged}) == len(test_frames)


def test_checksums_accumulate_across_partial_pulls(layout: DatasetLayout) -> None:
    first = DownloadReport(
        outcomes=[Outcome("1:4", "20240103201734", 4, "downloaded", sha256="a" * 64)]
    )
    write_checksums(layout, first)

    second = DownloadReport(
        outcomes=[Outcome("1:9", "20240103201734", 9, "downloaded", sha256="b" * 64)]
    )
    write_checksums(layout, second)

    got = read_checksums(layout)
    assert got == {
        "images/20240103201734/000004.jpg": "a" * 64,
        "images/20240103201734/000009.jpg": "b" * 64,
    }


def test_failed_download_drops_its_checksum_entry(layout: DatasetLayout) -> None:
    write_checksums(
        layout,
        DownloadReport(
            outcomes=[Outcome("1:4", "20240103201734", 4, "downloaded", sha256="a" * 64)]
        ),
    )
    # Redownload que sai com dimensão errada: o caminho canônico deixa de
    # existir, e um hash órfão viraria falha de verify sem causa real.
    write_checksums(
        layout,
        DownloadReport(
            outcomes=[Outcome("1:4", "20240103201734", 4, "bad_size", sha256="c" * 64)]
        ),
    )
    assert read_checksums(layout) == {}


# ── braço do Experimento 1 → frames baixáveis ────────────────────────────────


def test_sample_frames_become_downloadable_frames() -> None:
    resp = SampleResponse.model_validate(
        {
            "mode": "query", "seed": 42, "n_requested": 2, "n_returned": 2,
            "exclude_collection": "<CARCARA_COLLECTION_ID>",
            "frames": [
                {
                    "acq_id": "20240103201734", "sec": 4,
                    "gps": {"lat": -3.7, "lon": -38.5},
                    "image": {
                        "drive_id": "ABC123", "url_original": "https://x/o",
                        "url_cdn": "https://x/c",
                    },
                },
                {"acq_id": "20240103201734", "sec": 9, "gps": None, "image": None},
            ],
        }
    )
    frames = sample_to_frames(resp)
    # Frame sem imagem não vira nada: não há o que baixar nem o que rotular.
    assert len(frames) == 1
    f = frames[0]
    assert f.image is not None and f.image.drive_id == "ABC123"
    # Sem gabarito, e o modelo tem que aceitar isso — é a invariante do manifest.
    assert f.reviewed is False
    assert f.ground_truth is None
    assert not f.is_labeled


# ── entrega ao ultralytics: lista de caminhos ≠ arquivo de lista ─────────────


def test_source_list_writes_one_resolved_path_per_line(tmp_path: Path) -> None:
    from forge.detect.source import source_list

    imgs = [tmp_path / "a" / "000004.jpg", tmp_path / "a" / "000009.jpg"]
    with source_list(imgs) as src:
        lines = Path(src).read_text(encoding="utf-8").split()
        assert lines == [str(p.resolve()) for p in imgs]
        written = Path(src)
    assert not written.exists()  # some junto com o diretório temporário


def test_txt_source_respects_batch_but_a_python_list_does_not(tmp_path: Path) -> None:
    """A regressão que custou uma OOM de 30 GiB.

    Com uma lista de caminhos o ultralytics escolhe `LoadPilAndNumpy` e faz
    `bs = len(source)`, ignorando `batch` — o conjunto inteiro vira um batch só.
    Com um `.txt`, cai no loader que lê sob demanda e respeita `batch`.
    """
    pytest.importorskip("ultralytics")
    from ultralytics.data.build import load_inference_source

    from forge.detect.source import source_list

    imgs = []
    for sec in range(5):
        p = tmp_path / "images" / "20240103201734" / f"{sec:06d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 64)).save(p, "JPEG")
        imgs.append(p)

    as_list = load_inference_source(source=[str(p) for p in imgs], batch=2)
    assert as_list.bs == len(imgs), "se isto mudar, o comentário de source.py envelheceu"

    with source_list(imgs) as src:
        as_txt = load_inference_source(source=src, batch=2)
    assert as_txt.bs == 2
    assert len(as_txt.files) == len(imgs)


# ── a semente precisa chegar no treino ───────────────────────────────────────


def test_seed_reaches_the_dataloader_generator() -> None:
    """Sem isto, `--seed 0/1/2` produz três modelos byte a byte idênticos.

    O ultralytics semeia `random`/`numpy`/`torch` no processo principal, mas o
    gerador do DataLoader — que em treino de 1 GPU governa a ordem dos dados E,
    via `worker_init_fn`, a augmentation de cada worker — leva
    `manual_seed(6148914691236517205 + RANK)`, constante no código-fonte. Com
    pesos pré-treinados e `deterministic=True` não sobra aleatoriedade nenhuma, e
    o desvio padrão de "3 sementes" sairia 0,000 por construção.
    """
    pytest.importorskip("ultralytics")
    import torch
    from ultralytics.models.yolo.detect import train as detect_train

    from forge.detect.train import seeded_dataloaders

    class FakeLoader:
        def __init__(self) -> None:
            self.generator = torch.Generator()
            self.generator.manual_seed(6148914691236517205)  # o valor do ultralytics

    original = detect_train.build_dataloader
    detect_train.build_dataloader = lambda *a, **k: FakeLoader()
    try:
        sem_patch = detect_train.build_dataloader()
        assert sem_patch.generator.initial_seed() == 6148914691236517205

        with seeded_dataloaders(7):
            com_patch = detect_train.build_dataloader()
        assert com_patch.generator.initial_seed() == 7

        # E devolve o módulo como estava: o patch é do treino, não do processo.
        assert detect_train.build_dataloader is not original
        assert detect_train.build_dataloader().generator.initial_seed() == 6148914691236517205
    finally:
        detect_train.build_dataloader = original
