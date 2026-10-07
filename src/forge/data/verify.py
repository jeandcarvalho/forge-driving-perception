"""`forge verify` — revalida o que o `pull` deixou no disco.

Checa cinco coisas, e as três últimas são invariantes de método, não de arquivo:

1. checksum de cada imagem bate com checksums.sha256
2. dimensão medida é 1920x1080
3. toda caixa do gabarito é geometricamente válida e cabe dentro da imagem
4. nenhuma imagem sem gabarito aparece numa lista de split treinável
   (no ultralytics, imagem sem label é contada como background)
5. nenhum frame do split `discarded` vazou para uma lista treinável
   (aquisição contaminada pela regra de antivazamento)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from forge.api.models import Frame
from forge.config import EXPECTED_IMAGE_SIZE
from forge.core.geometry import is_valid_box_n
from forge.data.download import read_checksums
from forge.data.layout import POOL_LIST_SUFFIX, DatasetLayout


@dataclass(slots=True)
class VerifyReport:
    checked_images: int = 0
    checked_frames: int = 0
    checksum_mismatch: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)
    bad_dimensions: list[str] = field(default_factory=list)
    bad_boxes: list[str] = field(default_factory=list)
    unlabeled_in_split: list[str] = field(default_factory=list)
    discarded_leak: list[str] = field(default_factory=list)

    @property
    def failures(self) -> dict[str, list[str]]:
        return {
            k: v
            for k, v in {
                "checksum_mismatch": self.checksum_mismatch,
                "missing_files": self.missing_files,
                "bad_dimensions": self.bad_dimensions,
                "bad_boxes": self.bad_boxes,
                "unlabeled_in_split": self.unlabeled_in_split,
                "discarded_leak": self.discarded_leak,
            }.items()
            if v
        }

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def total_failures(self) -> int:
        return sum(len(v) for v in self.failures.values())


def verify(
    layout: DatasetLayout,
    frames: list[Frame],
    *,
    check_hashes: bool = True,
) -> VerifyReport:
    rep = VerifyReport()
    by_key = {f.key: f for f in frames}

    # ── 1 e 2: arquivos ──────────────────────────────────────────────────────
    expected = read_checksums(layout)
    for rel, sha in expected.items():
        path = layout.root / rel
        if not path.is_file():
            rep.missing_files.append(rel)
            continue
        rep.checked_images += 1
        data = path.read_bytes()
        if check_hashes and hashlib.sha256(data).hexdigest() != sha:
            rep.checksum_mismatch.append(rel)
        try:
            with Image.open(path) as im:
                if im.size != EXPECTED_IMAGE_SIZE:
                    rep.bad_dimensions.append(f"{rel}: {im.size[0]}x{im.size[1]}")
        except Exception as exc:  # noqa: BLE001
            rep.bad_dimensions.append(f"{rel}: não abre — {exc}")

    # ── 3: geometria do gabarito ─────────────────────────────────────────────
    for f in frames:
        rep.checked_frames += 1
        for i, g in enumerate(f.ground_truth or []):
            box = g.box_n.to_box() if g.box_n is not None else None
            if not is_valid_box_n(box):
                rep.bad_boxes.append(f"{f.key}[{i}] {g.cls_name}: caixa inválida {g.box_n}")
                continue
            # Caixa degenerada em px: some ao projetar na imagem.
            w_px = box.width * EXPECTED_IMAGE_SIZE[0]
            h_px = box.height * EXPECTED_IMAGE_SIZE[1]
            if w_px < 1 or h_px < 1:
                rep.bad_boxes.append(
                    f"{f.key}[{i}] {g.cls_name}: caixa degenerada ({w_px:.2f}x{h_px:.2f} px)"
                )

    # ── 4 e 5: listas de split ───────────────────────────────────────────────
    if layout.splits_dir.is_dir():
        for list_path in sorted(layout.splits_dir.glob("*.txt")):
            trainable = not list_path.name.endswith(POOL_LIST_SUFFIX)
            if not trainable:
                continue
            for line in list_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                key = _key_from_image_path(Path(line))
                if key is None:
                    rep.unlabeled_in_split.append(f"{list_path.name}: caminho ilegível {line}")
                    continue
                frame = by_key.get(key)
                if frame is None:
                    rep.unlabeled_in_split.append(f"{list_path.name}: {key} não está no manifest")
                    continue
                if not frame.is_labeled:
                    rep.unlabeled_in_split.append(
                        f"{list_path.name}: {key} não tem gabarito — no ultralytics viraria "
                        "background"
                    )
                if frame.split == "discarded":
                    rep.discarded_leak.append(
                        f"{list_path.name}: {key} é do split discarded (aquisição contaminada)"
                    )

    path = layout.verify_failures_path
    path.write_text(
        json.dumps(
            {
                "checked_images": rep.checked_images,
                "checked_frames": rep.checked_frames,
                "total_failures": rep.total_failures,
                "failures": rep.failures,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return rep


def _key_from_image_path(path: Path) -> str | None:
    """images/<acq_id>/<sec:06d>.jpg → "<acq_id>:<sec>"."""
    try:
        return f"{path.parent.name}:{int(path.stem)}"
    except (ValueError, IndexError):
        return None
