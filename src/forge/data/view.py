"""View no formato que o ultralytics espera, por HARDLINK.

O layout de arquivo do Forge (`images/<acq_id>/<sec>.jpg` +
`labels/yolo/<split>/<acq>__<sec>.txt`) é bom como acervo e não é o que o
ultralytics pareia sozinho: ele deriva o caminho do rótulo trocando `/images/`
por `/labels/` no caminho da imagem. Então o treino precisa de uma view com essa
forma.

**Hardlink, nunca cópia.** No Experimento 2 são até 80 mil JPEG; duplicar isso
custa dezenas de GB à toa, e o acervo já mora em D:\\carcara-data\\ — origem e
destino no mesmo volume, que é a única exigência do hardlink no Windows. Não
precisa de privilégio de administrador (symlink precisaria de developer mode,
por isso não usamos).

**Se o hardlink falhar, falha alto.** Cópia silenciosa transformaria um erro de
configuração em dezenas de GB consumidos sem ninguém perceber.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from forge.data.layout import frame_stem


class ViewError(RuntimeError):
    pass


@dataclass(slots=True)
class ViewResult:
    root: Path
    images_dir: Path
    labels_dir: Path
    linked: int = 0
    reused: int = 0
    skipped_missing_label: list[str] = field(default_factory=list)


def _same_volume(a: Path, b: Path) -> bool:
    return os.path.splitdrive(a.resolve())[0].lower() == os.path.splitdrive(b.resolve())[0].lower()


def _hardlink(src: Path, dst: Path) -> bool:
    """True se criou, False se já existia. Erro se não deu para linkar."""
    if dst.exists():
        return False
    try:
        os.link(src, dst)
    except OSError as exc:
        raise ViewError(
            f"não consegui criar hardlink {dst} → {src}: {exc}. "
            "Origem e destino precisam estar no MESMO volume. "
            "O Forge não cai para cópia de propósito: seriam dezenas de GB "
            "duplicados em silêncio."
        ) from exc
    return True


def materialize_view(
    view_root: Path,
    split: str,
    images: list[Path],
    label_for: dict[Path, Path],
) -> ViewResult:
    """Monta `<view_root>/<split>/{images,labels}/` por hardlink.

    `label_for` mapeia cada imagem ao seu arquivo de rótulo. Imagem sem rótulo
    no mapa NÃO entra: no ultralytics ela seria contada como background.
    """
    root = Path(view_root) / split
    images_dir = root / "images"
    labels_dir = root / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    res = ViewResult(root=root, images_dir=images_dir, labels_dir=labels_dir)

    for img in images:
        if not _same_volume(img, images_dir):
            raise ViewError(
                f"{img} e {images_dir} estão em volumes diferentes — hardlink é "
                "impossível entre volumes. Aponte a view para o mesmo disco do "
                "dataset (D:\\carcara-data\\)."
            )
        label = label_for.get(img)
        if label is None or not label.is_file():
            # Sem rótulo em disco a imagem viraria background silencioso.
            res.skipped_missing_label.append(str(img))
            continue

        stem = frame_stem(img)
        created = _hardlink(img, images_dir / f"{stem}{img.suffix}")
        _hardlink(label, labels_dir / f"{stem}.txt")
        if created:
            res.linked += 1
        else:
            res.reused += 1

    return res
