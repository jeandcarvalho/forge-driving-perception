"""Como entregar uma lista de imagens ao ultralytics sem estourar a memória.

`model.predict(source=[caminho, caminho, ...])` parece a forma óbvia e é uma
armadilha: com uma **lista**, o ultralytics escolhe o loader `LoadPilAndNumpy` e
monta um batch único com o conteúdo inteiro — `dataset.bs == len(source)` — e o
parâmetro `batch=` é ignorado em silêncio.

Com 313 imagens e um modelo pequeno isso passa despercebido (foi assim que o
`forge infer` rodou). Com 1.406 imagens 1920x1080 e o YOLO11x, não passa: a RAM
estoura no PIL ou a VRAM pede 30 GiB num `warmup` e o processo morre. E o modo de
falha é enganoso — o erro aparece na primeira iteração, como se fosse o modelo
não caber na GPU, quando o que não cabe é o batch que ninguém pediu.

Um arquivo `.txt` com um caminho por linha cai no `LoadImagesAndVideos`, que
respeita `batch` e lê sob demanda. O custo é um arquivo temporário; o ganho é que
o consumo de memória para de depender do tamanho do conjunto.
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def write_source_list(images: list[Path], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(Path(p).resolve()) for p in images) + ("\n" if images else ""),
        encoding="utf-8",
    )
    return path


@contextmanager
def source_list(images: list[Path]) -> Iterator[str]:
    """Lista de imagens → caminho de um `.txt` temporário, removido no fim."""
    tmp = Path(tempfile.mkdtemp(prefix="forge-source-")) / "images.txt"
    try:
        yield str(write_source_list(images, tmp))
    finally:
        tmp.unlink(missing_ok=True)
        tmp.parent.rmdir()
