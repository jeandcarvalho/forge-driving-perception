"""Download concorrente das imagens, com verificação na porta de entrada.

Nada é aceito no disco antes de passar por duas checagens: os bytes são JPEG e a
dimensão medida é 1920x1080. O manifest deriva width/height de px ÷ normalizado e
pode mentir (ou vir null) — quem manda é o arquivo.

Arquivo que falha a dimensão NÃO recebe o nome final: fica como `.badsize` ao
lado. Assim ele existe como evidência, e ao mesmo tempo todo o resto do pipeline
o exclui naturalmente, porque o caminho canônico não existe.
"""

from __future__ import annotations

import hashlib
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from threading import local

from PIL import Image
from tqdm import tqdm

from forge.api.models import Frame
from forge.config import EXPECTED_IMAGE_SIZE
from forge.data.drive import DriveError, DriveFetcher
from forge.data.layout import DatasetLayout

Status = str  # "downloaded" | "cached" | "no_image" | "bad_size" | "failed"


@dataclass(slots=True)
class Outcome:
    key: str
    acq_id: str
    sec: int
    status: Status
    path: Path | None = None
    sha256: str | None = None
    width: int | None = None
    height: int | None = None
    error: str | None = None


@dataclass(slots=True)
class DownloadReport:
    outcomes: list[Outcome] = field(default_factory=list)

    def by_status(self, status: Status) -> list[Outcome]:
        return [o for o in self.outcomes if o.status == status]

    @property
    def counts(self) -> dict[str, int]:
        c: dict[str, int] = {}
        for o in self.outcomes:
            c[o.status] = c.get(o.status, 0) + 1
        return c

    @property
    def dims_by_key(self) -> dict[str, tuple[int, int]]:
        return {
            o.key: (o.width, o.height)
            for o in self.outcomes
            if o.width and o.height and o.status in ("downloaded", "cached")
        }

    @property
    def ok(self) -> list[Outcome]:
        return [o for o in self.outcomes if o.status in ("downloaded", "cached")]


_tls = local()


def _fetcher(fast: bool) -> DriveFetcher:
    """Um DriveFetcher por thread — o cookie jar do Drive não é compartilhável."""
    f = getattr(_tls, "fetcher", None)
    if f is None:
        f = DriveFetcher(fast=fast)
        _tls.fetcher = f
    return f


def _measure(data: bytes) -> tuple[int, int]:
    with Image.open(io.BytesIO(data)) as im:
        return im.size


def _one(frame: Frame, layout: DatasetLayout, *, fast: bool, force: bool) -> Outcome:
    key, acq, sec = frame.key, frame.acq_id, frame.sec

    if frame.image is None:
        return Outcome(key, acq, sec, "no_image", error="frame sem link de imagem no manifest")

    final = layout.image_path(acq, sec)
    if final.is_file() and not force:
        try:
            data = final.read_bytes()
            w, h = _measure(data)
            return Outcome(
                key, acq, sec, "cached", final, hashlib.sha256(data).hexdigest(), w, h
            )
        except Exception:  # noqa: BLE001 — qualquer falha aqui é cache podre
            final.unlink(missing_ok=True)  # cache corrompido: refaz

    try:
        data = _fetcher(fast).fetch(frame.image.drive_id)
    except (DriveError, Exception) as exc:  # noqa: BLE001 — vira linha de relatório
        return Outcome(key, acq, sec, "failed", error=f"{type(exc).__name__}: {exc}")

    try:
        w, h = _measure(data)
    except Exception as exc:  # noqa: BLE001
        return Outcome(key, acq, sec, "failed", error=f"bytes não abrem como imagem: {exc}")

    final.parent.mkdir(parents=True, exist_ok=True)

    if (w, h) != EXPECTED_IMAGE_SIZE:
        # Não recebe o nome canônico: fica como evidência e sai do pipeline.
        bad = final.with_suffix(".badsize.jpg")
        bad.write_bytes(data)
        return Outcome(
            key, acq, sec, "bad_size", bad, hashlib.sha256(data).hexdigest(), w, h,
            error=(
                f"dimensão medida {w}x{h}, esperada "
                f"{EXPECTED_IMAGE_SIZE[0]}x{EXPECTED_IMAGE_SIZE[1]}"
                + (
                    f" (manifest declarava {frame.image.width}x{frame.image.height})"
                    if frame.image.width
                    else " (manifest não declarava dimensão)"
                )
            ),
        )

    tmp = final.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(final)
    return Outcome(key, acq, sec, "downloaded", final, hashlib.sha256(data).hexdigest(), w, h)


def download_frames(
    layout: DatasetLayout,
    frames: list[Frame],
    *,
    workers: int = 6,
    fast: bool = False,
    force: bool = False,
    progress: bool = True,
) -> DownloadReport:
    report = DownloadReport()
    layout.images_dir.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, f, layout, fast=fast, force=force): f for f in frames}
        bar = tqdm(
            as_completed(futures),
            total=len(futures),
            disable=not progress,
            unit="img",
            desc="baixando",
        )
        for fut in bar:
            report.outcomes.append(fut.result())

    return report


def write_checksums(layout: DatasetLayout, report: DownloadReport, *, merge: bool = True) -> Path:
    """Grava `checksums.sha256`, preservando o que veio de outros pulls.

    `merge=True` porque o pull é por split: um `--split train` com sobrescrita
    apagaria os checksums das 313 imagens do teste, e o `forge verify` passaria
    a dizer "tudo íntegro" sobre um arquivo que não confere mais nada. Silêncio
    é o pior modo de falha possível para uma verificação de integridade.

    Frame que falhou (ou saiu com dimensão errada) tem a entrada REMOVIDA: o
    caminho canônico não existe mais, e um hash órfão viraria falha de verify.
    """
    entries = read_checksums(layout) if merge else {}
    for o in report.outcomes:
        rel = f"images/{o.acq_id}/{o.sec:06d}.jpg"
        if o.sha256 and o.status in ("downloaded", "cached"):
            entries[rel] = o.sha256
        else:
            entries.pop(rel, None)

    lines = [f"{h}  {rel}" for rel, h in sorted(entries.items())]
    path = layout.checksums_path
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def read_checksums(layout: DatasetLayout) -> dict[str, str]:
    """{'images/<acq>/<sec>.jpg': sha256}"""
    path = layout.checksums_path
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("  ", 1)
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    return out
