# HISTORICAL REFERENCE -- NOT EXECUTABLE FROM THIS REPOSITORY.
# Downloads the images of the extra random arm from the private storage using file identifiers that
# are not distributed (frame-list JSON with Drive ids). Kept only to document how those images were
# obtained. Moved here from pillar1/; the code below is unchanged except for the path edits listed in
# PUBLIC_RELEASE_CHANGES.md. Its `sys.path` line no longer finds `src/` from this location (intended).
"""Baixa as imagens que faltam para o braco NOVO (random_30m) do Pillar 1.

Aditivo: escreve so imagens que ainda nao existem, no caminho canonico
`images/<acq_id>/<sec:06d>.jpg`. NAO toca em manifest.json, arms/, labels/ nem
checksums.sha256 — nada antigo e alterado.

Qualidade ORIGINAL (nunca --fast): o CDN recomprime o JPEG e mudaria o numero.
Mesmo gate de entrada do `forge pull`: os bytes tem que abrir como imagem e medir
exatamente 1920x1080, senao o arquivo nao recebe o nome canonico.
"""

from __future__ import annotations

import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import local

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from forge.config import EXPECTED_IMAGE_SIZE  # noqa: E402
from forge.data.drive import DriveFetcher  # noqa: E402

import os
LISTS = Path(os.environ.get("FORGE_P1_FRAMELISTS", "pillar1-content-framelists-s42-51.json"))  # private: contains Drive ids
IMAGES = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one" / "images"
REPORT = Path(__file__).resolve().parent / "pillar1_content_fetch_report_s42-51.json"
WORKERS = 20

_tls = local()


def fetcher() -> DriveFetcher:
    f = getattr(_tls, "f", None)
    if f is None:
        f = DriveFetcher(fast=False)
        _tls.f = f
    return f


def path_of(acq_id: str, sec: int) -> Path:
    return IMAGES / acq_id / f"{sec:06d}.jpg"


def one(item: dict) -> dict:
    acq, sec, did = item["acq_id"], item["sec"], item["drive_id"]
    final = path_of(acq, sec)
    if final.is_file():
        return {"key": f"{acq}/{sec}", "status": "cached"}
    if not did:
        return {"key": f"{acq}/{sec}", "status": "no_image"}
    try:
        data = fetcher().fetch(did)
    except Exception as exc:  # noqa: BLE001
        return {"key": f"{acq}/{sec}", "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except Exception as exc:  # noqa: BLE001
        return {"key": f"{acq}/{sec}", "status": "failed", "error": f"nao abre como imagem: {exc}"}
    if (w, h) != EXPECTED_IMAGE_SIZE:
        final.parent.mkdir(parents=True, exist_ok=True)
        final.with_suffix(".badsize.jpg").write_bytes(data)
        return {"key": f"{acq}/{sec}", "status": "bad_size", "wh": [w, h]}
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(final)
    return {"key": f"{acq}/{sec}", "status": "downloaded", "bytes": len(data)}


def main() -> None:
    d = json.loads(LISTS.read_text(encoding="utf-8"))
    need: dict[str, dict] = {}
    for arm in d["arms"]:
        for f in arm["frames"]:
            k = f'{f["acq_id"]}/{f["sec"]}'
            if k not in need and not path_of(f["acq_id"], f["sec"]).is_file():
                need[k] = f
    items = list(need.values())
    print(f"frames distintos faltando em disco: {len(items)}")
    if not items:
        print("nada a baixar")
        return

    outcomes = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futs = [pool.submit(one, it) for it in items]
        for fut in tqdm(as_completed(futs), total=len(futs), unit="img", desc="baixando"):
            outcomes.append(fut.result())

    counts: dict[str, int] = {}
    for o in outcomes:
        counts[o["status"]] = counts.get(o["status"], 0) + 1
    print("resultado:", counts)
    total_mb = sum(o.get("bytes", 0) for o in outcomes) / 1024 / 1024
    print(f"baixado: {total_mb:.1f} MB")
    REPORT.write_text(json.dumps(
        {"requested": len(items), "counts": counts, "downloaded_mb": total_mb,
         "failures": [o for o in outcomes if o["status"] not in ("downloaded", "cached")]},
        indent=2), encoding="utf-8")
    print(f"-> {REPORT}")


if __name__ == "__main__":
    main()
