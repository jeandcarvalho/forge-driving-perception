"""Paginação de `GET /public/collections/:id/export/manifest`.

Devolve as páginas cruas junto com as parseadas: o cru vai para `raw/` como
proveniência, porque é o único registro do que a API respondeu naquele dia.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from forge.api.client import ApiError, CarcaraClient
from forge.api.models import Frame, ManifestPage

MAX_PER_PAGE = 2000
DEFAULT_PER_PAGE = 500


def iter_manifest_pages(
    client: CarcaraClient,
    collection_id: str,
    *,
    split: str = "all",
    per_page: int = DEFAULT_PER_PAGE,
) -> Iterator[tuple[dict[str, Any], ManifestPage]]:
    """Itera (json cru, página parseada) até acabar."""
    if split not in {"test", "train", "discarded", "all"}:
        raise ValueError(f"split inválido: {split!r}")
    per_page = max(1, min(MAX_PER_PAGE, per_page))

    page = 1
    seen = 0
    while True:
        raw = client.get_json(
            f"/public/collections/{collection_id}/export/manifest",
            {"split": split, "page": page, "per_page": per_page},
        )
        parsed = ManifestPage.model_validate(raw)
        yield raw, parsed

        seen += len(parsed.frames)
        if not parsed.has_more:
            break
        if not parsed.frames:
            # has_more=true com página vazia seria loop infinito.
            raise ApiError(
                f"manifest devolveu página {page} vazia com has_more=true "
                f"(split={split}, total={parsed.total})"
            )
        page += 1

    if seen != parsed.total:
        raise ApiError(
            f"manifest incompleto: recebi {seen} frames, o servidor anunciou {parsed.total} "
            f"(split={split})"
        )


def fetch_manifest(
    client: CarcaraClient,
    collection_id: str,
    *,
    split: str = "all",
    per_page: int = DEFAULT_PER_PAGE,
    on_page: object = None,
) -> tuple[list[Frame], ManifestPage, list[dict[str, Any]]]:
    """Baixa o manifest inteiro.

    Retorna (frames, metadados da última página, páginas cruas).
    """
    frames: list[Frame] = []
    raws: list[dict[str, Any]] = []
    meta: ManifestPage | None = None

    for raw, parsed in iter_manifest_pages(
        client, collection_id, split=split, per_page=per_page
    ):
        frames.extend(parsed.frames)
        raws.append(raw)
        meta = parsed
        if callable(on_page):
            on_page(parsed)

    if meta is None:  # pragma: no cover — iter_manifest_pages sempre rende ≥1
        raise ApiError("manifest não devolveu nenhuma página")

    _assert_unique_keys(frames)
    return frames, meta, raws


def merge_frames(pulled: list[Frame], previous: list[Frame]) -> list[Frame]:
    """Une o que este pull trouxe com o que já estava no manifest em disco.

    O pull é por split — ninguém baixa as 4.364 imagens da coleção para treinar
    em 1.406 — então sobrescrever o manifest a cada pull apagaria os frames dos
    outros splits, e com o do teste iria junto o `instances_test.json` que o
    `eval` usa como gabarito. O frame recém-vindo da API ganha do que estava em
    disco: ele é mais novo.

    Ordem canônica (acq_id, sec) para que o arquivo não mude só por causa da
    ordem em que os splits foram baixados.
    """
    pulled_keys = {f.key for f in pulled}
    merged = pulled + [f for f in previous if f.key not in pulled_keys]
    return sorted(merged, key=lambda f: (f.acq_id, f.sec))


def _assert_unique_keys(frames: list[Frame]) -> None:
    """Um frame é (acq_id, sec). Chave repetida = paginação instável ou dado duplicado."""
    seen: set[str] = set()
    dupes: list[str] = []
    for f in frames:
        if f.key in seen:
            dupes.append(f.key)
        seen.add(f.key)
    if dupes:
        raise ApiError(
            f"{len(dupes)} frame(s) duplicado(s) no manifest, ex.: {dupes[:5]}. "
            "Paginação instável — o dataset materializado não seria reprodutível."
        )
