"""`GET /api/dataset/sample` — os três braços do Experimento 1.

O `exclude_collection` NÃO é opcional aqui. Sem ele, o pool de amostragem inclui
as 127 aquisições que doaram frame ao gabarito, e o treino contamina a avaliação.
A assinatura exige o parâmetro para que esquecê-lo seja um erro de digitação, não
um default silencioso.
"""

from __future__ import annotations

from typing import Any, Literal

from forge.api.client import CarcaraClient
from forge.api.models import Frame, ImageRef, SampleResponse

Mode = Literal["random", "query"]


def sample_frames(
    client: CarcaraClient,
    *,
    mode: Mode,
    n: int,
    seed: int,
    exclude_collection: str,
    dedup_m: float | None = None,
    exclude_mode: str = "acqs",
    spread: bool = False,
    filters: dict[str, Any] | None = None,
) -> SampleResponse:
    params: dict[str, Any] = {
        "mode": mode,
        "n": n,
        "seed": seed,
        "exclude_collection": exclude_collection,
        "exclude_mode": exclude_mode,
        "dedup_m": dedup_m,
        # Rodízio entre aquisições: um frame de cada gravação por volta, até N.
        # É a alavanca de diversidade — ver o comentário em SampleResponse.
        "spread": "acq" if spread else None,
        **(filters or {}),
    }
    raw = client.get_json("/api/dataset/sample", params)
    return SampleResponse.model_validate(raw)


def sample_to_frames(resp: SampleResponse) -> list[Frame]:
    """Frames de um braço no formato que o downloader consome.

    `/api/dataset/sample` devolve menos campos que o manifest — ele é uma lista
    de seleção, não um export. O que falta é justamente o que um frame de braço
    não tem mesmo: gabarito (`reviewed=False`, `ground_truth=None`) e split de
    coleção.

    O `split="train"` é uma etiqueta de conveniência para o downloader, não uma
    afirmação sobre a coleção-alvo: o braço vem do acervo com as aquisições do
    gabarito já excluídas, então "material de treino" é literalmente o que ele é.
    Nenhum desses frames entra em manifest.json — eles vivem só na lista do braço.
    """
    frames: list[Frame] = []
    for f in resp.frames:
        if f.image is None:
            continue
        frames.append(
            Frame(
                acq_id=f.acq_id,
                sec=f.sec,
                split="train",
                reviewed=False,
                image=ImageRef(
                    drive_id=f.image.drive_id,
                    ext="jpg",
                    source_link=f.image.source_link,
                    url_original=f.image.url_original,
                    url_cdn=f.image.url_cdn,
                ),
                ground_truth=None,
                curation_counts=None,
            )
        )
    return frames


def budget_ceiling(responses: list[SampleResponse]) -> int:
    """Maior N comum aos três braços.

    O braço com dedup encolhe muito (na query `truck`: 42.831 → 1.257 → 201).
    Comparar braços com N diferente mata o experimento, então o orçamento é o
    menor pool disponível, não o que caberia no maior.
    """
    pools = [
        r.pool_after_dedup or r.pool_after_exclusion or r.pool_matched or 0
        for r in responses
    ]
    return min(pools) if pools else 0
