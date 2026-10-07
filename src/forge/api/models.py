"""Contrato de `GET /public/collections/:id/export/manifest` e `/api/dataset/sample`.

Esta é a fronteira de tipos do Forge: se a API mudar de forma, quebra aqui, alto,
e não trinta arquivos adiante em silêncio.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from forge.core.geometry import BoxN

Split = Literal["test", "train", "discarded"]
GtOrigin = Literal["kept", "corrected", "added"]
Verdict = Literal["kept", "corrected", "rejected", "unreviewed"]


class _Base(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class BoxNModel(_Base):
    x1: float
    y1: float
    x2: float
    y2: float

    def to_box(self) -> BoxN:
        return BoxN(self.x1, self.y1, self.x2, self.y2)


class GtBox(_Base):
    """Uma instância do gabarito humano, já dobrada pela API."""

    cls_name: str | None = Field(alias="class")
    box_n: BoxNModel | None
    origin: GtOrigin
    det_id: str | None = None
    ref_det_id: str | None = None


class Correction(_Base):
    cls_name: str | None = Field(alias="class")
    box_n: BoxNModel | None


class StoredDetection(_Base):
    """Pré-anotação, com o veredito que a curadoria deu a ela.

    NÃO é baseline avaliável: veio de tracking sobre vídeo com corte de confiança
    em ~0.2. Serve para auditar a curadoria (evaluate/parity.py) e nada mais.
    """

    det_id: str | None = None
    cls_name: str | None = Field(alias="class")
    conf: float | None = None
    box_n: BoxNModel | None = None
    track_id: int | None = None
    dist_m: float | None = None
    verdict: Verdict
    correction: Correction | None = None


class CurationCounts(_Base):
    model: int
    kept: int
    corrected: int
    rejected: int
    added: int
    unreviewed: int


class ImageRef(_Base):
    drive_id: str
    ext: str
    source_link: str | None = None
    url_original: str
    url_cdn: str
    #: Derivados de px ÷ normalizado no servidor; podem vir null, e podem estar
    #: errados. A dimensão que vale é a medida no arquivo baixado.
    width: int | None = None
    height: int | None = None


class Gps(_Base):
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None


class Telemetry(_Base):
    gps: Gps | None = None
    speed: float | None = None
    steering: float | None = None
    #: `BrakeInfoStatus` do CAN é CATEGÓRICO, não numérico: chega como
    #: "Brake pedal not pressed" / "Brake pedal pressed" / "Brake pedal confirmed
    #: pressed". Aceita número também porque outras aquisições podem gravar cru.
    brake: float | str | None = None
    vehicle: str | None = None
    period: str | None = None
    condition: str | None = None
    semseg: dict[str, float | None] | None = None


class Frame(_Base):
    acq_id: str
    sec: int
    split: Split
    reviewed: bool

    #: null quando o frame não tem link de imagem utilizável no banco.
    image: ImageRef | None

    #: null (NÃO []) quando o frame não foi revisado. "Sem anotação" e "anotado
    #: como vazio" são estados diferentes e não podem colapsar. Sem default de
    #: propósito: se a API parar de mandar o campo, queremos erro, não None.
    ground_truth: list[GtBox] | None

    curation_counts: CurationCounts | None
    stored_detections: list[StoredDetection] = Field(default_factory=list)
    telemetry: Telemetry | None = None

    @model_validator(mode="after")
    def _gt_matches_reviewed(self) -> Frame:
        # Invariante do export-manifest.ts: `ground_truth: reviewed ? resolved : null`.
        # Se isso deixar de valer, todo o resto do Forge está construído em cima
        # de uma premissa falsa — melhor descobrir na desserialização.
        if self.reviewed and self.ground_truth is None:
            raise ValueError(
                f"frame {self.acq_id}:{self.sec} veio reviewed=true com ground_truth=null"
            )
        if not self.reviewed and self.ground_truth is not None:
            raise ValueError(
                f"frame {self.acq_id}:{self.sec} veio reviewed=false com ground_truth não-null"
            )
        return self

    @property
    def key(self) -> str:
        return f"{self.acq_id}:{self.sec}"

    @property
    def is_labeled(self) -> bool:
        """Tem gabarito humano utilizável.

        Só isto autoriza o frame a entrar numa lista de imagens de split. Um
        frame não revisado NÃO pode aparecer em splits/*.txt: no formato do
        ultralytics, imagem sem .txt correspondente é contada como background —
        "conferida, sem objeto". Ausência de label não abstém, ela afirma.
        """
        return self.reviewed and self.ground_truth is not None

    @property
    def is_empty_labeled(self) -> bool:
        """Revisado e genuinamente sem objeto. Este SIM vira .txt vazio."""
        return self.reviewed and self.ground_truth == []


class CurationMeta(_Base):
    locked: bool | None = None
    frame_count: int | None = None
    min_distance_m: float | None = None
    generated_at: str | None = None
    generated_by: str | None = None


class CollectionMeta(_Base):
    id: str
    name: str | None = None
    description: str | None = None
    gps_spacing_m: float | None = None
    frames_total: int | None = None
    curation: CurationMeta | None = None


class SplitTotals(_Base):
    test: int
    train: int
    discarded: int


class ManifestPage(_Base):
    collection: CollectionMeta
    split: str
    totals: SplitTotals
    page: int
    per_page: int
    total: int
    has_more: bool
    stored_detections_note: str | None = None
    frames: list[Frame]


# ── /api/dataset/sample ──────────────────────────────────────────────────────


class SampleImage(_Base):
    drive_id: str
    source_link: str | None = None
    url_original: str
    url_cdn: str


class SampleFrame(_Base):
    acq_id: str
    sec: int
    gps: Gps | None = None
    image: SampleImage | None = None


class SampleResponse(_Base):
    """Registro reprodutível de qual braço do Experimento 1 gerou qual treino.

    Os campos de pool não são cosméticos: `pool_after_dedup` é o teto de N do
    braço com dedup, e é ele que decide o orçamento comum aos três braços.
    """

    mode: Literal["random", "query"]
    seed: int
    n_requested: int
    n_returned: int
    dedup_m: float | None = None
    exclude_collection: str | None = None
    exclude_mode: str | None = None
    excluded_acqs: int | None = None
    excluded_frames: int | None = None
    pool_matched: int | None = None
    pool_after_exclusion: int | None = None
    pool_after_dedup: int | None = None
    #: Diversidade de aquisição. O gabarito cobre 127 gravações em 313 frames
    #: (2,5 por gravação) porque a curadoria o montou espaçando a 1.593 m; uma
    #: amostra de treino com 23 frames da mesma gravação não se parece com ele.
    #: O dedup por GPS não move este número — ele espaça DENTRO da aquisição.
    spread: str | None = None
    acqs_in_pool: int | None = None
    acqs_sampled: int | None = None
    frames_per_acq_max: int | None = None
    frames_per_acq_mean: float | None = None
    frames: list[SampleFrame]
