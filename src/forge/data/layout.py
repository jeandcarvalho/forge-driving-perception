"""Caminhos canônicos do dataset em disco.

Layout de PROMPT.md §6, com dois acréscimos: `raw/` (páginas cruas do manifest,
proveniência) e `provenance.json` (quem/quando/de onde).

    <root>/<collection-slug>/
      raw/manifest_<split>_p001.json     páginas cruas, como vieram
      manifest.json                      normalizado pelo Forge
      provenance.json                    api, timestamp, versões, contagens
      images/<acq_id>/<sec:06d>.jpg
      annotations/instances_test.json    COCO, gabarito humano
      labels/yolo/<split>/<acq_id>__<sec>.txt
      labels/auto/<tag>/<acq_id>__<sec>.txt    rótulos do professor (`forge label`)
      splits/test.txt                    imagens COM rótulo (treináveis)
      splits/train_pool.txt              imagens SEM rótulo ainda (ver abaixo)
      splits/discarded_pool.txt          auditoria — nunca treine
      splits/auto_<tag>.txt              lista treinável que o `forge label` produz
      views/<tag>/{train,val}/{images,labels}/   view por hardlink p/ o ultralytics
      telemetry.parquet
      checksums.sha256
      verify_failures.json

Por que `train_pool.txt` e não `train.txt`
──────────────────────────────────────────
Todo frame do split `train` vem `reviewed=false` / `ground_truth=null`: o gabarito
humano só existe no split `test`. Rótulo de treino nasce depois, no `forge label`,
a partir do detector-professor.

Um `splits/train.txt` escrito no `pull` seria uma lista de imagens sem nenhum .txt
correspondente — e no ultralytics imagem sem label não é imagem ignorada, é imagem
contada como *background*. Treinar nisso ensinaria "não há objeto nenhum" em 1.406
frames que ninguém olhou.

Então o `pull` só escreve `_pool.txt`: um inventário, com nome que não engana e que
nenhum data.yaml aponta. A lista treinável é escrita pelo `forge label`, e só contém
frames que de fato ganharam arquivo de rótulo.

Por que a lista do `label` é `auto_<tag>.txt` e não `train.txt`
───────────────────────────────────────────────────────────────
A ablação do PROMPT §1 pede o mesmo pool rotulado com dois cortes de confiança
(0.25 e 0.5). Num único `train.txt` a segunda rodada sobrescreveria a primeira em
silêncio, e o treino seguinte usaria rótulos que ninguém pediu — com o mesmo nome
de arquivo, sem nada no disco denunciando a troca. O `tag` (professor + corte)
entra no nome do diretório de rótulos E no da lista, então as duas rodadas
coexistem e o `data.yaml` de cada treino aponta para uma delas explicitamente.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

#: Listas de imagens que um data.yaml pode apontar. `_pool` nunca entra aqui.
TRAINABLE_LIST_SUFFIX = ".txt"
POOL_LIST_SUFFIX = "_pool.txt"


def frame_stem(image_path: Path) -> str:
    """`images/<acq_id>/<sec:06d>.jpg` → `<acq_id>__<sec>`.

    Chave plana de um frame: nome de arquivo de rótulo e de item da view. O `sec`
    perde o zero à esquerda de propósito — é assim que `label_path` já escreve.
    """
    return f"{image_path.parent.name}__{int(image_path.stem)}"


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "collection"


@dataclass(frozen=True, slots=True)
class DatasetLayout:
    root: Path

    @classmethod
    def for_collection(cls, data_root: Path, collection_name: str | None, collection_id: str) -> DatasetLayout:
        slug = slugify(collection_name) if collection_name else collection_id
        return cls(root=Path(data_root) / slug)

    # ── diretórios ───────────────────────────────────────────────────────────
    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def images_dir(self) -> Path:
        return self.root / "images"

    @property
    def annotations_dir(self) -> Path:
        return self.root / "annotations"

    @property
    def labels_dir(self) -> Path:
        return self.root / "labels" / "yolo"

    @property
    def splits_dir(self) -> Path:
        return self.root / "splits"

    @property
    def auto_labels_root(self) -> Path:
        """Rótulos do professor. Separado de `labels/yolo/` de propósito: aquilo
        é gabarito humano, isto é saída de modelo. Misturar os dois num diretório
        só faria a distinção depender de quem lembra qual split é qual."""
        return self.root / "labels" / "auto"

    @property
    def views_dir(self) -> Path:
        return self.root / "views"

    # ── arquivos ─────────────────────────────────────────────────────────────
    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    @property
    def provenance_path(self) -> Path:
        return self.root / "provenance.json"

    @property
    def telemetry_path(self) -> Path:
        return self.root / "telemetry.parquet"

    @property
    def checksums_path(self) -> Path:
        return self.root / "checksums.sha256"

    @property
    def verify_failures_path(self) -> Path:
        return self.root / "verify_failures.json"

    def raw_page_path(self, split: str, page: int) -> Path:
        return self.raw_dir / f"manifest_{split}_p{page:03d}.json"

    def image_path(self, acq_id: str, sec: int) -> Path:
        return self.images_dir / acq_id / f"{sec:06d}.jpg"

    def label_path(self, split: str, acq_id: str, sec: int) -> Path:
        return self.labels_dir / split / f"{acq_id}__{sec}.txt"

    def coco_path(self, split: str) -> Path:
        return self.annotations_dir / f"instances_{split}.json"

    def split_list_path(self, split: str, *, trainable: bool) -> Path:
        """Lista de imagens de um split.

        `trainable=False` produz `<split>_pool.txt` — inventário de imagens que
        ainda não têm rótulo. Nunca aponte um data.yaml para um `_pool`.
        """
        suffix = TRAINABLE_LIST_SUFFIX if trainable else POOL_LIST_SUFFIX
        return self.splits_dir / f"{split}{suffix}"

    # ── rotulagem automática (`forge label`) ─────────────────────────────────
    def auto_labels_dir(self, tag: str) -> Path:
        return self.auto_labels_root / tag

    def auto_label_path(self, tag: str, acq_id: str, sec: int) -> Path:
        return self.auto_labels_dir(tag) / f"{acq_id}__{sec}.txt"

    def auto_split_list_path(self, tag: str) -> Path:
        """Lista treinável de uma rodada de `forge label`. Sempre treinável: ela
        só é escrita depois que os rótulos existem em disco. Um nome por tag —
        é o que permite as duas pernas da ablação coexistirem."""
        return self.splits_dir / f"auto_{tag}.txt"

    def view_dir(self, tag: str) -> Path:
        return self.views_dir / tag

    def ensure_dirs(self) -> None:
        for d in (self.root, self.raw_dir, self.images_dir, self.annotations_dir,
                  self.labels_dir, self.splits_dir):
            d.mkdir(parents=True, exist_ok=True)
