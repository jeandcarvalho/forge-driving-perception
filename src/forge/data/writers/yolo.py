"""Writer de rótulos YOLO/ultralytics.

A REGRA QUE GOVERNA ESTE ARQUIVO
────────────────────────────────
No formato do ultralytics, uma imagem sem arquivo de label correspondente NÃO é
uma imagem ignorada: é uma imagem contada como *background* — "conferida, nenhum
objeto". O próprio scan do dataset a soma em `backgrounds`.

Logo, ausência de .txt não abstém. Ela AFIRMA "vazio".

Consequência, e é o inverso do que a intuição sugere:

  frame NÃO revisado  → não escreve .txt E NÃO ENTRA NA LISTA DE IMAGENS.
                        A exclusão tem que ser da lista; omitir o label seria
                        justamente afirmar o que não sabemos.

  frame revisado e vazio → escreve .txt VAZIO e ENTRA na lista.
                        Aqui a afirmação "sem objeto" é verdadeira e foi
                        conferida por gente. São 8 frames no split de teste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from forge.api.models import Frame
from forge.config import GT_EXCLUDED_CLASSES
from forge.core.classes import class_id, is_valid_class, normalize_class
from forge.core.geometry import is_valid_box_n
from forge.data.layout import DatasetLayout


@dataclass(slots=True)
class YoloWriteResult:
    split: str
    labels_written: int = 0
    empty_labels: int = 0
    instances: int = 0
    #: Frames deixados de fora da lista de imagens por não terem gabarito.
    excluded_unreviewed: int = 0
    #: Frames sem link de imagem utilizável — também não podem ser listados.
    excluded_no_image: int = 0
    #: Instâncias tiradas por GT_EXCLUDED_CLASSES (rótulo trocado, não classe rara).
    excluded_class: int = 0
    dropped_boxes: list[str] = field(default_factory=list)
    #: Imagens que podem entrar numa lista TREINÁVEL — só as que têm rótulo.
    listed_images: list[Path] = field(default_factory=list)
    #: Todo JPEG do split que existe em disco, com rótulo ou sem. É o que vai
    #: para o `_pool.txt`: um inventário do que dá para rotular depois. Sem isto
    #: o `train_pool.txt` sairia vazio (nenhum frame de treino é revisado) e o
    #: `forge label` não teria de onde ler a lista de trabalho.
    inventory_images: list[Path] = field(default_factory=list)


def write_yolo_labels(
    layout: DatasetLayout,
    frames: list[Frame],
    split: str,
    *,
    only_downloaded: bool = True,
) -> YoloWriteResult:
    """Escreve labels do gabarito humano e devolve a lista de imagens elegíveis.

    `only_downloaded=True` também exige que o JPEG exista em disco: listar uma
    imagem que não foi baixada quebra o treino no primeiro epoch.
    """
    out = YoloWriteResult(split=split)
    target_dir = layout.labels_dir / split
    target_dir.mkdir(parents=True, exist_ok=True)

    for f in frames:
        if f.split != split:
            continue

        # Sem JPEG em disco o frame não serve para nada — nem para rotular
        # depois. Some antes de qualquer outra consideração.
        if f.image is None:
            out.excluded_no_image += 1
            continue

        img_path = layout.image_path(f.acq_id, f.sec)
        if only_downloaded and not img_path.is_file():
            out.excluded_no_image += 1
            continue

        # Tem imagem: entra no inventário, com rótulo ou sem.
        out.inventory_images.append(img_path)

        # ── A exclusão é DA LISTA TREINÁVEL, não do arquivo de label ─────────
        if not f.is_labeled:
            out.excluded_unreviewed += 1
            continue

        lines: list[str] = []
        for g in f.ground_truth or []:
            if not is_valid_class(g.cls_name):
                out.dropped_boxes.append(f"{f.key}: classe inválida {g.cls_name!r}")
                continue
            # Erro de anotação, não classe rara — ver GT_EXCLUDED_CLASSES.
            if normalize_class(g.cls_name or "") in GT_EXCLUDED_CLASSES:
                out.excluded_class += 1
                continue
            box = g.box_n.to_box() if g.box_n is not None else None
            if not is_valid_box_n(box):
                out.dropped_boxes.append(f"{f.key}: caixa inválida em {g.cls_name!r}")
                continue
            cx, cy, w, h = box.to_yolo_cxcywh()
            # Índice COCO de 80 classes, sempre. Ver core/classes.py.
            lines.append(f"{class_id(g.cls_name)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        label_path = layout.label_path(split, f.acq_id, f.sec)
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

        out.labels_written += 1
        out.instances += len(lines)
        if not lines:
            out.empty_labels += 1
        out.listed_images.append(img_path)

    return out


def write_split_list(
    layout: DatasetLayout,
    images: list[Path],
    split: str,
    *,
    trainable: bool,
) -> Path:
    """Grava a lista de imagens de um split.

    `trainable=True` é uma afirmação: toda imagem desta lista tem rótulo em disco.
    Um data.yaml só pode apontar para listas assim.
    """
    if trainable and split == "discarded":
        raise ValueError(
            "split `discarded` nunca é treinável: são os frames restantes das "
            "aquisições que doaram frame ao teste. Existem para auditoria — "
            "treinar neles contamina a avaliação pela regra de antivazamento."
        )

    path = layout.split_list_path(split, trainable=trainable)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(p.resolve()) for p in images) + ("\n" if images else ""),
        encoding="utf-8",
    )
    return path


def write_auto_split_list(
    layout: DatasetLayout, images: list[Path], tag: str, *, source_split: str | None = None
) -> Path:
    """Lista treinável de uma rodada de `forge label`.

    Só chame depois de escrever os rótulos: cada imagem daqui precisa ter `.txt`
    em `labels/auto/<tag>/`. `source_split` existe só para a mesma guarda de
    sempre — `discarded` não vira material de treino nem com rótulo automático,
    porque a contaminação é da aquisição, não da falta de rótulo.
    """
    if source_split == "discarded":
        raise ValueError(
            "split `discarded` nunca é treinável, nem com rótulo do professor: "
            "são frames de aquisições que doaram frame ao teste. Rotulá-los não "
            "desfaz o vazamento."
        )

    path = layout.auto_split_list_path(tag)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(str(p.resolve()) for p in images) + ("\n" if images else ""),
        encoding="utf-8",
    )
    return path


def _assert_has_labels(role: str, p: Path) -> None:
    """Um alvo de data.yaml precisa de rótulo pareável, e a checagem depende da forma.

    Lista `.txt` → basta o nome não ser `_pool`, porque o caminho do rótulo não é
    derivável da imagem no layout do acervo (é o `forge train` que resolve o
    pareamento montando a view).

    Diretório → é uma view do ultralytics, e aí o pareamento é por convenção de
    caminho (`/images/` → `/labels/`). Um `images/` sem `labels/` irmão não dá
    erro no ultralytics: ele treina com o split inteiro contado como background.
    """
    from forge.data.layout import POOL_LIST_SUFFIX

    if p.name.endswith(POOL_LIST_SUFFIX):
        raise ValueError(
            f"{role}: {p.name} é uma lista de POOL (imagens sem rótulo). "
            "Gere os rótulos com `forge label` antes de treinar."
        )
    if not p.is_dir():
        return

    labels = p.parent / "labels"
    if not labels.is_dir() or not any(labels.glob("*.txt")):
        raise ValueError(
            f"{role}: {p} não tem `labels/` irmão com rótulos. No ultralytics isso "
            "não é erro — é um split inteiro contado como background. Monte a view "
            "com data/view.py antes de escrever o data.yaml."
        )


def write_data_yaml(
    layout: DatasetLayout, *, train_list: Path, val_list: Path, path: Path | None = None
) -> Path:
    """data.yaml do ultralytics, com o espaço de 80 classes COCO.

    `train_list`/`val_list` são listas `.txt` de imagens ou diretórios `images/`
    de uma view. Recusa listas `_pool` — elas são inventário de imagens sem
    rótulo, e apontar um treino para lá é exatamente o dano que o writer inteiro
    existe para evitar.

    `path` grava fora da raiz (o `forge train` põe um por view, já que cada tag
    de rotulagem tem o seu; um data.yaml único na raiz seria sobrescrito a cada
    treino e nenhum run saberia com qual conjunto rodou).
    """
    from forge.core.classes import COCO_CLASSES

    for role, p in (("train", train_list), ("val", val_list)):
        _assert_has_labels(role, p)

    names = "\n".join(f"  {i}: {n}" for i, n in enumerate(COCO_CLASSES))
    path = path or layout.root / "data.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Gerado pelo carcara-forge. Espaço de 80 classes COCO — NÃO remapeie\n"
        "# para o subconjunto do gabarito: o Experimento 2 compara com os pesos\n"
        "# COCO crus, e trocar a cabeça do modelo mata a comparação.\n"
        f"path: {layout.root.resolve()}\n"
        f"train: {train_list.resolve()}\n"
        f"val: {val_list.resolve()}\n"
        "names:\n"
        f"{names}\n",
        encoding="utf-8",
    )
    return path
