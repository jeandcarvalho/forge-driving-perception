"""Rotulagem automática com o detector-professor (YOLO11x).

O professor é **ferramenta de método, não resultado**: não vai em tabela nenhuma,
não é baseline, não é medido. Ele existe para converter frames sem gabarito em
material de treino — o insumo dos dois experimentos.

Corte de confiança: o inverso do `forge infer`
──────────────────────────────────────────────
Na inferência, `conf=0.001` é obrigatório: sem a cauda de baixa confiança não há
curva precision-recall, logo não há mAP. Aqui a mesma cauda é veneno — cada
detecção duvidosa vira um rótulo afirmativo, e o aluno aprende o ruído como se
fosse verdade. Daí 0.25, e daí a ablação 0.25 × 0.5 do PROMPT §1: rótulo farto e
sujo contra rótulo escasso e limpo.

Frame em que o professor não vê nada
────────────────────────────────────
Em `writers/yolo.py` a ausência de `.txt` AFIRMA "vazio", e é por isso que frame
não revisado fica fora da lista de imagens. Aqui o regime é outro: a função de
rotulagem **é** o professor, declarada no método, e "não vi nada acima do corte"
é uma saída dele como qualquer outra. O frame recebe `.txt` vazio e permanece na
lista — o aluno precisa de negativos.

A segunda razão é a que decide, e é do desenho do Experimento 1: os três braços
têm o **mesmo orçamento de N frames**. Descartar frame sem detecção encolheria o
braço aleatório muito mais que os outros — 61% do acervo não tem objeto nenhum —
e a comparação passaria a medir tamanho de conjunto em vez de qualidade da
seleção. `--drop-empty` existe para a ablação, nunca para o número principal.

Espaço de classes: 80 do COCO, e o índice sai do NOME (`model.names` →
`class_id`), nunca da posição. Um professor com cabeça diferente falha alto em
vez de produzir rótulos com a classe trocada em silêncio.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from forge.config import LABEL_CONF, LABEL_IMGSZ, LABEL_NMS_IOU
from forge.core.classes import class_id, class_name, is_valid_class
from forge.data.layout import frame_stem
from forge.detect.source import source_list

#: (nome da classe, caixa normalizada centro+tamanho) — o formato de rótulo do
#: ultralytics já é o que `boxes.xywhn` devolve, então não há conversão de
#: coordenada no meio do caminho para errar sinal.
Detection = tuple[str, tuple[float, float, float, float]]


@dataclass(slots=True)
class LabelResult:
    labels_dir: Path
    images: int = 0
    labels_written: int = 0
    #: Frames em que o professor não viu nada acima do corte. Com `keep_empty`
    #: eles têm `.txt` vazio e estão na lista; sem, ficaram de fora dos dois.
    empty_labels: int = 0
    instances: int = 0
    classes_seen: dict[str, int] = field(default_factory=dict)
    dropped_boxes: list[str] = field(default_factory=list)
    listed_images: list[Path] = field(default_factory=list)

    @property
    def instances_per_image(self) -> float:
        return self.instances / self.images if self.images else 0.0


def default_tag(weights: str | Path, conf: float) -> str:
    """`yolo11x.pt`, 0.25 → `yolo11x_c025`.

    O tag nomeia o diretório de rótulos e a lista de treino. Ele carrega o
    professor e o corte porque são exatamente os dois eixos da ablação: sem isso
    no nome, as duas rodadas se sobrescrevem e nada no disco denuncia.
    """
    stem = Path(str(weights)).stem or "teacher"
    return f"{stem}_c{round(conf * 100):03d}"


def format_label_lines(dets: Iterable[Detection]) -> tuple[list[str], list[str]]:
    """Detecções → linhas de rótulo YOLO. Devolve `(linhas, descartes)`.

    Descarta em vez de abortar: um professor pode emitir caixa degenerada, e
    perder uma caixa é melhor que perder a rodada inteira. O descarte volta
    nomeado para virar linha de relatório.
    """
    lines: list[str] = []
    dropped: list[str] = []

    for name, (cx, cy, w, h) in dets:
        if not is_valid_class(name):
            dropped.append(f"classe fora do espaço COCO de 80: {name!r}")
            continue
        if not (w > 0 and h > 0):
            dropped.append(f"{name}: caixa degenerada w={w:.6f} h={h:.6f}")
            continue
        # Caixa que estoura a borda existe (o detector extrapola objeto cortado);
        # o ultralytics rejeita label fora de 0..1, então grampeia-se aqui.
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        cw, ch = x2 - x1, y2 - y1
        if not (cw > 0 and ch > 0):
            dropped.append(f"{name}: caixa inteiramente fora da imagem")
            continue
        lines.append(
            f"{class_id(name)} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {cw:.6f} {ch:.6f}"
        )

    return lines, dropped


def write_label_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def generate_labels(
    weights: str | Path,
    images: list[Path],
    out_dir: Path,
    *,
    conf: float = LABEL_CONF,
    iou: float = LABEL_NMS_IOU,
    imgsz: int = LABEL_IMGSZ,
    device: str | int | None = None,
    batch: int = 16,
    keep_empty: bool = True,
    progress: bool = True,
) -> LabelResult:
    """Roda o professor sobre `images` e escreve um `.txt` YOLO por frame."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names: dict[int, str] = model.names

    # Mesma guarda do `infer`: falha antes de gastar GPU. Rótulo com índice de
    # outra cabeça é indetectável depois — vira treino com classe trocada.
    unknown = sorted({n for n in names.values() if not is_valid_class(n)})
    if unknown:
        raise ValueError(
            f"o professor {weights} tem classes fora do espaço COCO de 80: {unknown[:10]}. "
            "O Forge mantém indexação COCO de ponta a ponta (ver core/classes.py)."
        )

    out = LabelResult(labels_dir=Path(out_dir))
    out.labels_dir.mkdir(parents=True, exist_ok=True)

    # `.txt` e não lista de caminhos: com uma lista o ultralytics ignora `batch`
    # e monta um batch único do conjunto inteiro. Ver detect/source.py.
    with source_list(images) as source:
        stream = model.predict(
            source=source,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            stream=True,
            verbose=False,
            batch=batch,
        )

        for res in tqdm(
            stream, total=len(images), disable=not progress, unit="img", desc="rotulando"
        ):
            img_path = Path(res.path)
            out.images += 1

            dets: list[Detection] = []
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xywhn = boxes.xywhn.cpu().tolist()
                clss = boxes.cls.cpu().tolist()
                for (cx, cy, w, h), cls_idx in zip(xywhn, clss, strict=True):
                    dets.append((names[int(cls_idx)], (cx, cy, w, h)))

            lines, dropped = format_label_lines(dets)
            out.dropped_boxes.extend(f"{frame_stem(img_path)}: {d}" for d in dropped)

            if not lines:
                out.empty_labels += 1
                if not keep_empty:
                    # Sai do rótulo E da lista, junto. Deixar o frame na lista sem
                    # `.txt` seria afirmar "vazio" pela porta dos fundos.
                    continue

            write_label_file(out.labels_dir / f"{frame_stem(img_path)}.txt", lines)
            out.labels_written += 1
            out.instances += len(lines)
            out.listed_images.append(img_path)
            # Conta a partir das LINHAS, não de `dets`: o que virou rótulo é o que
            # sobreviveu ao descarte, e é esse número que descreve o conjunto.
            for line in lines:
                name = class_name(int(line.split(" ", 1)[0]))
                out.classes_seen[name] = out.classes_seen.get(name, 0) + 1

    out.classes_seen = dict(sorted(out.classes_seen.items(), key=lambda kv: -kv[1]))
    return out
