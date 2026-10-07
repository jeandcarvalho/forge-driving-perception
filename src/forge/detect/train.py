"""Treino do aluno (YOLO11n) sobre os rótulos do professor.

Duas coisas o ultralytics faz por conveniência e que aqui seriam dano:

1. **Ele pareia imagem e rótulo por convenção de caminho** — troca `/images/` por
   `/labels/` e a extensão. O layout do acervo (`images/<acq>/<sec>.jpg` +
   `labels/auto/<tag>/<acq>__<sec>.txt`) não tem essa forma, de propósito: ele é
   organizado por proveniência, não pelo que uma biblioteca espera achar. Por
   isso o treino roda sobre uma **view por hardlink** (`data/view.py`), montada a
   partir de um mapa explícito imagem → rótulo. O pareamento vira decisão do
   Forge, e não coincidência de string.

2. **Imagem sem `.txt` não é erro, é background.** Um pareamento errado não
   estoura: treina em silêncio com o split inteiro como "nenhum objeto". Daí a
   view recusar imagem sem rótulo no mapa, e o `write_data_yaml` recusar um
   `images/` sem `labels/` irmão.

Semente: `seed` + `deterministic=True`. O Experimento 1 reporta média ± desvio
sobre 3 sementes — a variação precisa vir da semente declarada, não do cuDNN
escolhendo algoritmo por tempo de execução.

E aí mora a terceira armadilha, medida
──────────────────────────────────────
**O `seed` do ultralytics não chega no que importa.** Ele semeia `random`,
`numpy` e `torch` no processo principal (`init_seeds`), mas as duas fontes de
aleatoriedade que decidem um treino ficam de fora:

- `data/build.py:365` — `generator.manual_seed(6148914691236517205 + RANK)`,
  constante no código. Esse gerador é o do DataLoader: em treino de 1 GPU
  (`rank == -1`, `sampler is None`) ele governa **a ordem dos dados** e, via
  `worker_init_fn`, **a semente de cada worker** — ou seja, toda a augmentation.
- `data/build.py:210` — o sampler distribuído embaralha com `manual_seed(epoch)`.

Com pesos pré-treinados (init determinística) e `deterministic=True`, sobra zero
aleatoriedade: rodar com `--seed 0`, `1` e `2` produziu aqui três `best.pt`
**byte a byte idênticos**. As três "repetições" seriam a mesma execução copiada,
e o desvio padrão publicado seria 0,000 por construção — não por estabilidade.

Por isso o Forge re-semeia o gerador do DataLoader com a semente pedida. É o
mínimo para que "3 sementes" signifique alguma coisa.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from forge.config import STUDENT_WEIGHTS, TRAIN_BATCH, TRAIN_EPOCHS, TRAIN_IMGSZ


@contextmanager
def seeded_dataloaders(seed: int) -> Iterator[None]:
    """Faz a semente alcançar a ordem dos dados e a augmentation.

    Envelopa `build_dataloader` no módulo que o treinador de detecção importou
    (ele fez `from ultralytics.data import build_dataloader`, então corrigir o
    módulo de origem não teria efeito) e re-semeia o gerador do loader já
    construído. O `DataLoader` só lê `generator` na hora de criar o iterador, de
    modo que re-semear depois vale para todas as épocas.
    """
    from ultralytics.models.yolo.detect import train as detect_train

    original = detect_train.build_dataloader

    def patched(*args, **kwargs):
        loader = original(*args, **kwargs)
        generator = getattr(loader, "generator", None)
        if generator is not None:
            generator.manual_seed(seed)
        return loader

    detect_train.build_dataloader = patched
    try:
        yield
    finally:
        detect_train.build_dataloader = original


@dataclass(slots=True)
class TrainResult:
    weights_best: Path | None
    weights_last: Path | None
    output_dir: Path
    epochs_ran: int
    metrics: dict[str, float] = field(default_factory=dict)
    #: `speed`, `fitness` e afins que o ultralytics devolve fora do bloco métrico.
    extras: dict[str, object] = field(default_factory=dict)


def _clean_metrics(results_dict: dict) -> dict[str, float]:
    """Só os números que viram tabela, com nome curto e estável.

    O `results_dict` do ultralytics muda de chave entre versões
    (`metrics/mAP50-95(B)`…). Normalizar aqui evita que o registro de um run
    fique ilegível quando a dependência subir de versão.
    """
    alias = {
        "metrics/mAP50-95(B)": "mAP50_95",
        "metrics/mAP50(B)": "mAP50",
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "fitness": "fitness",
    }
    out: dict[str, float] = {}
    for key, value in (results_dict or {}).items():
        name = alias.get(key, key)
        try:
            out[name] = float(value)
        except (TypeError, ValueError):
            continue
    return out


def train_detector(
    *,
    data_yaml: Path,
    weights: str | Path = STUDENT_WEIGHTS,
    epochs: int = TRAIN_EPOCHS,
    imgsz: int = TRAIN_IMGSZ,
    batch: int = TRAIN_BATCH,
    seed: int = 0,
    device: str | int | None = None,
    project: Path,
    name: str = "ultralytics",
    workers: int = 8,
    patience: int = 0,
    resume: bool = False,
) -> TrainResult:
    """Treina e devolve onde ficaram os pesos e o que o `val` final mediu.

    `patience=0` desliga o early stopping: com o mesmo número de épocas em todos
    os braços, a curva mAP × N compara treinos comparáveis. Parar cedo em um
    braço só transformaria orçamento de dado em orçamento de época.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    with seeded_dataloaders(seed):
        results = model.train(
            data=str(Path(data_yaml).resolve()),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            seed=seed,
            deterministic=True,
            device=device,
            workers=workers,
            patience=patience,
            project=str(Path(project).resolve()),
            name=name,
            exist_ok=True,
            pretrained=True,
            resume=resume,
            plots=True,
            val=True,
        )

    trainer = getattr(model, "trainer", None)
    save_dir = Path(getattr(trainer, "save_dir", Path(project) / name))
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"

    results_dict = dict(getattr(results, "results_dict", {}) or {})
    return TrainResult(
        weights_best=best if best.is_file() else None,
        weights_last=last if last.is_file() else None,
        output_dir=save_dir,
        epochs_ran=int(getattr(trainer, "epoch", epochs - 1) or 0) + 1,
        metrics=_clean_metrics(results_dict),
        extras={"speed": getattr(results, "speed", None)},
    )
