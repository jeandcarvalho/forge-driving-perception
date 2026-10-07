"""Configuração e constantes de método.

Tudo que muda um número do paper mora aqui, nomeado e comentado. Nada de
constante mágica enterrada no matcher ou no writer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# ── Constantes de método ─────────────────────────────────────────────────────

#: Inferência local. `conf` quase zero porque sem detecção de baixa confiança não
#: existe curva precision-recall, logo não existe mAP. É por isso que o `yolo[]`
#: guardado no banco (cortado em ~0.2) não serve para avaliar.
INFER_CONF = 0.001
INFER_NMS_IOU = 0.7
INFER_IMGSZ = 640
INFER_IMGSZ_ALT = 1280  # variação registrada no paper

#: Rotulagem automática (`forge label`). O professor é ferramenta de método, não
#: resultado: não vai em tabela nenhuma. O corte de confiança é o inverso do da
#: inferência — aqui a cauda é ruído que vira rótulo errado, e o aluno aprende o
#: ruído. 0.25 vs 0.5 é a ablação do PROMPT §1: rótulo farto e sujo contra rótulo
#: escasso e limpo.
TEACHER_WEIGHTS = "yolo11x.pt"
LABEL_CONF = 0.25
LABEL_CONF_ABLATION = (0.25, 0.5)
LABEL_NMS_IOU = 0.7
LABEL_IMGSZ = 640

#: Treino do aluno (`forge train`). Semente fixa e `deterministic` porque o
#: Experimento 1 reporta média ± desvio sobre 3 sementes — a variação tem que vir
#: da semente declarada, não da não-determinismo do cuDNN.
STUDENT_WEIGHTS = "yolo11n.pt"
TRAIN_EPOCHS = 50
TRAIN_IMGSZ = 640
TRAIN_BATCH = 16
TRAIN_SEEDS = (0, 1, 2)

#: Workers do dataloader. Baixo de propósito: no Windows cada worker é um
#: PROCESSO novo que reimporta o torch (~1 GB de RSS cada), e não uma thread.
#: Com 8 workers a máquina de 16 GB fica sem RAM no meio do treino — e o erro
#: sai como `cv2.OutOfMemoryError` ao ler um JPEG de 6 MB, que parece problema
#: de imagem e não de configuração.
TRAIN_WORKERS = 4

#: IoU do `curation/stats` da API. Aqui ele compara a caixa ORIGINAL com a
#: CORREÇÃO dela — "o curador empurrou ou refez a caixa?" — e não predição com
#: gabarito. Ver evaluate/parity.py.
PARITY_IOU = 0.7

#: Toda imagem do acervo é JPEG 1920x1080. O manifest deriva width/height de
#: px ÷ normalizado e pode devolver null; a dimensão de verdade sai do arquivo.
EXPECTED_IMAGE_SIZE = (1920, 1080)

#: Splits do manifest. `discarded` são os frames restantes das aquisições que
#: doaram frame ao teste — existem para auditoria e NADA mais. Nunca treine
#: neles: a aquisição inteira está contaminada pela regra de antivazamento.
SPLITS = ("test", "train", "discarded")
TRAINABLE_SPLITS = ("train",)


#: Classes descartadas do gabarito por serem erro de anotação, não classe rara.
#:
#: `airplane` — instância única, e verificada: o objeto está anotado como avião e
#: é um caminhão. A predição `truck` casa com ele em IoU 0.912 e score 0.928,
#: enquanto as caixas de classe `airplane` ficam em IoU 0.24 e 0.48. Numa base de
#: direção veicular em rodovia, avião não é classe rara — é rótulo trocado.
#: `train` — 3 instâncias, mesma natureza.
#:
#: O efeito no número do paper é grande porque o mAP é média SIMPLES entre
#: classes: com 8 classes, uma única instância de avião valia 12,5% da nota.
#: Ela sozinha explicava a divergência entre o `val` do ultralytics (0.296) e o
#: `forge eval` (0.190) sobre os mesmos pesos.
#:
#: Classes fora do escopo do GABARITO. As predições delas somem por consequência,
#: porque `EvalClassScope.GT_PRESENT` descarta o que o gabarito não contém.
#: O `forge audit` NÃO aplica esta lista: ele confere paridade com a API, e a API
#: continua servindo as 735 instâncias originais.
GT_EXCLUDED_CLASSES: frozenset[str] = frozenset({"airplane", "train"})


class EvalClassScope(str, Enum):
    """O que fazer com detecção de classe que o gabarito não contém.

    O YOLO11n cru emite `traffic light`, `stop sign`, `bench`, `boat`… O gabarito
    só tem 8 classes, e não porque as outras não existem na cena — é porque a
    curadoria não anotou nada além do que a pré-anotação propôs nessas 8.

    GT_PRESENT (padrão): a avaliação se restringe às classes presentes no
        gabarito; detecção fora desse conjunto é descartada antes do matching,
        não vira falso positivo. É o honesto — punir o modelo por achar um
        semáforo que ninguém anotou mede a lacuna do gabarito, não o modelo.
        Precisa ir escrito no método do paper.

    ALL_COCO: nada é descartado; classe fora do gabarito vira FP de fundo.
        Deixa a métrica pessimista e pouco interpretável. Existe para produzir
        o número da nota de rodapé, se algum revisor pedir.
    """

    GT_PRESENT = "gt_present"
    ALL_COCO = "all_coco"


EVAL_CLASS_SCOPE_DEFAULT = EvalClassScope.GT_PRESENT

#: Índice de categoria nos JSON COCO que o Forge escreve == índice YOLO de 80
#: classes (0=person … 79=toothbrush). NÃO é o category_id 1..90 do COCO
#: oficial. Como o Forge escreve tanto o gabarito quanto as predições, o único
#: requisito é consistência — e um espaço só evita a conversão que erra sinal.
COCO_CATEGORY_ID_IS_YOLO_INDEX = True

# ── Experimento A1: VLM como curador ─────────────────────────────────────────

#: O modelo é INSTRUCT, não a variante thinking, e isso não é preferência.
#: Medido em 19/08/2026 na RTX 5060 Ti, mesma imagem e mesmo schema:
#:   qwen3-vl:8b          (thinking) 10,9 s/caixa @1280x720 · 30 s @1920x1080
#:   qwen3-vl:8b-instruct            3,6 s/caixa @1280x720 · 5,3 s @1920x1080
#: A variante thinking gasta ~1.556 caracteres de raciocínio por caixa que são
#: descartados — o veredito sai do mesmo enum nos dois casos. 3x mais barato
#: pelo mesmo resultado observável.
VLM_MODEL = "qwen3-vl:8b-instruct"
VLM_URL = "http://localhost:11434"

#: Temperatura zero: o veredito é medição, não geração. Duas rodadas do mesmo
#: frame têm que dar o mesmo número, senão a matriz de concordância mede ruído
#: de amostragem junto com discordância real.
VLM_TEMPERATURE = 0.0

#: Imagem vai em resolução NATIVA (1920x1080). Reduzir para 1280x720 economiza
#: 1,7 s por chamada e custa detalhe justamente no objeto pequeno e distante,
#: que é onde a curadoria humana mais interveio. 45 min de máquina é mais barato
#: que um viés sistemático contra classe rara.
VLM_IMAGE_MAX_SIDE = 1920

#: Fase 1 manda DUAS imagens por caixa: o frame inteiro com a caixa desenhada
#: (contexto de cena) e um recorte ampliado ao redor dela (detalhe). É o que o
#: curador humano fez na interface — olhar a cena, depois aproximar.
VLM_CROP_PAD = 0.6  # padding do recorte, em fração do lado maior da caixa
VLM_CROP_MIN_PX = 224  # recorte nunca menor que isto, senão vira mosaico

#: Fase 2 (recall): teto duro de iterações. "Até não achar mais nada" é laço não
#: terminante — o modelo sempre acha mais uma coisa se você continuar pedindo.
VLM_RECALL_MAX_ITERS = 3

#: Teto de itens por resposta da Fase 2. Sem ele o modelo inunda: medido em
#: 20/08/2026, um frame gerou 144 itens e 5.968 tokens até truncar o JSON no
#: meio. 25 é folgado — o frame mais povoado do gabarito tem 14 instâncias.
VLM_RECALL_MAX_ITEMS = 25

#: IoU do dedup entre propostas da Fase 2 (proposta nova x já confirmada).
VLM_RECALL_DEDUP_IOU = 0.5

#: IoU FROUXO para casar proposta da Fase 2 com caixa humana desenhada na mão.
#: A pergunta da Fase 2 é "o VLM percebeu que faltava um objeto aqui?", não
#: "o VLM desenhou uma caixa boa?" — ele é fraco em geometria e medir geometria
#: fina aqui responderia a pergunta errada. Centro-dentro-da-caixa é o critério
#: alternativo, aplicado em OR com este.
VLM_RECALL_PRESENCE_IOU = 0.3

#: As cinco categorias do Hub. Ordem é a da matriz de confusão, do menos ao mais
#: severo — não reordene sem refazer as tabelas.
VLM_VERDICTS = (
    "kept",
    "corrected-minor",
    "corrected-redrawn",
    "class-changed",
    "rejected",
)

# ── Ambiente ─────────────────────────────────────────────────────────────────

DEFAULT_API_URL = "http://localhost:8080"
DEFAULT_DATA_ROOT = Path("D:/carcara-data")
TARGET_COLLECTION_ID = "6a149cdc170cb35d6f7bce11"


@dataclass(frozen=True, slots=True)
class Settings:
    api_url: str
    data_root: Path
    collection_id: str

    @classmethod
    def from_env(cls) -> Settings:
        _load_dotenv()
        return cls(
            api_url=os.environ.get("CARCARA_API_URL", DEFAULT_API_URL).rstrip("/"),
            data_root=Path(os.environ.get("FORGE_DATA_ROOT", str(DEFAULT_DATA_ROOT))),
            collection_id=os.environ.get("FORGE_COLLECTION_ID", TARGET_COLLECTION_ID),
        )


def _load_dotenv(path: Path | None = None) -> None:
    """Lê um .env raso, sem dependência extra. Ambiente real tem precedência."""
    env_path = path or Path.cwd() / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())
