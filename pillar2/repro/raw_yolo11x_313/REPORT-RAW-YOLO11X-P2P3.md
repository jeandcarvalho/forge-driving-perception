# P2/P3 com YOLO11x puro — relatório de reconstrução

Gerado em 10/09/2026. Documento de método/dados, **não é texto de paper e o paper
não foi tocado**. Mongo não foi tocado. Nenhum resultado antigo foi sobrescrito.
VLM não foi rodado.

Artefatos: `carcara-forge/paper/repro/raw_yolo11x_313/`

| arquivo | o que é |
|---|---|
| `run_raw_yolo11x.py` | Partes A+B: inferência pura, 1 `predict` por JPEG |
| `analyze_raw_p2p3.py` | Partes C–F: métricas, auditoria contra as antigas, prontidão do VLM |
| `config.json` | parâmetros efetivos, `predictor.args` como rodou, versões, sha256 dos 313 JPEGs |
| `manifest.json` / `manifest.csv` | **1.003 propostas** (conf ≥ 0,25), id local `raw11x:<acq>:<sec>:<rank>` |
| `manifest_conf020_sensitivity.*` | 1.107 propostas (conf ≥ 0,20), sensibilidade |
| `raw_predictions_tail.json` | cauda inteira, 14.858 predições (conf ≥ 0,001), bbox em px e normalizada |
| `predictions_coco_tail.json` | a mesma cauda em COCO results (para AP) |
| `diagnostic_imgsz1280/` | só para a Parte D; não entra em P2/P3 |
| `analysis.json` | todos os números deste relatório |
| `SHA256SUMS.txt` | hash de cada arquivo acima |

---

## 1. Raw YOLO11x configuration

| parâmetro | valor | fonte |
|---|---|---|
| pesos | `yolo11x.pt`, sha256 `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24`, 114.636.239 B | `config.py:29` (`TEACHER_WEIGHTS`); hash travado = o do run `20260816T220716Z-infer` |
| caminho | `C:\Users\REPRO_USER\Documents\carcara-forge\yolo11x.pt` | idem |
| Ultralytics | 8.4.120 | pacote instalado |
| torch / CUDA / GPU | 2.11.0+cu128 / 12.8 / RTX 5060 Ti | `config.json → environment` |
| modo | `predict`, 1 chamada por JPEG, `batch=1`, sem `stream` de vídeo | `run_raw_yolo11x.py` |
| `imgsz` | 640 → entrada letterbox **384×640** (rect, stride 32) | `config.py:21`; default do Ultralytics |
| conf da inferência | 0,001 (cauda inteira, para AP) | `config.py:19` |
| **conf das propostas** | **0,25** (corte pós-hoc da cauda) | `config.py:30` (`LABEL_CONF`) = default do `predict` do Ultralytics |
| conf de sensibilidade | 0,20 | piso medido das propostas antigas |
| NMS IoU | 0,70, por classe (`agnostic_nms=False`), `max_det=300` | `config.py:20`; defaults |
| classes | todas as 80 COCO (`classes=None`); escopo decidido na avaliação | `infer.py`, `classes.py` |
| device / precisão | `cuda:0`, FP32 (`half=False`), `augment=False` | `config.json` |
| transformação de imagem | nenhuma fora do Ultralytics: JPEG do disco (conferido contra `checksums.sha256`), LetterBox, BGR→RGB, /255 | — |
| tracker / persistência / IDs | **nenhum**. Verificado: depois da rodada o predictor não tem atributo `trackers` | `run_raw_yolo11x.py` |
| determinismo | a rodada repetida saiu **idêntica bit a bit** (max \|Δconf\| = 0) | `config.json` |

`predictor.args` registra `tracker: tracktrack.yaml`: é o valor default do campo,
**inerte em modo `predict`** (o tracker só é registrado por `model.track`). O
aviso `'half' is deprecated` do Ultralytics 8.4 é cosmético: `half=False` já é o
padrão.

**Checagem cruzada.** A rodada nova casa **99,8 %** das 14.860 predições do run
de 16/08 (mesma classe, IoU ≥ 0,98; IoU mediano 0,9996; \|Δconf\| máx. 0,0015).
A diferença é ruído de ponto flutuante entre batch 8 e batch 1. O run de 16/08
reproduz **exatamente** a linha publicada do sweep (969 / 601 / 368 / 130); o
novo dá 969 / 602 / 367 / 129.

### O que NÃO pode ser recuperado — o detector das 505 antigas

| item | situação |
|---|---|
| pesos / sha256 | **não registrado** |
| imgsz, NMS IoU | **não registrado** |
| conf de corte | não registrado; piso observado 0,2006 no acervo inteiro |
| tracker e parâmetros | não registrado; `track_id` em 505/505 prova tracking |
| quadro de origem | o CSV bruto era 15 fps (`detections_yolo11x_track_dist.csv`); a pasta `C:\backup hds\...\yolo\` **não existe mais** |
| código do pipeline | **não existe em disco** (busca em `Documents` inteira; só `enrich-yolo-bbox.mjs`, que consome o CSV) |

Por isso a baseline nova **não tenta imitar** o pipeline antigo: usa os parâmetros
documentados do Forge, que são os defaults do Ultralytics.

---

## 2. Raw YOLO11x manifest

| | conf ≥ 0,25 | conf ≥ 0,20 |
|---|---:|---:|
| propostas (80 classes) | **1.003** | 1.107 |
| nas 6 classes do gabarito | 969 | 1.056 |
| fora do escopo | 34 (traffic light 18, train 7, stop sign 5, airplane, bird, parking meter, bottle) | 51 |
| propostas / frame | 3,20 | 3,54 |
| frames sem nenhuma proposta | 7 | 6 |
| frames sem proposta nas 6 classes | 8 | 7 |

Cauda conf ≥ 0,001: 14.858 predições. Id local estável: posto por confiança
decrescente dentro do frame. `manifest.json` sha256 `eaee4c4c87d3…`.

---

## 3. Raw YOLO11x vs 731 human GT

### C1 — avaliação padrão

`forge.evaluate.matcher.match`: one-to-one, greedy por confiança decrescente,
mesma classe, escopo `gt_present` (predição de classe ausente do gabarito é
descartada antes; 34 descartadas). Conf ≥ 0,25.

| | IoU | det. | TP | FP | FN | P | R | F1 | IoU TP média / mediana |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **global** | 0,50 | 969 | **602** | 367 | 129 | **62,1 %** | **82,4 %** | **70,8** | 0,783 / 0,795 |
| global | 0,75 | 969 | 367 | 602 | 364 | 37,9 % | 50,2 % | 43,2 | 0,865 / 0,867 |
| **car** (GT 427) | 0,50 | 552 | 359 | 193 | 68 | 65,0 % | 84,1 % | 73,3 | 0,766 / 0,771 |
| car | 0,75 | 552 | 198 | 354 | 229 | 35,9 % | 46,4 % | 40,4 | 0,860 / 0,863 |
| **truck** (GT 255) | 0,50 | 328 | 213 | 115 | 42 | 64,9 % | 83,5 % | 73,1 | 0,815 / 0,836 |
| truck | 0,75 | 328 | 154 | 174 | 101 | 47,0 % | 60,4 % | 52,8 | 0,872 / 0,879 |

Demais classes @0,50, **só contagem, sem claim** (n pequeno): bus 23 GT / 46 det
/ 17 TP; motorcycle 13 / 13 / 4; person 10 / 28 / 8; bicycle 3 / 2 / 1.

TIDE @0,50: `fp_localization` 64 · `fp_classification` 75 · `fp_both` 22 ·
`fp_duplicate` 1 · `fp_background` 205 · `fn_from_errors` 96 ·
`fn_undetected` 33.

COCO (pycocotools, cauda conf ≥ 0,001, 6 classes, média simples):
**mAP50-95 0,313 · AP50 0,603 · AP75 0,259**. Por classe: car AP50 0,812 /
AP50-95 0,423; truck 0,795 / 0,507.

Sensibilidade conf ≥ 0,20 @0,50: 1.056 det, TP 608, P 57,6 %, R 83,2 %.

### C2 — métricas compatíveis com a lógica antiga

As propostas novas não têm `ref_det_id`, então o vínculo proposta→objeto é
**geométrico**, e a regra foi validada antes nas 501 antigas, onde o vínculo real
é conhecido.

**Definições (idênticas para antigas e novas):**

- **vínculo:** por frame, one-to-one. Estágio 1 = mesma classe, greedy por IoU
  decrescente; estágio 2 = o que sobrou, qualquer classe. Limiar de vínculo
  τ = **0,10**.
- **correta** = vinculada, mesma classe, IoU(proposta, caixa humana) ≥ **0,70**.
  É exatamente o que `kept + corrected-minor` significava: `kept` tem IoU 1 e
  `minor` tem IoU ≥ 0,70 com a própria correção (`PARITY_IOU`).
- `GT total` = 731
- `reachable by detector` = objetos vinculados a alguma proposta
- `not proposed` = 731 − reachable
- `proposal precision` = corretas / propostas
- `coverage of GT` = corretas / 731
- `reachable coverage` = corretas / reachable

**Validação nas 501 antigas (τ = 0,10):**

| | oficial (`ref_det_id`) | regra geométrica |
|---|---:|---:|
| corretas | 329 | **329** ✓ |
| precision | 65,7 % | **65,7 %** ✓ |
| coverage /731 | 45,0 % | **45,0 %** ✓ |
| reachable | 477 | **451** |
| vínculos errados (outro objeto) | — | **0** |
| vinculou proposta que o humano rejeitou | — | **0** |

Os 26 que faltam em `reachable` são 23 `corrected-redrawn` + 3 `class-changed`
cuja correção humana tem IoU < 0,10 com a caixa original: o curador moveu a
caixa para outro lugar. Por geometria isso é irrecuperável. Com τ 0,30 ou 0,50
se perdem mais vínculos (437, 414); por isso τ = 0,10.

**Tabela C2** (conf ≥ 0,25, propostas das 6 classes + train/airplane = 977;
semáforo e placa ficam fora porque o protocolo nunca os anotou):

| | propostas | reachable | not proposed | corretas | precision | coverage /731 | reachable cov. |
|---|---:|---:|---:|---:|---:|---:|---:|
| antigas, oficial | 501 | 477 | 254 | 329 | 65,7 % | 45,0 % | 69,0 % |
| antigas, geométrico | 501 | 451 | 280 | 329 | 65,7 % | 45,0 % | 72,9 % |
| **raw YOLO11x** | **977** | **702** | **29** | **438** | **44,8 %** | **59,9 %** | **62,4 %** |
| raw — car (GT 427) | 552 | 414 | 13 | 242 | 43,8 % | 56,7 % | 58,5 % |
| raw — truck (GT 255) | 328 | 243 | 12 | 178 | 54,3 % | 69,8 % | 73,3 % |
| antigas — car (oficial) | 304 | 294 | 133 | 208 | 68,4 % | 48,7 % | 70,7 % |
| antigas — truck (oficial) | 149 | 162 | 93 | 107 | 71,8 % | 42,0 % | 66,0 % |

Vereditos geométricos das 977 novas: 438 corretas · 205 `redrawn` · 59
`class-changed` · 275 `rejected`. Escopo 6 classes (969): precision 45,2 %.
Conf ≥ 0,20 (1.065): 442 corretas, precision 41,5 %, coverage 60,5 %.

Diferença a favor da tabela nova: na raw, `correta` exige a mesma classe do
objeto, então a linha por classe conta o acerto na classe do GABARITO, e o
artefato de denominador da tabela antiga (bus 125 %, train 400 %) deixa de
existir.

**C3 — o que o humano teve de desenhar.** Dos **254** objetos `added` (nenhuma
proposta antiga), o YOLO11x cru acha **182** com mesma classe e IoU ≥ 0,50 (car
102, truck 66, bus 6, person 5, motorcycle 2, bicycle 1), e 233 têm alguma
proposta crua a IoU ≥ 0,10. O "detector miss" do P2 antigo era, em grande
parte, perda da cadeia de tracking, não do detector.

### ⚠️ Os números antigos são contraditos

- A tabela P2 antiga (P 65,7 % / R 45,0 % / 254 não propostos) **não descreve o
  YOLO11x**: descreve o pipeline com tracking.
- O paper já contém uma linha de YOLO11x cru (sweep conf 0,25: P 62,0 % /
  R 82,2 %) que contradiz a tabela P2 no mesmo documento. A reconstrução resolve
  a contradição a favor do sweep.

### As 244 propostas sem objeto no gabarito

**244** propostas cruas nas 6 classes (conf ≥ 0,25, mediana 0,40, p95 0,78) não
estão a IoU ≥ 0,10 de nenhum objeto do gabarito, e nenhuma proposta antiga
cobria aquela região. Pela política do projeto (o gabarito manual é verdade
absoluta; detecção ausente dele é falso positivo do modelo, nunca evidência de
falha do gabarito), **elas contam como FP**, e a precision acima **é o valor**,
não um piso. O único registro cabível é de método: o curador trabalhou sobre
pré-anotações de outro pipeline e completou o gabarito com `added`. Ver §5.

---

## 4. Raw YOLO11x vs old Forge proposals — auditoria interna, não resultado

Raw conf ≥ 0,25 (1.003) × antigas (505). Pareamento one-to-one por frame,
qualquer classe, greedy por IoU, IoU ≥ 0,30.

| | valor |
|---|---:|
| raw / old | 1.003 / 505 |
| pares | 434 |
| só antigas | 71 (**59** mesmo contra a cauda inteira a conf 0,001) |
| só raw | 569 |
| frames em que toda caixa antiga tem par | 243 / 313 |
| mesma classe | 86,4 % (59 pares com classe diferente) |
| IoU raw×old — mediana / p05 / p95 | 0,819 / 0,437 / 0,947 |
| IoU ≥ 0,95 · [0,90,0,95) · [0,80,0,90) · [0,70,0,80) · [0,50,0,70) · [0,30,0,50) | 19 · 87 · 126 · 75 · 90 · 37 |
| \|Δconf\| — mediana / p95 | **0,153** / 0,461 |
| \|Δconf\| < 0,01 · [0,01,0,05) · [0,05,0,10) · ≥ 0,10 | 9 · 56 · 66 · **303** |
| quase idênticas (mesma classe, IoU ≥ 0,90, \|Δconf\| ≤ 0,05) | **17 / 505** |
| antiga cobrindo ≥ 2 raw (IoU ≥ 0,5) — fusão | 19 |
| raw cobrindo ≥ 2 antigas — duplicata do tracker | 12 |

**Mecanismos, em ordem de evidência:**

1. **Caixa = estado do filtro de Kalman.** No Ultralytics 8.4, `STrack.tlwh`
   devolve `self.mean[:4]` (`byte_tracker.py:163-170`): a caixa do tracker não é
   a detecção. A confiança, por outro lado, é a da detecção (`self.score =
   new_track.score`).
2. **Limiares do tracker.** `track_high_thresh` / `new_track_thresh` = 0,25;
   objetos só viram track depois de confirmados. Isso explica objetos ausentes
   e só 6 propostas antigas abaixo de 0,25.
3. **A confiança não bate, e isso não se explica por quadro diferente.** Com o
   mesmo pixel e o mesmo modelo, a confiança seria igual mesmo com tracking.
   Testado com velocidade da telemetria: com o carro **parado** (11 pares, IoU
   mediano 0,94), o \|Δconf\| mediano é **0,169**, e nenhum par fica abaixo de
   0,01. Rodar a 1280 também não aproxima (quase idênticas: 15; \|Δconf\| mediano
   0,169). **Conclusão: o detector das antigas difere de qualquer configuração
   reprodutível a partir destes JPEGs.** A causa exata não é recuperável.

> **The old 501 proposals materially differ from raw YOLO11x.**

---

## 5. Impact on Pillar 2

1. **Quase idênticas às antigas:** 17 / 501 (15 `kept`, 2 `corrected-minor`).
2. **Mudaram materialmente ou sumiram:** 484 / 501.
3. **O que continua válido sem ressalva:** o **gabarito** — 731 caixas e classes
   humanas. Ele é a régua e não depende de qual proposta está sendo medida. O
   veredito de cada proposta nova sai da geometria contra o gabarito, com a
   regra validada na §3: nas 501 antigas ela reproduz 329/329 corretas, 24/24
   `rejected`, 52/55 `class-changed` e 70/93 `redrawn`, com 0 vínculo errado.
   Nos 17 pares quase idênticos, o veredito antigo transferido e o geométrico
   concordam 17/17; a transferência é, portanto, **desnecessária**.
   Também seguro: a rejeição humana antiga como **negativo explícito** para
   proposta crua sobre a mesma região (IoU ≥ 0,5): 5 casos.
4. **Não é aceitável reutilizar o veredito antigo** quando IoU(raw, old) < 0,90
   ou a classe difere. O veredito é função da geometria e do rótulo da caixa
   antiga: `redrawn` diz que a caixa de Kalman estava ruim, não a crua (Q3: IoU
   antigo 0,375, cru 0,885); `class-changed` pertence ao rótulo antigo (Q2); e
   `rejected` sobre um fantasma do tracker não diz nada sobre outra caixa.
5. **Propostas novas que nenhum humano julgou:** **244** (6 classes, conf ≥
   0,25) — car 162, truck 58, person 12, bus 9, motorcycle 3. **Nenhuma precisa
   de auditoria para o P2**: pela política do gabarito, são FP do modelo. Uma
   revisão humana dessas regiões só caberia como enriquecimento de coleção, e o
   resultado **não** altera os 731 nem a pontuação. Outras 25 sem vínculo
   sobrepõem uma caixa antiga não rejeitada (duplicatas de objeto já anotado →
   FP pela regra one-to-one), e 27 são de classe fora do escopo (mascaradas em
   `gt_present`). Auditoria humana nova, portanto: **0 obrigatórias**.

**Regra conservadora de reaproveitamento (proposta, NÃO aplicada):** mesma
classe ∧ IoU(raw, old) ≥ 0,90 ∧ \|Δconf\| ≤ 0,05. Pega 17 caixas e é redundante
com o veredito geométrico. **Recomendação: não transferir veredito nenhum.**

**Viés de geometria herdado.** 245 caixas `kept` do gabarito **são** a caixa de
Kalman antiga, aceita pelo humano como justa. A melhor proposta crua de mesma
classe tem IoU mediano 0,854 com elas, e **42** ficam abaixo de 0,70; nessas, a
crua é contada como `redrawn` contra uma geometria que veio do tracker. Afeta C2
(IoU 0,70) e C1 @0,75; em C1 @0,50 o efeito é pequeno.

---

## 6. Readiness for Pillar 3

- **Input compatível.** `BoxTask(acq_id, sec, det_index, det_id, proposed_class,
  conf, box, truth)` é montável direto do `manifest.json` novo (`det_index` =
  `rank_in_frame`, `det_id` = `proposal_id`). `draw_box`/`crop_around`
  renderizaram (cena 1920×1080, recorte 312×379) e `phase1_prompt_v2` monta (sha
  `ada20377…`), tudo **sem chamar o modelo**. `run_phase1(..., v2=True)` aceita
  a lista como está.
- **Prompt e crop reutilizáveis sem alteração:** `_TAXONOMY_V2`,
  `PHASE1_SCHEMA_V2`, `verdict_from_v2`, `qwen3-vl:8b-instruct`, temperatura 0,
  duas imagens, pad 0,6. Nada ali depende de `ref_det_id` ou de track.
- **O que o scorer precisa trocar:**
  1. `truth`: sai de `human_verdict(stored_detection)` e passa a ser o veredito
     geométrico contra o gabarito (correta / redrawn / class-changed / rejected,
     definições da §3). `score_phase1` já trata `kept` e `corrected-minor` como
     a mesma coisa (`_TRUTH_OK`).
  2. `human_corrected_class`: indexado por `acq:sec:índice` das antigas; vira
     `proposal_id → classe do objeto vinculado`.
  3. `compare_arms.py` / `car_truck_breakdown.py`: `GT`, `RECOVERABLE` e
     `added` vêm de `stored_detections`; passam a vir do vínculo geométrico
     (731 / 702 / 29).
  4. Proposta sem vínculo (as 244 incluídas) recebe `truth = rejected`, como
     qualquer FP de fundo. É o que a política do gabarito determina; o VLM que
     as rejeita acerta.
  5. A Fase 2 (recall) usa `stored_detections` como "confirmados"; se ela
     continuar no paper, passa a usar as propostas cruas mantidas pelo VLM.
- **Custo:** 977 chamadas (6 classes + train/airplane) × 4,88 s ≈ **80 min** no
  v2 (taxa medida em 21/08).
- v1 e v3 só servem de ablação se forem **rerodados sobre as mesmas 977**; os
  runs antigos são sobre as 501 e não se comparam.

---

## G. Exemplos qualitativos — os quatro atuais caem

Verificado nas predições cruas, sem rodar nada:

| figura | afirmação atual | com YOLO11x cru |
|---|---|---|
| Q1 detector miss | nenhuma proposta para o caminhão | **detecta**: `truck` 0,925, IoU 0,920 com o GT |
| Q2 class fix | proposta `train` corrigida para `truck` | **não há `train`**: só `truck` 0,949, IoU 0,970 com o GT |
| Q3 bad box accepted | caixa ruim, IoU 0,375 | caixa crua IoU **0,885** com a humana |
| Q4 false positive | `car` 0,756 no acostamento | **nenhuma proposta** no frame (cauda máx. 0,029, sem sobreposição) |

Os quatro modos de falha da figura atual são propriedades do pipeline com
tracking. A figura precisa ser refeita depois do novo P2/P3; até lá, ela não
pode ilustrar YOLO11x. Isso inclui o `fig3_qualitative_examples.png` do zip
`paper_ready`.

---

## 7. Recommendation

1. **P2 = YOLO11x cru, quadro a quadro, contra os 731 objetos.** Pesos com sha,
   imgsz 640, NMS 0,70, propostas conf ≥ 0,25. Duas leituras declaradas lado a
   lado:
   - **C1** detector padrão: IoU 0,50 como principal, 0,75 e AP/mAP50-95 como
     complemento. É a mesma linha que o sweep já publica.
   - **C2** qualidade da proposta para curadoria: vínculo geométrico τ 0,10,
     correta = mesma classe e IoU ≥ 0,70.
2. **Os 731 ficam como estão, e não há auditoria humana bloqueando o P2.**
   Proposta crua sem objeto no gabarito é FP (política do gabarito). A precision
   sai como um número só, sem intervalo.
3. **P3 = as mesmas propostas cruas → VLM decomposed (v2), sem mudar prompt nem
   schema**, pontuado com a verdade geométrica acima. v1/v3 só como ablação e
   só se rerodados nas mesmas propostas; senão, sair do paper.
4. **Sai do paper:** a tabela P2 das 501, os três braços do P3 sobre as 501, os
   números car/truck derivados deles e a figura qualitativa atual. O
   Forge-com-tracking não entra nem como comparação.
5. **Vai no método:** (a) as pré-anotações que o curador viu vieram de outro
   pipeline, com tracking; (b) 245 caixas do gabarito herdam a geometria desse
   pipeline; (c) a completude do gabarito fora das regiões propostas depende do
   `added` do curador. (a) e (b) são limitações, não impedimentos.
