# P2 × P3 no conjunto congelado de 969 propostas

10/09/2026. Documento de dados/método, **não é texto de paper; o paper não foi
tocado.** Mongo não foi tocado. Nenhum resultado antigo foi sobrescrito.

## Conjunto congelado

| | valor |
|---|---|
| detector | YOLO11x cru, quadro a quadro, sem tracking (`REPORT-RAW-YOLO11X-P2P3.md` §1) |
| corte | conf ≥ 0,25 |
| classes | só as 6 do gabarito: car, truck, bus, motorcycle, bicycle, person |
| propostas | **969** (car 552 · truck 328 · bus 46 · person 28 · motorcycle 13 · bicycle 2) |
| arquivo | `p2p3_frozen_969.json`, sha256 `6b03b3238e50…` |
| gabarito | os mesmos 731 objetos |

Por que já apareceram outros números: **1.003** = todas as 80 classes; **977** =
969 + 7 `train` + 1 `airplane` (escopo que eu tinha usado na tabela C2 do
relatório anterior). Os dois braços abaixo leem o mesmo arquivo congelado.

## VLM (P3)

| | valor |
|---|---|
| braço | decomposed (v2), **sem** v1/v3 |
| modelo | `qwen3-vl:8b-instruct`, digest `0533d743…` (o mesmo dos runs antigos), Ollama 0.32.15, temperatura 0 |
| prompt / schema / render | `phase1_prompt_v2`, `PHASE1_SCHEMA_V2`, `draw_box` + `crop_around`, sem alteração |
| chamadas | 969, **0 falhas** |
| saída | `vlm_v2_969/phase1.json` + `meta.json` |

**Sessões.** As 499 primeiras caixas saíram em duas sessões com o `ollama serve`
nos padrões (flash attention ligado automaticamente, cache de prompt em RAM
ligado). As duas foram mortas por falta de RAM; o checkpoint por frame preservou
tudo, sem duplicata. As 470 restantes saíram com o mesmo servidor, **só com o
cache de prompt desligado** (`LLAMA_ARG_CACHE_RAM=0`). Esse cache guarda ~490 MiB
por chamada, até 8 GiB, e não reaproveita nada aqui. A troca foi validada antes:
78 caixas já feitas foram rerodadas nas duas configurações e deram **78/78
vereditos idênticos** ao checkpoint. O `meta.json` só registra a última sessão
(470 chamadas, 4,51 s/chamada); as duas primeiras morreram antes de gravá-lo.

## Braços e réguas

- **YOLO:** as 969, com o rótulo do detector.
- **YOLO+VLM:** tira o que o VLM rejeitou; `class-changed` troca o rótulo pelo do
  VLM; kept / corrected-minor / corrected-redrawn mantêm a caixa. O VLM não move
  caixa.
- **C1** (padrão de detector): matcher one-to-one, confiança decrescente, mesma
  classe, IoU 0,50 (principal) e 0,75.
- **C2** (lógica da curadoria): vínculo geométrico τ 0,10; **correta** = rótulo
  final igual à classe do objeto **e** IoU ≥ 0,70. É mais estrita que o
  `score_phase1` antigo, que dava acerto a correção de classe sem olhar a caixa.

## Resultados — global

**C1, IoU 0,50**

| | det. | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO | 969 | 602 | 367 | 129 | 62,1 % | 82,4 % | 70,8 |
| YOLO+VLM | 939 | 628 | 311 | 103 | 66,9 % | 85,9 % | 75,2 |
| Δ absoluto | −30 | +26 | −56 | −26 | +4,8 pp | +3,5 pp | +4,4 |
| Δ relativo | −3,1 % | +4,3 % | −15,3 % | −20,2 % | +7,7 % | +4,2 % | +6,2 % |

**C1, IoU 0,75**

| | det. | TP | FP | FN | P | R | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO | 969 | 367 | 602 | 364 | 37,9 % | 50,2 % | 43,2 |
| YOLO+VLM | 939 | 383 | 556 | 348 | 40,8 % | 52,4 % | 45,9 |
| Δ absoluto | −30 | +16 | −46 | −16 | +2,9 pp | +2,2 pp | +2,7 |
| Δ relativo | −3,1 % | +4,4 % | −7,6 % | −4,4 % | +7,7 % | +4,4 % | +6,2 % |

**C2**

| | retidas | corretas | precision | coverage /731 | reachable coverage /698 |
|---|---:|---:|---:|---:|---:|
| YOLO | 969 | 438 | 45,2 % | 59,9 % | 62,8 % |
| YOLO+VLM | 939 | 454 | 48,3 % | 62,1 % | 65,0 % |
| Δ absoluto | −30 | +16 | +3,1 pp | +2,2 pp | +2,2 pp |
| Δ relativo | −3,1 % | +3,7 % | +6,9 % | +3,7 % | +3,5 % |

## Resultados — car e truck

**C1, IoU 0,50** (det./P por rótulo final; FN/R pela classe do gabarito)

| | braço | det. | TP | FP | FN | P | R | F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **car** (GT 427) | YOLO | 552 | 359 | 193 | 68 | 65,0 % | 84,1 % | 73,3 |
| | YOLO+VLM | 565 | 378 | 187 | 49 | 66,9 % | 88,5 % | 76,2 |
| | Δ | +13 | +19 (+5,3 %) | −6 (−3,1 %) | −19 (−27,9 %) | +1,9 pp (+2,9 %) | +4,4 pp (+5,2 %) | +2,9 (+4,0 %) |
| **truck** (GT 255) | YOLO | 328 | 213 | 115 | 42 | 64,9 % | 83,5 % | 73,1 |
| | YOLO+VLM | 296 | 219 | 77 | 36 | 74,0 % | 85,9 % | 79,5 |
| | Δ | −32 | +6 (+2,8 %) | −38 (−33,0 %) | −6 (−14,3 %) | +9,1 pp (+14,0 %) | +2,4 pp (+2,9 %) | +6,4 (+8,8 %) |

**C2**

| | braço | retidas | corretas | precision | coverage | reachable cov. |
|---|---|---:|---:|---:|---:|---:|
| **car** | YOLO | 552 | 242 | 43,8 % | 56,7 % | 58,5 % |
| | YOLO+VLM | 565 | 257 | 45,5 % | 60,2 % | 62,1 % |
| | Δ | +13 | +15 (+6,2 %) | +1,7 pp (+3,9 %) | +3,5 pp (+6,2 %) | +3,6 pp (+6,2 %) |
| **truck** | YOLO | 328 | 178 | 54,3 % | 69,8 % | 74,5 % |
| | YOLO+VLM | 296 | 181 | 61,1 % | 71,0 % | 75,7 % |
| | Δ | −32 | +3 (+1,7 %) | +6,8 pp (+12,5 %) | +1,2 pp (+1,7 %) | +1,2 pp (+1,6 %) |

`car` ganha em recall (o VLM leva 44 `truck` → `car`); `truck` ganha em precisão
(as mesmas caixas saem da linha de truck).

## O que o VLM fez com as 969

| ação | n |
|---|---:|
| manteve caixa e rótulo | **853** (kept 195 · corrected-minor 642 · corrected-redrawn 16) |
| manteve caixa, **trocou a classe** | **86** |
| **rejeitou** | **30** |

**Trocas de classe (86):** truck→car 44 · bus→truck 15 · person→motorcycle 9 ·
car→truck 5 · car→motorcycle 4 · bus→car 3 · truck→motorcycle 2 · outras 4.
- **31** levaram a caixa para a classe do objeto do gabarito;
- **4** estragaram uma caixa que estava correta;
- **44** caíram em caixas sem objeto vinculado — na maioria duplicata car+truck
  sobre o mesmo veículo, que o NMS por classe deixa passar; o rótulo novo não
  vira acerto porque o objeto já foi levado pela outra caixa;
- **7** sobre caixa de geometria ruim ou para a classe errada.

**Rejeições (30):** 28 eram mesmo FP de fundo, 1 `redrawn`, 1 `class-changed`
(**93 % de acerto quando rejeita**). Mas das **271** caixas que são FP de fundo,
o VLM rejeitou só **28 (10 %)**: 180 ele marcou `corrected-minor`.

**Geometria:** das 205 caixas que precisam ser redesenhadas (mesma classe, IoU
< 0,70), o VLM pediu redesenho em **1**; 196 viraram `corrected-minor`. É o mesmo
achado do P3 antigo, agora sobre YOLO cru: o VLM julga semântica (existe? que
classe?) muito melhor que geometria.

**Leitura:** o ganho do VLM é **correção de classe** e uma poda pequena e precisa
de FP. Ele não conserta caixa e deixa passar 90 % do fundo.

## Ressalvas medidas

- **Caixas quase duplicadas no gabarito:** 4 pares com IoU ≥ 0,70 (8 caixas; 3
  pares da mesma classe), herdados de propostas antigas co-localizadas. Explicam
  **5 dos 129 FN** do YOLO @0,50. Pela política o gabarito não muda.
- **245 caixas `kept` do gabarito** são geometria de Kalman do pipeline antigo
  (ver `REPORT-RAW-YOLO11X-P2P3.md` §5). Afeta mais o C2 e o C1 @0,75.
- **Cache do Ollama:** as duas configurações foram verificadas idênticas em 78
  caixas, não nas 969.

## Novos exemplos candidatos

Lista completa em `p2p3_969_examples.json`. Folha de conferência (**não é figura
do paper**) em `p2p3_969_candidates_contact_sheet.jpg`: verde tracejado =
gabarito, âmbar = proposta crua. Critérios: a frase da legenda tem de ser
literalmente verdadeira e o painel não pode ter uma segunda proposta sobre o
mesmo objeto (o problema do Q2 antigo).

| painel | recomendado | por quê | alternativa |
|---|---|---|---|
| **detector miss** (25) | `20240201163320:36` — truck (desenhado pelo humano) | caminhão bem visível, de dia; nenhuma proposta ≥ 0,25 de nenhuma classe a IoU ≥ 0,3; a melhor predição da cauda sobre ele tem conf 0,11 | `20240129194123:80` — carro à noite, melhor da cauda `truck` 0,199 |
| **class correction** (29) | `20240129221622:175` — YOLO `truck` 0,878 → VLM `car` = gabarito, IoU 0,861 | carro inequívoco; nenhuma outra proposta no objeto | `20240129195123:283` (truck→car, noite, IoU 0,823); `20240129163448:39` (bus→truck, IoU 0,884, não conferido visualmente) |
| **bad box accepted** (186) | `20240201163320:133` — truck 0,43, IoU **0,378** com a caixa humana | a caixa pega só o cavalo mecânico, o humano desenhou o caminhão inteiro; o VLM respondeu `needs_nudge` ("slightly cropped") e não pediu redesenho | `20240201164822:268` (truck, IoU 0,466) |
| **false positive rejected** (28) | `20240129195623:299` — car 0,268 sobre reflexo de farol | nenhum objeto do gabarito toca a caixa; o VLM rejeitou citando o reflexo | `20240129212623:147` (car 0,33, também reflexo) |

Ressalvas dos exemplos:
- Os FP rejeitados maiores são **reflexo de farol à noite**: atrás de um farol
  existe fisicamente um veículo em sentido contrário. Pela política é FP (o
  gabarito não o anota), mas um revisor pode questionar. Nenhum dos 28 cai numa
  região que o humano rejeitou explicitamente.
- `class correction`: vários candidatos são **van → car**, que depende da
  convenção do gabarito; o recomendado é carro sem ambiguidade.
- Os exemplos `detector miss` e `bad box accepted` recomendados são do mesmo
  vídeo (`20240201163320`, s 36 e 133), em momentos diferentes.
- Caption sugerida para (b): "one detector proposal labeled truck was relabeled
  car by the decomposed VLM, agreeing with the human annotation" — sem proposta
  co-localizada no painel.
