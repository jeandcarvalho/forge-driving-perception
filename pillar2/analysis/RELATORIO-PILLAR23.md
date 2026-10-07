# Pillars 2 e 3 — relatório de fechamento experimental

Estado em 07/09/2026. **Os quatro braços estão fechados.** Este documento é do
sistema/dados — não é texto de paper.

Todos os números abaixo saem de artefato em disco. Onde um número publicado não
tem artefato que o reproduza, isso está dito explicitamente.

---

## 0. Situação do quarto braço — FECHADO

| | |
|---|---|
| Status | **COMPLETO — 501 de 501 caixas**, zero erros |
| Run | `runs\20260907T164834Z-vlm-audit-v3-single-classfix` |
| `phase1.json` | 501 linhas, 501 chaves únicas, nenhum `vlm` nulo |
| Custo da última sessão | 50 chamadas, 209,1 s, 4,18 s/caixa, 170 681 prompt tokens |

### Histórico das três tentativas

| Tentativa | Quando | Resultado |
|---|---|---|
| 1ª | 04/09 19:40 | máquina travada; perdeu 100% (o `phase1.json` só era escrito no fim). Deixou `runs\20260904T224035Z-...` vazio, preservado |
| 2ª | 04/09 20:10 | parou em 6 caixas (5 + 1 falha). Já com checkpoint — nada perdido |
| 3ª | 07/09 13:20–13:48 | **fechou as 501**, em três lançamentos: 451 → +0 → +50 |

O checkpoint (`runs\_v3_classfix_cache\phase1.jsonl`, JSONL append com `flush` +
`fsync` por caixa) funcionou como projetado: os dois primeiros lançamentos da 3ª
tentativa foram mortos por um watchdog de memória externo ao script, e nenhuma
inferência já paga foi repetida.

### Diagnóstico da memória — corrigido

O relatório de 04/09 atribuiu a parada a `llama-server` consumindo **16,4 GB de
commit**. **Medido em 07/09, o número real é 8,04 GB** (0,77 GB residentes);
carregar o modelo move o commit total em +9,29 GB, não +16,4.

A causa provável da lentidão de ~30× da 2ª tentativa é outra, e o próprio
relatório anterior registrou a pista: **8,8 GB de VRAM livres** naquela hora. O
modelo ocupa ~8,6 GB mais contexto, então não coube inteiro na placa e camadas
transbordaram para a CPU — o que produz os dois sintomas de uma vez, o commit
inchado no host e a inferência lenta. Em 07/09, com 14,3 GB de VRAM livres,
`ollama ps` reportou `100% GPU` e a rodada andou a **4,4 s/caixa**, a mesma
velocidade da v1. Não houve thrashing.

Precaução que fica registrada: **manter a GPU livre durante a rodada**. Não é
preciso mexer no pagefile nem fechar aplicativos.

## 1. Decomposição 402 = 254 + 148

Artefato: `pillar23-audit-reanalysis.json` → `decomposition_254_402`.

| | |
|---|---|
| Ground truth | 731 |
| Propostas do detector | 501 |
| TP base | 329 |
| FP base | 172 |
| **FN base** | **402** |

Identidade conferida no artefato: `GT − TP = 731 − 329 = 402 = FN`.

| Parcela | n | Significado |
|---|---|---|
| `added` | **254** | objetos do gabarito que o detector **nunca** propôs — desenhados à mão (`origin == "added"`) |
| `had_proposal_but_failed` | **148** | tinham proposta, mas a proposta não foi aceita como correta |
| — geometria refeita | 93 | mesma classe, IoU(orig, correção) < 0,70 |
| — classe trocada | 55 | o humano mudou o rótulo |

`254 + 148 = 402` ✓ (`sum_check.equal = true`).

Os 254 por classe: car 133, truck 93, bus 11, motorcycle 8, person 6, bicycle 3.

## 2. Os cinco verdicts das 501 propostas

| Verdict | n | % |
|---|---:|---:|
| kept | 245 | 48,9 |
| corrected-minor | 84 | 16,8 |
| corrected-redrawn | 93 | 18,6 |
| class-changed | 55 | 11,0 |
| rejected | 24 | 4,8 |
| **total** | **501** | 100 |

Antes da exclusão de classes (avião e trem), a régua era 505, com kept = 249 e
todo o resto igual.

O banco guarda **quatro** estados brutos (`kept | corrected | rejected |
unreviewed`). As cinco categorias saem de `corrected` decomposto por classe e por
IoU. Sanidade: `unreviewed` em frames revisados = 0, e `corrected` sem payload de
correção = 0.

## 3. Regra minor vs. redraw

**IoU(caixa original, correção humana) ≥ 0,70 → `corrected-minor`; abaixo →
`corrected-redrawn`.** Mesmo limiar (`parity_iou = 0.7`) usado no relatório de
paridade. É IoU do original contra a *correção humana*, não contra o gabarito.

## 4. Definição de "correto" em cada configuração

Sem matching por IoU contra o gabarito: o vínculo é `ref_det_id`.

- **Base YOLO** — TP = veredito humano ∈ {kept, corrected-minor}. As 501 propostas ficam.
- **Braços com VLM** — TP = a caixa **sobrevive** à curadoria do VLM **E**
  (veredito humano ∈ {kept, corrected-minor} com o rótulo preservado)
  **OU** (veredito humano = class-changed **e** o rótulo final é exatamente o que o humano pôs).

Consequência que vale registrar: `corrected-redrawn` **nunca** vira TP em nenhum
braço. Nenhum braço da Fase 1 devolve coordenada, então o julgamento geométrico
do VLM não move a tabela principal — ele só aparece na análise da §7.

Denominadores de recall:

| | n |
|---|---:|
| GT total | 731 |
| GT sem proposta do detector | 254 |
| **GT com proposta (recuperável)** | **477** |

`477 + 254 = 731` ✓. As 24 `rejected` são propostas sobre nada: entram em
propostas, não em GT.

## 5. Tabela comparável das configurações

Artefato: `pillar3-single-classfix.json` → `main_table`. Gerada por
`compare_arms.py`, que faz glob do run do v3.

| # | Configuração | Caixas | Correto | Errado | Não recup. | Precisão | Cobertura /731 | Recuperação /477 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Base YOLO11x (sem VLM) | 501 | 329 | 172 | 402 | 65,7% | 45,0% | 69,0% |
| 2 | VLM single verdict (v1) | 417 | 312 | 105 | 419 | 74,8% | 42,7% | 65,4% |
| 3 | **VLM single + class correction (v3)** | **418** | **317** | **101** | **414** | **75,8%** | **43,4%** | **66,5%** |
| 4 | VLM decomposed (v2) | 468 | 368 | 100 | 363 | 78,6% | 50,3% | 77,1% |

Correção de classe:

| Configuração | Tentadas | Certas | Erradas |
|---|---:|---:|---:|
| Base YOLO | 0 | 0 | 0 |
| VLM single (v1) | 7 | 0 | 6 |
| **VLM single + classfix (v3)** | **8** | **7** | **1** |
| VLM decomposed (v2) | 54 | 43 | 4 |

**A linha 3 responde a pergunta do experimento, e a resposta é a decomposição.**

O ganho do v2 sobre o v1 era +56 TP, +3,8 pp de precisão e +11,7 pp de
recuperação, e confundia dois fatores: decompor a pergunta **e** permitir corrigir
classe. Isolando o segundo, o v3 rende **+5 TP, +1,0 pp de precisão e +1,1 pp de
recuperação** sobre o v1 — perto de nada. O v2 continua **+51 TP** à frente do v3.

Ou seja: **~9% do ganho vem de deixar o modelo corrigir classe; ~91% vem de
decompor a pergunta.**

O bug de schema da v1 foi de fato consertado — o v1 tentou 7 correções e emitiu
classe em 0 delas (`correct_class` estava fora de `required` e o decodificador
restrito o omitia); o v3 tentou 8 e emitiu classe em 8, acertando 7. Mas
consertar o bug quase não move o placar, porque **com um enum único de 5
vereditos o modelo raramente escolhe `class-changed`**: 8 vezes no v3 contra 54
no v2. A capacidade estava travada por duas coisas, e o schema era a menor
delas.

Detalhe por transição (`detail` no artefato):

| | v1 | **v3** | v2 |
|---|---:|---:|---:|
| manteve caixa boa | 312 | **310** | 325 |
| manteve caixa mal posicionada | 69 | **69** | 73 |
| manteve FP de fundo | 3 | **3** | 13 |
| removeu caixa boa | 17 | **18** | 0 |
| removeu FP de erro | 46 | **44** | 22 |
| removeu FP de fundo | 21 | **21** | 11 |
| corrigiu rótulo certo | 0 | **7** | 43 |
| estragou caixa boa | 0 | **1** | 4 |
| rótulo continua errado | 33 | **28** | 10 |

O v3 é o v1 quase linha a linha; a única coluna que se move de verdade é
"corrigiu rótulo certo" (0 → 7), e ela vem acompanhada de "rótulo continua
errado" caindo só de 33 para 28.

## 6. Car e Truck separados

Artefato: `pillar3-single-classfix.json` → `car_truck`.

> Ressalva registrada no artefato: `class` é a classe **proposta** pelo detector,
> não a do gabarito. Uma correção de classe move o acerto para a linha da classe
> proposta, não da verdadeira.

### car — GT 427, sem proposta 133, com proposta 294, propostas 304

| Configuração | Retidas | Correto | Errado | Precisão | Cobertura | Recuperação |
|---|---:|---:|---:|---:|---:|---:|
| Base YOLO | 304 | 208 | 96 | 68,4% | 48,7% | 70,7% |
| VLM single (v1) | 255 | 192 | 63 | 75,3% | 45,0% | 65,3% |
| **VLM single + classfix (v3)** | **255** | **192** | **63** | **75,3%** | **45,0%** | **65,3%** |
| VLM decomposed (v2) | 283 | 217 | 66 | 76,7% | 50,8% | 73,8% |

Em `car` o v3 é **idêntico** ao v1 nas seis colunas.

### truck — GT 255, sem proposta 93, com proposta 162, propostas 149

| Configuração | Retidas | Correto | Errado | Precisão | Cobertura | Recuperação |
|---|---:|---:|---:|---:|---:|---:|
| Base YOLO | 149 | 107 | 42 | 71,8% | 42,0% | 66,0% |
| VLM single (v1) | 139 | 107 | 32 | 77,0% | 42,0% | 66,0% |
| **VLM single + classfix (v3)** | **139** | **110** | **29** | **79,1%** | **43,1%** | **67,9%** |
| VLM decomposed (v2) | 142 | 118 | 24 | 83,1% | 46,3% | 72,8% |

Todo o ganho do v3 sobre o v1 está em `truck`: +3 acertos, das 7 correções de
classe que o v3 acertou (6 para `car`, 2 para `truck` como rótulo emitido). Em
`car` não há diferença nenhuma.

## 7. Recuperação condicional (universo com proposta)

Coluna "Recuperação /477" da §5: TP sobre os 477 objetos do gabarito que
**tinham** proposta do detector. É a métrica honesta para julgar a curadoria,
porque os 254 nunca propostos estão fora do alcance de qualquer braço de Fase 1 —
nenhum deles inventa caixa.

Base 69,0% → v1 65,4% → **v3 66,5%** → v2 77,1%.

O v3 recupera 1,1 pp a mais que o v1 e fica 10,6 pp atrás do v2.

## 8. Semântica vs. geometria

Artefato: `pillar3-semantic-geometry.json`.

- **Semântica** = presença (`object_present` / `rejected`) e classe.
- **Geometria** = qualidade da caixa (kept × corrected-minor × corrected-redrawn).

### Presença

| | v1 | **v3** | v2 |
|---|---:|---:|---:|
| fundo corretamente rejeitado | 21 | **21** | 11 |
| fundo mantido por engano | 3 | **3** | 13 |
| objeto real rejeitado por engano | 63 | **62** | 22 |
| objeto real corretamente mantido | 414 | **415** | 455 |
| precisão da rejeição | 25,0% | **25,3%** | 33,3% |
| recall da rejeição (sobre 24 de fundo) | 87,5% | **87,5%** | 45,8% |
| caixas boas destruídas | 17 | **18** | 0 |

O comportamento de presença do v3 é o do v1 dentro de uma caixa — o que era de
esperar, já que o schema não mexe na pergunta de presença.

### Correção de classe (sobre os 55 erros de classe do humano)

| | v1 | **v3** | v2 |
|---|---:|---:|---:|
| tentativas | 7 | **8** | 54 |
| certas | **0** | **7** | **43** |
| erradas numa classe realmente errada | 0 | **0** | 0 |
| sem classe emitida (bug de schema) | **6** | **0** | 0 |
| tentou onde o humano não relabelou | 1 | **1** | 11 |
| erros de classe não vistos | 49 | **48** | 12 |
| precisão das tentativas | 0,0% | **87,5%** | 79,6% |
| recall sobre os 55 | 0,0% | **12,7%** | 78,2% |

**Este é o bloco decisivo do experimento.** Quando o v3 tenta, ele acerta —
87,5% de precisão, a maior das três. Mas ele quase não tenta: **8 tentativas
contra 54 do v2**, e por isso o recall sobre os 55 erros de classe do humano fica
em 12,7% contra 78,2%. Consertar o schema deu ao modelo a *capacidade* de
corrigir classe; foi decompor a pergunta que o fez *exercê-la*.

> Registrado no artefato: **o número publicado "43 de 55" é _recall_ sobre os 55
> erros de classe do humano, não acurácia das tentativas do VLM.** A acurácia das
> tentativas do v2 é 43/54 = 79,6%.

### Geometria — headline

| | v1 | **v3** | v2 |
|---|---:|---:|---:|
| caixas ruins aceitas como boas | 62 de 93 (66,7%) | **62 de 93 (66,7%)** | 21 de 93 (22,6%) |
| caixas boas marcadas ou rejeitadas | 44 de 329 (13,4%) | **44 de 329 (13,4%)** | 194 de 329 (59,0%) |
| **separação** | **20,0 pp** | **20,0 pp** | **18,5 pp** |

Separação = P(VLM diz "caixa ok" | humano aceitou) − P(mesmo | humano refez).
Zero significaria que o julgamento geométrico não carrega informação nenhuma. Os
dois esquemas ficam perto de 20 pp: o sinal geométrico existe, mas é fraco, e os
dois o deslocam em direções opostas — o v1 é complacente (aceita caixa ruim), o
v2 é ansioso (marca caixa boa). O v3 reproduz o v1 **dígito por dígito** nas três
linhas, o que confirma que o schema não toca o julgamento geométrico.

## 9. Matriz human verdict × VLM verdict

Artefato: `arms.*.geometry.human_x_vlm_full_matrix`. Linha = humano, coluna = VLM.

### v1 — single verdict

| humano \ VLM | kept | corrected-minor | corrected-redrawn | class-changed | rejected |
|---|---:|---:|---:|---:|---:|
| kept (245) | 205 | 0 | 24 | 1 | 15 |
| corrected-minor (84) | 79 | 0 | 3 | 0 | 2 |
| corrected-redrawn (93) | 62 | 0 | 7 | 0 | 24 |
| class-changed (55) | 10 | 0 | 17 | 6 | 22 |
| rejected (24) | 1 | 0 | 2 | 0 | 21 |

Acurácia de 5 categorias: **47,7%** (239/501). A coluna `corrected-minor` é
inteiramente zero — com um enum único o modelo nunca escolhe a categoria do meio.

> Correção sobre a versão de 04/09 deste relatório, que trazia 47,3%: a matriz é
> a mesma, a soma da diagonal (205+0+7+6+21 = 239) é que estava errada.

### v3 — single verdict + class correction

| humano \ VLM | kept | corrected-minor | corrected-redrawn | class-changed | rejected |
|---|---:|---:|---:|---:|---:|
| kept (245) | 206 | 0 | 22 | 1 | 16 |
| corrected-minor (84) | 78 | 0 | 4 | 0 | 2 |
| corrected-redrawn (93) | 62 | 0 | 7 | 0 | 24 |
| class-changed (55) | 10 | 0 | 18 | 7 | 20 |
| rejected (24) | 1 | 0 | 2 | 0 | 21 |

Acurácia de 5 categorias: **48,1%** (241/501) — 0,4 pp acima do v1. A coluna
`corrected-minor` continua inteiramente zero.

### v2 — decomposed

| humano \ VLM | kept | corrected-minor | corrected-redrawn | class-changed | rejected |
|---|---:|---:|---:|---:|---:|
| kept (245) | 95 | 146 | 0 | 4 | 0 |
| corrected-minor (84) | 36 | 48 | 0 | 0 | 0 |
| corrected-redrawn (93) | 19 | 52 | 0 | 2 | 20 |
| class-changed (55) | 4 | 6 | 0 | 43 | 2 |
| rejected (24) | 1 | 7 | 0 | 5 | 11 |

Acurácia de 5 categorias: **39,3%** — *menor* que a do v1, embora o v2 ganhe em
todas as métricas da tabela principal. A coluna `corrected-redrawn` é zero por
construção: no v2 o veredito é derivado de `box_quality` por
`verdict_from_v2`. Concordância de veredito caixa a caixa entre v1 e v2:
**191 de 501 (38,1%)**.

### Concordância entre braços — a previsão se confirmou

Artefato: `verdict_agreement_between_arms`.

| par | vereditos idênticos | % |
|---|---:|---:|
| **v1 × v3** | **494 de 501** | **98,6%** |
| v1 × v2 | 191 de 501 | 38,1% |
| v2 × v3 | 192 de 501 | 38,3% |

A previsão registrada em 04/09 era que v1 × v3 ficasse perto de 100%, porque
`verdict` é a primeira propriedade declarada e a gramática do llama.cpp a gera
*antes* de `correct_class` existir. **Deu 98,6%**: as 7 divergências são ruído de
amostragem, não efeito do schema.

Isso fecha a inferência causal do experimento. Trocar o schema **não muda o
veredito**; o v3 é o v1 com o rótulo preenchido. Logo o ganho do v2 **é da
decomposição da pergunta**, não da permissão de corrigir classe.

Matriz v1 × v3 (linha = v1, coluna = v3), só as 7 células fora da diagonal:

| v1 → v3 | n |
|---|---:|
| kept → corrected-redrawn | 2 |
| corrected-redrawn → kept | 2 |
| corrected-redrawn → rejected | 1 |
| rejected → corrected-redrawn | 1 |
| rejected → class-changed | 1 |

## 10. Coordinate adjustment

Artefato: `pillar3-semantic-geometry.json` → `coordinate_adjustment`, com
`per_box` caixa a caixa.

- Braço: **v1** (o único em que o experimento foi rodado).
- Regra de pontuação: rótulo igual ao do gabarito **E** IoU(caixa, gabarito) ≥ 0,70.
  Regra **geométrica**, diferente da contagem por veredito da tabela principal —
  mas reproduz 329/172 no base.
- Fonte da substituição: YOLO11x fresco @ conf 0,25 (`runs/_flow_cache/detections.json`).

| | |
|---|---:|
| Correções pedidas pelo VLM | **53** |
| Geometria substituída | **52** |
| Sem candidato sobreposto | 1 |
| **TP antes** | **312** |
| **TP depois** | **304** |
| Delta | **−8** |
| IoU melhorou | 11 |
| IoU piorou | 39 |
| IoU inalterado | 2 |
| **Caixa boa quebrada (TP→FP)** | **9** |
| **Caixa ruim consertada (FP→TP)** | **1** |
| Continuou TP | 18 |
| Continuou FP | 24 |
| **IoU médio antes** | **0,7963** |
| **IoU médio depois** | **0,6989** |
| Delta IoU médio | −0,0974 |
| Delta IoU mediano | −0,1217 |

`9 − 1 = 8` ✓ — o saldo por transição bate com o delta de TP.

## 11. Confidence-threshold sweep

Artefato: `pillar23-sweep-repro.json`. 20 combinações (2 escopos de classe × 2
IoU × 5 confianças).

Predições: `runs\20260816T220716Z-infer\predictions.json`
Gabarito: `annotations\instances_test.json`
Pesos: `yolo11x.pt`, sha256 `7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24`, 114 636 239 bytes
Params: split test, conf 0,001, iou 0,7, imgsz 640, device 0, batch 8, 313 imagens

**Reproduz sob `class_scope = gt_present`, `IoU = 0,50`** — as 5 confianças batem:

| conf | Detecções | Precisão | Recall | Objetos perdidos |
|---:|---:|---:|---:|---:|
| 0,25 | 969 | 62,0% | 82,2% | 130 |
| 0,50 | 707 | 77,5% | 75,0% | 183 |
| 0,70 | 533 | 88,0% | 64,2% | 262 |
| 0,80 | 390 | 93,8% | 50,1% | 365 |
| 0,90 | 148 | 98,0% | 19,8% | 586 |

O dossiê 6.2 não registrava qual `class_scope` nem qual IoU tinha usado; por isso
as 4 combinações foram testadas. Só `gt_present` @ IoU 0,50 reproduz a linha
inteira. **Esse par precisa ser declarado junto da tabela** — em IoU 0,70 os
mesmos dados dão 45,2% / 59,9% em conf 0,25, e a tabela publicada vira outra.

## 12. Confirmação dos números pedidos

| Número | Status | Fonte |
|---|---|---|
| 1 caixa útil / 11,15 propostas | ✅ **confirmado** | `one_in_eleven`: 1528 propostas / 137 = 11,15 |
| TP 312 → 304 | ✅ **confirmado** | `coordinate_adjustment` |
| IoU médio 0,7963 → 0,6989 | ✅ **confirmado** | `coordinate_adjustment` |
| 402 = 254 + 148 | ✅ **confirmado** | `sum_check.equal = true` |
| 148 = 93 + 55 | ✅ **confirmado** | `had_proposal_but_failed_148` |
| 245 / 84 / 93 / 55 / 24 | ✅ **confirmado** | `verdicts.ruler_501` |
| 395 → 387 | ❌ **não reproduzível** | ver abaixo |

### Detalhe do 1-em-11

| | |
|---|---:|
| Run de origem | `runs\20260820T173145Z-vlm-audit` |
| Propostas de recall (Fase 2) | 1528 |
| Casaram por presença | 172 |
| Casaram por presença **e** classe | **137** |
| Alvos desenhados à mão | 254 |
| Propostas por acerto de presença | 8,88 |
| **Propostas por acerto de presença + classe** | **11,15** |

Regra de presença: IoU ≥ 0,30 (`VLM_RECALL_PRESENCE_IOU`) **ou** centro dentro da
caixa humana.

## 13. O par 395 → 387

**Não existe artefato que produza esse par.** Registrado em
`coordinate_adjustment.published_claim_check` com
`"395_387_found_in_artifacts": false`.

Onde ele aparece:
- `paper/dossie.md:359` — "Piora: 395 → 387"
- `paper/numeros.csv:96-97` — as duas linhas já vêm marcadas
  **"SEM ARTEFATO DE PONTUAÇÃO — conferir antes do texto"**

O par medido é **312 → 304**. Os dois têm delta **−8**, o que é provavelmente por
que o par sobreviveu na prosa: a conclusão qualitativa ("piora") nunca mudou, só
a base. Recomendação: usar 312 → 304, que tem artefato com `per_box`.

## 14. Detalhes do VLM

| | |
|---|---|
| Modelo | `qwen3-vl:8b-instruct` |
| Digest | `0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319` |
| Parâmetros | 8,8 B, quantização Q4_K_M, formato GGUF |
| Servidor | Ollama 0.32.15, `http://localhost:11434`, `/api/chat` |
| Temperatura | **0,0** |
| Decodificação restrita | JSON schema via campo `format` do Ollama (gramática llama.cpp) |
| `OLLAMA_NUM_PARALLEL` | 1 — medido em 19/08/2026: 2 slots davam 0,86× do throughput sequencial |
| Imagens por chamada | 2 — cena inteira com a caixa em rosa + recorte ampliado |
| `VLM_IMAGE_MAX_SIDE` | 1920 |
| `VLM_CROP_PAD` | 0,6 do maior lado da caixa |
| `VLM_CROP_MIN_PX` | 224 |
| Timeout | 300 s |
| Hardware | RTX 5060 Ti 16 GB, 15,8 GB RAM |
| Frames | 313 revisados |
| Caixas | 501 (régua), 505 antes da exclusão de avião e trem |
| Custo do v3 (última sessão) | 50 chamadas, 209,1 s, 4,18 s/caixa, 170 681 prompt tokens, 2 545 eval tokens, 0 falhas |

Determinismo: temperatura 0 com decodificação restrita. Não há seed explícita na
chamada — a repetibilidade vem do greedy decoding, não de uma semente fixada.
Ver §16.

## 15. Prompts

Dump completo e verbatim em **`pillar3-prompts.txt`**, gerado por
`dump_prompts.py`, renderizado sobre a primeira das 501 caixas
(`20240103201734:4:0`, person, conf 0,76).

O arquivo traz, lado a lado:

- **A)** prompt single original (v1) + schema v1
- **B)** prompt decomposed (v2) + schema v2
- **C)** prompt single + class correction (v3) + schema v3
- **D)** prompt e schema da Fase 2 (recall) — contexto, não é braço da tabela
- diff v1 → v3 e diff v1 → v2

### O diff que faz o experimento

**v1 → v3 — texto do prompt idêntico byte a byte** (`phase1_prompt` é reusada, não
copiada). Muda só o schema:

| | v1 | v3 |
|---|---|---|
| `correct_class.enum` | ausente | `[person, bicycle, car, motorcycle, bus, truck]` |
| `required` | `[verdict, reason]` | `[verdict, correct_class, reason]` |

Mesmo enum de 5 vereditos, mesma ordem de propriedades, mesmas imagens, mesmo
modelo, mesma temperatura. **v1 → v3 muda só a capacidade de corrigir classe.**

**v1 → v2**, por contraste, muda três coisas de uma vez: (1) o texto da tarefa é
outro — 5 vereditos num enum único contra três perguntas ordenadas
`object_present → object_class → box_quality`; (2) o schema v2 não tem campo
`verdict`, que é derivado no código por `verdict_from_v2`; (3) `object_class` é
obrigatória e travada nas 6 classes.

É essa confusão de três fatores que o v3 desfaz.

## 16. Problemas de reprodutibilidade

1. **Sem seed explícita no VLM.** Temperatura 0 + gramática dão greedy decoding,
   mas nenhuma semente é passada ao Ollama. Reexecução idêntica é esperada, não
   garantida por construção.

2. **`class_scope` e IoU do sweep não estavam registrados.** O dossiê 6.2 publicou
   a tabela sem dizer qual combinação usou. Recuperado por força bruta: só
   `gt_present` @ IoU 0,50 reproduz. Precisa ser declarado no texto.

3. **395 → 387 sem artefato.** §13.

4. **`usage` do v3 está fragmentado.** O braço foi retomado três vezes, e o
   `usage` gravado no `meta.json` cobre só a última sessão (50 das 501 caixas).
   Registrado no próprio `meta.json` como `usage_note`. O custo total do v3 não
   é recuperável de um artefato só; a taxa medida (~4,2–4,4 s/caixa) é
   consistente entre as três sessões e serve de estimativa: ~37 min para as 501.

5. **Régua 501 vs. 505.** O v1 gravou 505 linhas (antes da exclusão de avião e
   trem) e é pontuado sobre 501, com 4 linhas descartadas. O v2 e o v3 gravam 501.
   Está registrado em `rows_in_phase1_file` / `rows_scored_on_501` de cada braço,
   mas é uma assimetria a mencionar.

6. **Falhas transitórias no checkpoint do v3, todas reinferidas.** Três chamadas
   devolveram "resposta vazia" sob pressão de memória. O resume reinfere linhas
   com erro em vez de herdá-las, e o `phase1.json` final tem **zero erros** — a
   comparação com o v1, que fechou com zero falhas, está preservada.

7. **`reanalysis.py` é v1/v2 por construção** (runs fixos em constantes). Não
   precisa mudar: o que depende dos braços sai de `compare_arms.py` e
   `semantic_geometry.py`, ambos v3-aware por glob.

8. **O diagnóstico de memória de 04/09 estava errado por ~2×.** Ver §0: o número
   publicado de 16,4 GB de commit para o `llama-server` mediu 8,04 GB quando
   remedido. A causa provável do episódio foi VRAM ocupada forçando transbordo
   para CPU, não uma exigência de memória do modelo.

9. **Duas mortes por watchdog externo na 3ª tentativa.** Os dois primeiros
   lançamentos de 07/09 foram encerrados por um watchdog de memória do harness
   que executava o comando, não pelo script nem pelo Windows. Sem efeito nos
   dados — o checkpoint absorveu — mas explica por que o run final foi lançado em
   primeiro plano.

## 17. Comandos

Tudo já foi executado; a lista é para reprodução.

```bash
cd C:\Users\REPRO_USER\Documents\carcara-forge

# quarto braço — inferência (idempotente, retomável). EXECUTADO 07/09, 501/501.
.venv\Scripts\python.exe -u paper\pillar23\single_classfix.py

# agregadores — EXECUTADOS 07/09, já com a linha do v3:
.venv\Scripts\python.exe paper\pillar23\compare_arms.py       # tabela das 4 configurações + car/truck
.venv\Scripts\python.exe paper\pillar23\semantic_geometry.py  # semântica/geometria + matrizes + coordenadas

# fechados antes, não precisam rodar de novo:
.venv\Scripts\python.exe paper\pillar23\reanalysis.py   # decomposição, verdicts, 1-em-11
.venv\Scripts\python.exe paper\pillar23\sweep_repro.py  # sweep de confiança
.venv\Scripts\python.exe paper\pillar23\dump_prompts.py # dump dos prompts
```

Os dois agregadores são só leitura — não tocam run nem banco. O `single_classfix.py`
relançado agora **não reinfere nada**: o checkpoint cobre as 501 e ele sai
avisando `nada a inferir`. Para refazer o braço do zero seria preciso apagar
`runs\_v3_classfix_cache\phase1.jsonl` — não faça isso sem querer perder ~37 min
de GPU.

Pré-requisito: `ollama serve` no ar e **GPU livre** (ver §0).

## 18. Caminhos dos outputs

### Análise — `C:\Users\REPRO_USER\Documents\carcara-forge\paper\pillar23\`

| Arquivo | Conteúdo | Precisa refazer? |
|---|---|---|
| `reanalysis.py` | decomposição, verdicts, denominadores, 1-em-11 | não |
| `pillar23-audit-reanalysis.json` | saída do acima | não |
| `sweep_repro.py` | sweep de confiança | não |
| `pillar23-sweep-repro.json` | saída do acima, 20 combinações | não |
| `dump_prompts.py` | dump verbatim dos prompts | não |
| `pillar3-prompts.txt` | saída do acima, já inclui o v3 | não |
| `semantic_geometry.py` | semântica × geometria, matrizes, coordenadas | não |
| `pillar3-semantic-geometry.json` | saída do acima — **com os 3 braços VLM** | não |
| `single_classfix.py` | **o quarto braço** (com checkpoint) | não — fechado |
| `single_classfix.py.bak-precheckpoint` | versão original, antes do checkpoint | — |
| `compare_arms.py` | tabela das 4 configurações + car/truck | não |
| `pillar3-single-classfix.json` | saída do acima — **4 linhas** | não |
| `pillar23-audit-reanalysis.csv` | mesma tabela em CSV `;` | não |
| `RELATORIO-PILLAR23.md` | este documento | não |
| `RELATORIO-PILLAR23.md.bak-pre-v3` | versão de 04/09, antes do fechamento do v3 | — |

### Runs — `D:\carcara-data\all-perception-in-one\runs\`

| Run | Papel |
|---|---|
| `20260820T031131Z-vlm-audit` | **v1** — single verdict (505 linhas) |
| `20260821T002801Z-vlm-audit` | **v2** — decomposed (501 linhas) |
| `20260820T173145Z-vlm-audit` | Fase 2 / recall — fonte do 1-em-11 |
| `20260816T220716Z-infer` | predições YOLO11x do sweep |
| `_flow_cache` | YOLO11x @ conf 0,25 — fonte das substituições de coordenada |
| `_v3_classfix_cache\phase1.jsonl` | **checkpoint do v3** — 501 caixas, completo |
| `20260907T164834Z-vlm-audit-v3-single-classfix` | **v3** — single + class correction (501 linhas) |
| `20260904T224035Z-vlm-audit-v3-single-classfix` | run vazio da 1ª tentativa (preservado) |

Gabarito: `D:\carcara-data\all-perception-in-one\manifest.json` e
`annotations\instances_test.json`. Coleção `<CARCARA_COLLECTION_ID>`
("All perception in one").

---

## Conclusão

A pergunta que o quarto braço existia para responder: **o ganho do v2 vem de
decompor a pergunta, ou só de deixar o modelo corrigir classe?**

**Vem de decompor a pergunta.** Três evidências independentes, todas em artefato:

1. **Placar.** O v3 rende +5 TP sobre o v1 (312 → 317) e fica 51 TP atrás do v2
   (368). Cerca de 9% do ganho do v2 é atribuível à correção de classe; ~91% à
   decomposição.

2. **Concordância de vereditos.** v1 × v3 = **98,6%** (494/501). Mudar o schema
   não muda o que o modelo decide — só o que ele consegue escrever. Como previsto
   em 04/09, `verdict` é gerado pela gramática antes de `correct_class` existir.

3. **Taxa de tentativa.** Com o schema consertado, o v3 acerta quando tenta
   (87,5% de precisão, a melhor das três) — mas tenta **8 vezes contra 54** do
   v2. O recall sobre os 55 erros de classe do humano vai de 0% (v1) a 12,7%
   (v3) a 78,2% (v2). A capacidade estava travada por duas coisas, e o bug de
   schema era a menor.

Corolário para o método: **o formato da pergunta importa mais que a expressividade
do schema.** Um enum único de cinco vereditos faz o modelo colapsar em `kept` e
`rejected` — a categoria do meio (`corrected-minor`) recebe **zero** escolhas em
v1 e em v3 — enquanto três perguntas ordenadas (`object_present` →
`object_class` → `box_quality`) o fazem exercer as categorias intermediárias.

Ressalva que continua valendo: nenhum braço de Fase 1 devolve coordenada, então
`corrected-redrawn` nunca vira TP em configuração nenhuma, e o julgamento
geométrico do VLM não move a tabela principal. A tentativa de fazer o VLM corrigir
coordenada foi medida em separado (§10) e **piorou** o resultado: TP 312 → 304,
IoU médio 0,796 → 0,699.
