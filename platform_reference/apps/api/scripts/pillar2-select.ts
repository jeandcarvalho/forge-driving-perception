/**
 * Selecao deterministica dos 63 frames (~20%) do conjunto travado de 313 para a
 * SEGUNDA anotacao independente (round 2).
 *
 * Nucleo PURO — sem prisma/fastify. Reusado pelo script de criacao e pela auditoria.
 *
 * Criterio: espalhar ao maximo entre aquisicoes, sem escolha manual e sem olhar
 * NENHUMA anotacao humana (nem do round 1). A unica informacao de conteudo usada e a
 * classe das PROPOSALS YOLO, e apenas para garantir presenca de car/truck.
 */
import { createHash } from 'crypto'

export const SEED = 'carcara-pillar2-second-annotator-v1'
export const TARGET_FRAMES = 63
export const REQUIRED_CLASSES = ['car', 'truck'] as const

export type Frame = { acqC: string; sec: number; classes: string[] }

// Hash estavel → ordem pseudo-aleatoria reprodutivel em qualquer maquina/versao de node.
// (Array.sort nao e usado como fonte de aleatoriedade; a chave e o hash.)
export function h(...parts: (string | number)[]): string {
  return createHash('sha256').update([SEED, ...parts].join(':')).digest('hex')
}

export type SelectionResult = {
  selected: Frame[]
  acqCount: number
  repairs: Array<{ cls: string; added: string; removed: string }>
}

/**
 * 1. Agrupa os 313 frames por aquisicao.
 * 2. Ordena as aquisicoes por h(acq) e, dentro de cada uma, os frames por h(acq, sec).
 * 3. Round-robin: uma passada pega 1 frame de cada aquisicao, depois a 2a, etc.
 *    Com 127 aquisicoes >= 63 alvo, a 1a passada basta -> 63 aquisicoes distintas.
 * 4. Reparo deterministico: se car/truck faltar entre as proposals selecionadas,
 *    troca o ULTIMO frame escolhido pelo primeiro frame (na mesma ordem canonica)
 *    que contenha a classe ausente, sem repetir aquisicao ja usada.
 */
export function selectSecondRound(all: Frame[], target = TARGET_FRAMES): SelectionResult {
  const byAcq = new Map<string, Frame[]>()
  for (const f of all) {
    if (!byAcq.has(f.acqC)) byAcq.set(f.acqC, [])
    byAcq.get(f.acqC)!.push(f)
  }
  const acqs = [...byAcq.keys()].sort((a, b) => (h(a) < h(b) ? -1 : h(a) > h(b) ? 1 : a < b ? -1 : 1))
  for (const a of acqs) {
    byAcq.get(a)!.sort((x, y) => {
      const hx = h(x.acqC, x.sec), hy = h(y.acqC, y.sec)
      return hx < hy ? -1 : hx > hy ? 1 : x.sec - y.sec
    })
  }

  const selected: Frame[] = []
  const maxDepth = Math.max(...acqs.map((a) => byAcq.get(a)!.length))
  for (let depth = 0; depth < maxDepth && selected.length < target; depth++) {
    for (const a of acqs) {
      if (selected.length >= target) break
      const f = byAcq.get(a)![depth]
      if (f) selected.push(f)
    }
  }

  // Ordem canonica global (a mesma usada para o reparo).
  const canonical = acqs.flatMap((a) => byAcq.get(a)!)
  const repairs: SelectionResult['repairs'] = []
  const key = (f: Frame) => `${f.acqC}:${f.sec}`

  for (const cls of REQUIRED_CLASSES) {
    if (selected.some((f) => f.classes.includes(cls))) continue
    const usedAcqs = new Set(selected.map((f) => f.acqC))
    const cand = canonical.find((f) => f.classes.includes(cls) && !usedAcqs.has(f.acqC))
      ?? canonical.find((f) => f.classes.includes(cls) && !selected.some((s) => key(s) === key(f)))
    if (!cand) continue
    const dropped = selected.pop()!
    selected.push(cand)
    repairs.push({ cls, added: key(cand), removed: key(dropped) })
  }

  selected.sort((a, b) => (a.acqC < b.acqC ? -1 : a.acqC > b.acqC ? 1 : a.sec - b.sec))
  return { selected, acqCount: new Set(selected.map((f) => f.acqC)).size, repairs }
}
