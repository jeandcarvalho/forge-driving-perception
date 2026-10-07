import { minDistanceToSet, LatLon } from '../../../lib/geo'
import { isHuman, RawDet } from './shared'

// Núcleo PURO da seleção do conjunto de teste (A1). Modelo de POOL INTEIRO:
// a coleção toda é candidata; o teste é escolhido só por espaçamento intra (diversidade
// geográfica + contagem ~300). O antivazamento é garantido a jusante por EXCLUSIVIDADE
// DE AQUISIÇÃO (toda acq que doa ≥1 frame de teste sai inteira do treino) — não há
// separação inter/buffer aqui. Sem prisma/fastify → testável isoladamente.

export const TARGET = 300
export const LO_BAND = 280
export const HI_BAND = 320
export const FLOOR_M = 10 // distância mínima do espaçamento intra
export const HI_CAP_M = 200_000 // teto de segurança da expansão

export type Frame = { acqC: string; sec: number }
export type Candidate = { acqC: string; sec: number; gps: LatLon }

type Can = { GPS_x?: number | null; GPS_y?: number | null } | null | undefined
export type PoolDoc = { acq_id: number | null; sec: number; can?: Can; yolo?: RawDet[] | null }

export function gpsOf(can: Can): LatLon | null {
  if (!can || can.GPS_x == null || can.GPS_y == null) return null
  return { lat: can.GPS_y, lon: can.GPS_x }
}

// GPS por frame + cobertura yolo por acq, sobre o pool inteiro da coleção.
export function extractGpsAndCoverage(poolDocs: PoolDoc[]) {
  const gpsBySecKey = new Map<string, LatLon>() // `${acqC}:${sec}`
  const acqCWithYolo = new Set<string>()
  for (const d of poolDocs) {
    if (d.acq_id == null) continue
    const acqC = String(d.acq_id)
    const gps = gpsOf(d.can)
    if (gps) gpsBySecKey.set(`${acqC}:${d.sec}`, gps)
    if (d.yolo && d.yolo.some((det) => !isHuman(det))) acqCWithYolo.add(acqC)
  }
  return { gpsBySecKey, acqCWithYolo }
}

// Candidatos: frame com GPS válido e acq com cobertura yolo. Ordenação determinística
// (acqC, sec) → conjunto reprodutível.
export function buildCandidates(
  poolFrames: Frame[],
  gpsBySecKey: Map<string, LatLon>,
  acqCWithYolo: Set<string>,
): Candidate[] {
  const candidates: Candidate[] = []
  for (const f of poolFrames) {
    if (!acqCWithYolo.has(f.acqC)) continue
    const gps = gpsBySecKey.get(`${f.acqC}:${f.sec}`)
    if (!gps) continue
    candidates.push({ acqC: f.acqC, sec: f.sec, gps })
  }
  candidates.sort((a, b) => (a.acqC < b.acqC ? -1 : a.acqC > b.acqC ? 1 : a.sec - b.sec))
  return candidates
}

// Espaçamento intra: greedy mantém o frame se estiver ≥ d de todos os já selecionados.
// f(d) é monotônica não-crescente em d → busca binária válida.
export function selectAt(candidates: Candidate[], d: number): Candidate[] {
  const selected: Candidate[] = []
  const selGps: LatLon[] = []
  for (const c of candidates) {
    if (minDistanceToSet(c.gps, selGps) >= d) {
      selected.push(c)
      selGps.push(c.gps)
    }
  }
  return selected
}

export type SelectResult = {
  selected: Candidate[]
  minDistanceM: number
  underTarget: boolean
}

// Busca binária mirando [280,320], preferindo o mais próximo de 300.
export function selectCurationSet(candidates: Candidate[]): SelectResult {
  const floorSel = selectAt(candidates, FLOOR_M)
  let best = floorSel
  let bestD = FLOOR_M

  if (floorSel.length < LO_BAND) {
    // Nem no piso de 10 m alcança 280 → grava o que sobrevive no piso.
    return { selected: floorSel, minDistanceM: FLOOR_M, underTarget: true }
  }

  // Expande hi até o count cair até/abaixo da banda superior (ou bater o teto).
  let hi = 50
  while (selectAt(candidates, hi).length > HI_BAND && hi < HI_CAP_M) hi *= 2

  let lo = FLOOR_M
  for (let i = 0; i < 40; i++) {
    const mid = (lo + hi) / 2
    const sel = selectAt(candidates, mid)
    if (Math.abs(sel.length - TARGET) < Math.abs(best.length - TARGET)) {
      best = sel
      bestD = mid
    }
    if (sel.length > TARGET) lo = mid // precisa de d maior para reduzir
    else hi = mid // precisa de d menor para aumentar
  }

  return { selected: best, minDistanceM: bestD, underTarget: best.length < LO_BAND }
}
