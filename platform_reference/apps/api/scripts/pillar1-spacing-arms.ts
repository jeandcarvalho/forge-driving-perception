/**
 * Pillar 1 — decomposicao do efeito: Naive Random / Random+spacing / Forge.
 *
 * SOMENTE LEITURA no Mongo. Escreve so os JSON/CSV de resultado.
 *
 * Duas logicas sao portadas LITERALMENTE e mantidas separadas de proposito:
 *
 *  (1) REGRA DE ESPACAMENTO DA PLATAFORMA — identica em add-by-query.ts,
 *      collections/spacing.ts e dataset-sample.ts (dedup_m):
 *      greedy temporal, por aquisicao, contra o ULTIMO frame aceito, reset entre
 *      aquisicoes, GPS ausente passa de graca sem atualizar o ultimo aceito,
 *      haversine de src/lib/geo.ts com R = 6_371_000.
 *
 *  (2) METRICA DO EXPERIMENTO ORIGINAL — porte de paper/pillar1_naive.py
 *      (nn_within_acq): distancia ao vizinho mais proximo DENTRO da mesma
 *      aquisicao, haversine com R = 6371008.8, e a fracao de FRAMES (nao de
 *      pares) abaixo do limiar. Percentis com interpolacao linear (np.percentile).
 *
 *  Os dois raios de Terra sao diferentes no original. Nao unificar: mudaria o
 *  numero da regra ou o da metrica, e a comparacao com os JSON antigos morreria.
 *
 * Amostragem: mulberry32 + Fisher-Yates, copia literal de dataset-sample.ts,
 * sobre a mesma ordem canonica (acq_id, sec).
 */
import { writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import prisma from '../src/lib/prisma'
import { haversineMeters } from '../src/lib/geo'
import { normalizeToCompact } from '../src/controllers/collections/get-collection-seconds-with-links'

const COLLECTION_ID = '<CARCARA_COLLECTION_ID>'
const N = 4364
const SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
const SPACINGS = [0, 30, 100]
const OUT_DIR = join(__dirname, 'out')

// ── (2) haversine da METRICA — R do pillar1_naive.py, nao o da plataforma ──
const R_METRIC = 6371008.8
function haversineMetric(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const toRad = (d: number) => (d * Math.PI) / 180
  const la1 = toRad(lat1), la2 = toRad(lat2)
  const dLat = la2 - la1
  const dLon = toRad(lon2) - toRad(lon1)
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(la1) * Math.cos(la2) * Math.sin(dLon / 2) ** 2
  return 2 * R_METRIC * Math.asin(Math.sqrt(Math.min(1, Math.max(0, h))))
}

// ── PRNG: copia literal de dataset-sample.ts ──
function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
function seededShuffle<T>(arr: T[], seed: number): T[] {
  const rnd = mulberry32(seed)
  const out = [...arr]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1))
    const tmp = out[i]
    out[i] = out[j]
    out[j] = tmp
  }
  return out
}

type Frame = { acq_id: number; sec: number; lat: number | null; lon: number | null }

// ── np.percentile com interpolacao linear ──
function percentile(sorted: number[], q: number): number {
  if (sorted.length === 0) return NaN
  const idx = (q / 100) * (sorted.length - 1)
  const lo = Math.floor(idx), hi = Math.ceil(idx)
  if (lo === hi) return sorted[lo]
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo)
}

// ── (2) metrica: vizinho mais proximo intra-aquisicao ──
function nnWithinAcq(frames: Frame[]): number[] {
  const byAcq = new Map<number, { lat: number; lon: number }[]>()
  for (const f of frames) {
    if (f.lat == null || f.lon == null) continue
    const l = byAcq.get(f.acq_id)
    if (l) l.push({ lat: f.lat, lon: f.lon })
    else byAcq.set(f.acq_id, [{ lat: f.lat, lon: f.lon }])
  }
  const nn: number[] = []
  for (const pts of byAcq.values()) {
    if (pts.length < 2) continue // igual ao numpy: acq com 1 ponto nao entra
    for (let i = 0; i < pts.length; i++) {
      let m = Infinity
      for (let j = 0; j < pts.length; j++) {
        if (i === j) continue
        const d = haversineMetric(pts[i].lat, pts[i].lon, pts[j].lat, pts[j].lon)
        if (d < m) m = d
      }
      nn.push(m)
    }
  }
  return nn
}

type Row = {
  arm: string; variant: string; spacing_m: number; seed: number | null
  n_final: number; n_with_gps: number; n_nn: number; acqs: number
  pool_after_spacing: number; sampling_fraction_pct: number
  lt10_n: number; lt10_pct: number; lt25_n: number; lt25_pct: number
  nn_min: number; nn_p05: number; nn_median: number
}

function measure(
  arm: string, variant: string, spacing: number, seed: number | null,
  frames: Frame[], pool: number,
): Row {
  const nn = nnWithinAcq(frames)
  const sorted = [...nn].sort((a, b) => a - b)
  const lt10 = nn.filter((d) => d < 10).length
  const lt25 = nn.filter((d) => d < 25).length
  const withGps = frames.filter((f) => f.lat != null && f.lon != null).length
  return {
    arm, variant, spacing_m: spacing, seed,
    n_final: frames.length, n_with_gps: withGps, n_nn: nn.length,
    acqs: new Set(frames.map((f) => f.acq_id)).size,
    pool_after_spacing: pool,
    sampling_fraction_pct: (100 * frames.length) / pool,
    lt10_n: lt10, lt10_pct: (100 * lt10) / nn.length,
    lt25_n: lt25, lt25_pct: (100 * lt25) / nn.length,
    nn_min: sorted[0], nn_p05: percentile(sorted, 5), nn_median: percentile(sorted, 50),
  }
}

// ── (1) regra da plataforma: greedy temporal contra o ULTIMO aceito, por acq ──
function platformSpacing(pool: Frame[], d: number): Frame[] {
  if (d <= 0) return pool
  const kept: Frame[] = []
  const lastByAcq = new Map<number, { lat: number; lon: number }>()
  for (const c of pool) {
    if (c.lat == null || c.lon == null) { kept.push(c); continue } // passa de graca
    const prev = lastByAcq.get(c.acq_id)
    if (!prev || haversineMeters(prev.lat, prev.lon, c.lat, c.lon) >= d) {
      kept.push(c)
      lastByAcq.set(c.acq_id, { lat: c.lat, lon: c.lon })
    }
  }
  return kept
}

// ── F2 (robustez): ordem aleatoria, contra TODOS os aceitos da mesma acq ──
function randomOrderAgainstAll(poolGps: Frame[], d: number, seed: number) {
  const sh = seededShuffle(poolGps, seed)
  const accepted = new Map<number, { lat: number; lon: number }[]>()
  const all: Frame[] = []
  for (const c of sh) {
    const list = accepted.get(c.acq_id)
    let ok = true
    if (list) {
      for (const p of list) {
        if (haversineMeters(p.lat, p.lon, c.lat as number, c.lon as number) < d) { ok = false; break }
      }
    }
    if (!ok) continue
    const pt = { lat: c.lat as number, lon: c.lon as number }
    if (list) list.push(pt)
    else accepted.set(c.acq_id, [pt])
    all.push(c)
  }
  return { maxReachable: all.length, firstN: all.slice(0, N) }
}

function agg(rows: Row[], key: keyof Row) {
  const v = rows.map((r) => Number(r[key]))
  const mean = v.reduce((a, b) => a + b, 0) / v.length
  const sd = v.length > 1
    ? Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length - 1))
    : 0
  return { mean, sd, min: Math.min(...v), max: Math.max(...v) }
}

async function main() {
  console.log('carregando acervo...')
  const raw = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $match: { 'links.0.link': { $exists: true } } },
      { $sort: { acq_id: 1, sec: 1 } },
      { $project: { _id: 0, acq_id: 1, sec: 1, lat: '$can.GPS_y', lon: '$can.GPS_x' } },
    ],
  })) as unknown as Frame[]
  const pool = raw.filter((c) => c.acq_id != null)
  console.log(`pool bruto (mode=random, exclude_mode=none) = ${pool.length} frames, ${new Set(pool.map((f) => f.acq_id)).size} acqs`)

  const rows: Row[] = []

  // ── braco 3: FORGE (a colecao curada, deterministica) ──
  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId: COLLECTION_ID }, select: { acq_id: true, secs: true },
  })
  const wanted = new Set<string>()
  for (const g of groups) {
    const a = Number(normalizeToCompact(g.acq_id))
    if (!isNaN(a)) for (const s of g.secs) wanted.add(`${a}:${s}`)
  }
  const forge = pool.filter((f) => wanted.has(`${f.acq_id}:${f.sec}`))
  console.log(`colecao Forge casada no acervo = ${forge.length} de ${wanted.size} chaves`)
  rows.push(measure('forge', 'collection', 30, null, forge, forge.length))

  // ── bracos 1 e 2: F1 (regra da plataforma) para cada spacing ──
  for (const d of SPACINGS) {
    const sp = platformSpacing(pool, d)
    const label = d === 0 ? 'naive_random' : `random_${d}m`
    console.log(`\n[F1] spacing=${d} m -> pool = ${sp.length}`)
    for (const seed of SEEDS) {
      const sampled = seededShuffle(sp, seed).slice(0, N)
      sampled.sort((a, b) => (a.acq_id - b.acq_id) || (a.sec - b.sec))
      const r = measure(label, 'F1_platform_last_accepted', d, seed, sampled, sp.length)
      rows.push(r)
      console.log(`   seed ${seed}: n=${r.n_final} acqs=${r.acqs} <10m=${r.lt10_n} (${r.lt10_pct.toFixed(2)}%) <25m=${r.lt25_n} (${r.lt25_pct.toFixed(2)}%)`)
    }
  }

  // ── robustez: F2 (ordem aleatoria, contra todos) ──
  const poolGps = pool.filter((f) => f.lat != null && f.lon != null)
  for (const d of [30, 100]) {
    console.log(`\n[F2] spacing=${d} m (ordem aleatoria, contra todos)`)
    for (const seed of SEEDS) {
      const { maxReachable, firstN } = randomOrderAgainstAll(poolGps, d, seed)
      const sampled = [...firstN].sort((a, b) => (a.acq_id - b.acq_id) || (a.sec - b.sec))
      const r = measure(`random_${d}m`, 'F2_random_order_against_all', d, seed, sampled, maxReachable)
      rows.push(r)
      console.log(`   seed ${seed}: n=${r.n_final} teto=${maxReachable} acqs=${r.acqs} <10m=${r.lt10_n} (${r.lt10_pct.toFixed(2)}%) <25m=${r.lt25_n} (${r.lt25_pct.toFixed(2)}%)`)
    }
  }

  // ── agregados ──
  const summary: any[] = []
  const keyOf = (r: Row) => `${r.arm}|${r.variant}`
  for (const k of [...new Set(rows.filter((r) => r.seed != null).map(keyOf))]) {
    const sub = rows.filter((r) => keyOf(r) === k)
    const s: any = {
      arm: sub[0].arm, variant: sub[0].variant, spacing_m: sub[0].spacing_m,
      seeds: sub.length, pool_after_spacing: sub[0].pool_after_spacing,
    }
    for (const m of ['n_final', 'n_nn', 'acqs', 'sampling_fraction_pct', 'lt10_n', 'lt10_pct',
      'lt25_n', 'lt25_pct', 'nn_min', 'nn_p05', 'nn_median', 'pool_after_spacing'] as (keyof Row)[]) {
      s[m] = agg(sub, m)
    }
    summary.push(s)
  }

  mkdirSync(OUT_DIR, { recursive: true })
  const jsonPath = join(OUT_DIR, 'pillar1-spacing-arms.json')
  const csvPath = join(OUT_DIR, 'pillar1-spacing-arms.csv')
  writeFileSync(jsonPath, JSON.stringify({
    generated_at: new Date().toISOString(),
    n_target: N, seeds: SEEDS, spacings: SPACINGS,
    pool_raw: pool.length,
    collection_id: COLLECTION_ID,
    metric: 'nearest neighbour within acq_id; fraction of FRAMES below threshold; haversine R=6371008.8',
    spacing_rule: 'greedy temporal per acq vs LAST accepted; missing GPS kept free; haversine R=6371000 (platform)',
    rows, summary,
  }, null, 2), 'utf-8')
  const cols = Object.keys(rows[0]) as (keyof Row)[]
  writeFileSync(csvPath, [cols.join(','), ...rows.map((r) => cols.map((c) => r[c]).join(','))].join('\n'), 'utf-8')
  console.log(`\nJSON -> ${jsonPath}`)
  console.log(`CSV  -> ${csvPath}`)
}

main().finally(() => prisma.$disconnect())
