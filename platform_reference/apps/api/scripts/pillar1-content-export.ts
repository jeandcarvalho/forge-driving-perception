/**
 * Exporta as listas de frames dos bracos do Pillar 1 (metricas de CONTEUDO).
 *
 * SOMENTE LEITURA no Mongo. Escreve so a lista de frames, para o lado Python
 * (download + YOLO11x) consumir. Nao toca em nada antigo.
 *
 * Bracos exportados:
 *   forge          colecao curada, deterministica
 *   naive_random   spacing 0 m,  seeds 42-51
 *   random_30m     spacing 30 m, seeds 42-51, variante F1 (regra da plataforma)
 *
 * As seeds 42-44 sao as que a Table 3 antiga usou; 45-51 estendem para as mesmas
 * 10 do experimento espacial. Escreve num arquivo NOVO — o de 3 seeds fica.
 *
 * A logica de spacing e de amostragem e copia literal de dataset-sample.ts,
 * identica a usada em pillar1-spacing-arms.ts.
 */
import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import prisma from '../src/lib/prisma'
import { haversineMeters } from '../src/lib/geo'
import { normalizeToCompact } from '../src/controllers/collections/get-collection-seconds-with-links'
import { extractDriveId } from '../src/controllers/collections/export-manifest'

const COLLECTION_ID = '<CARCARA_COLLECTION_ID>'
const N = 4364
const SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
const OUT_DIR = join(__dirname, 'out')
const OLD_ARMS = join(process.env.FORGE_DATA_ROOT ?? 'data', 'all-perception-in-one', 'arms')

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

type Frame = { acq_id: number; sec: number; lat: number | null; lon: number | null; link: string | null }

// regra da plataforma: greedy temporal, por acq, contra o ULTIMO aceito
function platformSpacing(pool: Frame[], d: number): Frame[] {
  if (d <= 0) return pool
  const kept: Frame[] = []
  const lastByAcq = new Map<number, { lat: number; lon: number }>()
  for (const c of pool) {
    if (c.lat == null || c.lon == null) { kept.push(c); continue }
    const prev = lastByAcq.get(c.acq_id)
    if (!prev || haversineMeters(prev.lat, prev.lon, c.lat, c.lon) >= d) {
      kept.push(c)
      lastByAcq.set(c.acq_id, { lat: c.lat, lon: c.lon })
    }
  }
  return kept
}

const out = (f: Frame) => ({
  acq_id: String(f.acq_id),
  sec: f.sec,
  drive_id: f.link ? extractDriveId(f.link) : null,
})

async function main() {
  const raw = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $match: { 'links.0.link': { $exists: true } } },
      { $sort: { acq_id: 1, sec: 1 } },
      {
        $project: {
          _id: 0, acq_id: 1, sec: 1, lat: '$can.GPS_y', lon: '$can.GPS_x',
          link: { $arrayElemAt: ['$links.link', 0] },
        },
      },
    ],
  })) as unknown as Frame[]
  const pool = raw.filter((c) => c.acq_id != null)
  console.log(`pool = ${pool.length} frames`)

  const arms: any[] = []

  // forge
  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId: COLLECTION_ID }, select: { acq_id: true, secs: true },
  })
  const wanted = new Set<string>()
  for (const g of groups) {
    const a = Number(normalizeToCompact(g.acq_id))
    if (!isNaN(a)) for (const s of g.secs) wanted.add(`${a}:${s}`)
  }
  const forge = pool.filter((f) => wanted.has(`${f.acq_id}:${f.sec}`))
  arms.push({ arm: 'forge', seed: null, spacing_m: 30, n: forge.length, frames: forge.map(out) })
  console.log(`forge: ${forge.length}`)

  // naive_random e random_30m
  for (const d of [0, 30]) {
    const sp = platformSpacing(pool, d)
    const label = d === 0 ? 'naive_random' : 'random_30m'
    for (const seed of SEEDS) {
      const sampled = seededShuffle(sp, seed).slice(0, N)
      sampled.sort((a, b) => (a.acq_id - b.acq_id) || (a.sec - b.sec))
      arms.push({ arm: label, seed, spacing_m: d, n: sampled.length,
                  pool_after_spacing: sp.length, frames: sampled.map(out) })
      console.log(`${label} s${seed}: ${sampled.length} (pool ${sp.length})`)
    }
  }

  // ── conferencia de identidade: o naive_random regenerado bate com o braco
  //    antigo em disco, frame a frame? ──
  console.log('\nconferencia frame-a-frame contra arms/random-n4364-s{seed}-full.json:')
  for (const seed of SEEDS) {
    const p = join(OLD_ARMS, `random-n4364-s${seed}-full.json`)
    if (!existsSync(p)) { console.log(`  s${seed}: arquivo antigo ausente`); continue }
    const old = JSON.parse(readFileSync(p, 'utf-8')).frames as { acq_id: string; sec: number }[]
    const oldKeys = new Set(old.map((f) => `${f.acq_id}:${f.sec}`))
    const mine = arms.find((a) => a.arm === 'naive_random' && a.seed === seed)!.frames
    const myKeys = new Set(mine.map((f: any) => `${f.acq_id}:${f.sec}`))
    const inter = [...myKeys].filter((k) => oldKeys.has(k as string)).length
    console.log(`  s${seed}: antigo=${old.length} novo=${mine.length} intersecao=${inter} ` +
                `${inter === old.length && old.length === mine.length ? 'IDENTICO' : 'DIVERGE'}`)
  }

  // uniao de imagens necessarias
  const union = new Set<string>()
  for (const a of arms) for (const f of a.frames) union.add(`${f.acq_id}/${f.sec}`)
  console.log(`\nuniao de frames distintos entre todos os bracos: ${union.size}`)

  mkdirSync(OUT_DIR, { recursive: true })
  const p = join(OUT_DIR, 'pillar1-content-framelists-s42-51.json')
  writeFileSync(p, JSON.stringify({
    generated_at: new Date().toISOString(),
    n_target: N, seeds: SEEDS, collection_id: COLLECTION_ID,
    union_frames: union.size, arms,
  }), 'utf-8')
  console.log(`-> ${p}`)
}

main().finally(() => prisma.$disconnect())
