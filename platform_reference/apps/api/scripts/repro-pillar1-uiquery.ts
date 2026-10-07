/** Reproduz a query da UI do Pillar 1 e compara com a coleção real. SOMENTE LEITURA. */
import prisma from '../src/lib/prisma'
import { buildMongoMatch } from '../src/lib/build-mongo-match'
import type { SearchQuery } from '@carcara/shared'

const COL = '<CARCARA_COLLECTION_ID>'
const SIX = 'car,truck,bus,motorcycle,bicycle,person'

function hav(lat1: number, lon1: number, lat2: number, lon2: number) {
  const R = 6_371_000, toRad = (d: number) => (d * Math.PI) / 180
  const dLat = toRad(lat2 - lat1), dLon = toRad(lon2 - lon1)
  const a = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

/** greedy idêntico ao de add-by-query.ts */
function spaced(rows: { sec: number; gx: number | null; gy: number | null }[], d: number) {
  rows.sort((a, b) => a.sec - b.sec)
  const kept: number[] = []
  let last: { lat: number; lon: number } | null = null
  for (const r of rows) {
    if (r.gx == null || r.gy == null) { kept.push(r.sec); continue }
    if (!last) { kept.push(r.sec); last = { lat: r.gy, lon: r.gx }; continue }
    if (hav(last.lat, last.lon, r.gy, r.gx) >= d) { kept.push(r.sec); last = { lat: r.gy, lon: r.gx } }
  }
  return kept
}

async function run(label: string, q: SearchQuery, d: number | null) {
  const match = buildMongoMatch(q)
  const raw = (await prisma.big1Hz.aggregateRaw({
    pipeline: [{ $match: match as any },
               { $project: { _id: 0, acq_id: 1, sec: 1, 'can.GPS_x': 1, 'can.GPS_y': 1 } }],
  })) as unknown as any[]
  const byAcq = new Map<string, { sec: number; gx: number | null; gy: number | null }[]>()
  for (const doc of raw) {
    if (doc.acq_id == null || doc.sec == null) continue
    const k = String(doc.acq_id)
    if (!byAcq.has(k)) byAcq.set(k, [])
    byAcq.get(k)!.push({ sec: Number(doc.sec), gx: doc.can?.GPS_x ?? null, gy: doc.can?.GPS_y ?? null })
  }
  const out = new Set<string>()
  for (const [acq, rows] of byAcq) {
    const secs = d == null ? rows.map(r => r.sec) : spaced(rows, d)
    for (const s of secs) out.add(`${acq}:${s}`)
  }
  const acqs = new Set([...out].map(k => k.split(':')[0]))
  console.log(`${label.padEnd(52)} pool=${String(raw.length).padStart(6)}  apos_spacing=${String(out.size).padStart(5)}  acqs=${String(acqs.size).padStart(4)}`)
  return out
}

async function main() {
  const groups = await prisma.collectionGroup.findMany({ where: { collectionId: COL }, select: { acq_id: true, secs: true } })
  const real = new Set<string>()
  for (const g of groups) for (const s of g.secs) real.add(`${g.acq_id}:${s}`)
  console.log(`COLECAO REAL: ${real.size} frames, ${groups.length} acquisitions\n`)

  const uiQuery = { 'y.class': SIX, 'y.conf': '0.66..', 'y.volume': 'low,mid,high' } as unknown as SearchQuery
  const got = await run('UI: 6 classes + conf>=0.66 + vol(all) + 30m', uiQuery, 30)
  await run('  variante: sem y.volume', { 'y.class': SIX, 'y.conf': '0.66..' } as unknown as SearchQuery, 30)
  await run('  variante: sem y.conf', { 'y.class': SIX, 'y.volume': 'low,mid,high' } as unknown as SearchQuery, 30)

  const missing = [...real].filter(k => !got.has(k))
  const extra = [...got].filter(k => !real.has(k))
  console.log('\n=== COMPARACAO COM A COLECAO REAL ===')
  console.log('na colecao e NAO na query :', missing.length, missing.slice(0, 12))
  console.log('na query e NAO na colecao :', extra.length, extra.slice(0, 12))
  console.log('interseccao               :', [...real].filter(k => got.has(k)).length)
  const missAcq = new Set(missing.map(k => k.split(':')[0]))
  console.log('acquisitions so na colecao:', [...missAcq])
}
main().catch(e => { console.error(e); process.exit(1) }).finally(() => prisma.$disconnect())
