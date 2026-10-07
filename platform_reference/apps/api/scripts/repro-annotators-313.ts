/** Somente leitura. Proveniencia restrita aos 313 frames da colecao-alvo. Emails anonimizados. */
import prisma from '../src/lib/prisma'
import { createHash } from 'crypto'

const COLLECTION_ID = '<CARCARA_COLLECTION_ID>'
const alias = new Map<string, string>()
const anon = (e?: string | null) => {
  if (!e) return 'Reviewer ?'
  if (!alias.has(e)) alias.set(e, `Reviewer ${String.fromCharCode(65 + alias.size)}`)
  return alias.get(e)!
}

async function main() {
  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId: COLLECTION_ID },
    select: { acq_id: true, curationSecs: true },
  })
  const target = new Set<string>()
  const or: any[] = []
  for (const g of groups) {
    if (!g.curationSecs.length) continue
    const n = Number(g.acq_id.replace(/\D/g, ''))
    or.push({ acq_id: n, sec: { in: g.curationSecs } })
    for (const s of g.curationSecs) target.add(`${n}:${s}`)
  }
  console.log('frames do conjunto travado:', target.size, '| acqs:', or.length)

  const docs = await prisma.big1Hz.findMany({ where: { OR: or }, select: { acq_id: true, sec: true, yolo: true } })
  console.log('docs encontrados:', docs.length)

  const byPerson: Record<string, any> = {}
  let human = 0, model = 0, multi = 0
  const refCount = new Map<string, number>()
  const byDay: Record<string, number> = {}
  const cidSeen = new Set<string>()

  for (const d of docs) {
    for (const det of ((d.yolo ?? []) as any[])) {
      if (det.action == null) { model++; continue }
      human++
      const p = anon(det.reviewed_by)
      byPerson[p] ??= { total: 0, actions: {} as any, first: null, last: null, frames: new Set<string>() }
      const b = byPerson[p]
      b.total++
      b.actions[det.action] = (b.actions[det.action] ?? 0) + 1
      b.frames.add(`${d.acq_id}:${d.sec}`)
      const at = det.reviewed_at ? new Date(det.reviewed_at).toISOString() : null
      if (at) {
        if (!b.first || at < b.first) b.first = at
        if (!b.last || at > b.last) b.last = at
        byDay[at.slice(0, 10)] = (byDay[at.slice(0, 10)] ?? 0) + 1
      }
      cidSeen.add(det.collection_id ?? '(sem collection_id)')
      if (det.ref_det_id) {
        const k = `${d.acq_id}:${d.sec}:${det.ref_det_id}`
        refCount.set(k, (refCount.get(k) ?? 0) + 1)
      }
    }
  }
  multi = [...refCount.values()].filter(n => n > 1).length

  console.log('\ndeteccoes originais do modelo nesses frames:', model)
  console.log('entradas humanas nesses frames:', human)
  for (const [p, b] of Object.entries<any>(byPerson)) {
    console.log(JSON.stringify({ reviewer: p, total: b.total, actions: b.actions,
      frames: b.frames.size, first_at: b.first, last_at: b.last }, null, 1))
  }
  console.log('identidades distintas:', alias.size)
  console.log('sha256(email)[0:12]:', [...alias.entries()].map(([e, a]) => `${a}=${createHash('sha256').update(e).digest('hex').slice(0, 12)}`))
  console.log('collection_id vistos nas entradas humanas:', [...cidSeen])
  console.log('originais com >1 entrada humana (2a revisao):', multi)
  console.log('entradas por dia:', JSON.stringify(byDay, null, 1))
}

main().catch(e => { console.error(e); process.exit(1) }).finally(() => prisma.$disconnect())
