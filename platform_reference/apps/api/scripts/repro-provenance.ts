/** Somente leitura. Proveniência da curadoria + estado da coleção do Pillar 1.
 *  Não escreve nada. Emails são anonimizados na saída (Reviewer A/B/...).
 */
import prisma from '../src/lib/prisma'
import { createHash } from 'crypto'

const COLLECTION_ID = '<CARCARA_COLLECTION_ID>'

const alias = new Map<string, string>()
function anon(email: string | null | undefined): string {
  if (!email) return 'Reviewer ?'
  if (!alias.has(email)) alias.set(email, `Reviewer ${String.fromCharCode(65 + alias.size)}`)
  return alias.get(email)!
}

async function main() {
  const col = await prisma.collection.findUnique({ where: { id: COLLECTION_ID } })
  console.log('=== COLLECTION RECORD (todos os campos persistidos) ===')
  console.log(JSON.stringify(col, null, 1))

  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId: COLLECTION_ID },
    select: { acq_id: true, secs: true, activeSecs: true, curationSecs: true, reviewedSecs: true },
  })
  const sum = (f: (g: (typeof groups)[0]) => number) => groups.reduce((a, g) => a + f(g), 0)
  console.log('\n=== GROUPS ===')
  console.log(JSON.stringify({
    acqs: groups.length,
    frames_secs: sum(g => g.secs.length),
    frames_activeSecs: sum(g => g.activeSecs.length),
    acqs_with_activeSecs: groups.filter(g => g.activeSecs.length > 0).length,
    curation_frames: sum(g => g.curationSecs.length),
    acqs_with_curation: groups.filter(g => g.curationSecs.length > 0).length,
    reviewed_frames: sum(g => g.reviewedSecs.length),
    acqs_with_reviewed: groups.filter(g => g.reviewedSecs.length > 0).length,
  }, null, 1))
  console.log('acq_id sample:', groups.slice(0, 3).map(g => g.acq_id))

  const rows = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $match: { 'yolo.action': { $exists: true } } },
      { $project: { _id: 0, acq_id: 1, sec: 1, h: { $filter: { input: '$yolo', as: 'd', cond: { $ne: [{ $ifNull: ['$$d.action', null] }, null] } } } } },
      { $unwind: '$h' },
      { $project: { acq_id: 1, sec: 1, action: '$h.action', by: '$h.reviewed_by', at: '$h.reviewed_at', cid: '$h.collection_id' } },
    ],
  })) as unknown as any[]

  console.log('\n=== ENTRADAS HUMANAS (todo o banco) ===')
  console.log('total:', rows.length)

  const byPerson: Record<string, any> = {}
  for (const r of rows) {
    const p = anon(r.by)
    byPerson[p] ??= { total: 0, actions: {} as Record<string, number>, first: null, last: null, collections: new Set<string>(), frames: new Set<string>() }
    const b = byPerson[p]
    b.total++
    b.actions[r.action] = (b.actions[r.action] ?? 0) + 1
    b.collections.add(r.cid ?? '(sem collection_id)')
    b.frames.add(`${r.acq_id}:${r.sec}`)
    const at = r.at?.$date ?? r.at
    if (at) {
      if (!b.first || at < b.first) b.first = at
      if (!b.last || at > b.last) b.last = at
    }
  }
  for (const [p, b] of Object.entries<any>(byPerson)) {
    console.log(JSON.stringify({
      reviewer: p, total: b.total, actions: b.actions,
      distinct_frames: b.frames.size, first_at: b.first, last_at: b.last,
      collection_ids: [...b.collections],
    }, null, 1))
  }
  console.log('identidades distintas em reviewed_by:', alias.size)
  console.log('sha256(email)[0:12]:', [...alias.entries()].map(([e, a]) => `${a}=${createHash('sha256').update(e).digest('hex').slice(0, 12)}`))

  console.log(JSON.stringify({
    sem_reviewed_by: rows.filter(r => !r.by).length,
    sem_reviewed_at: rows.filter(r => !r.at).length,
    sem_collection_id: rows.filter(r => !r.cid).length,
  }, null, 1))

  const dup = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $match: { 'yolo.action': { $exists: true } } },
      { $project: { _id: 0, acq_id: 1, sec: 1, h: { $filter: { input: '$yolo', as: 'd', cond: { $ne: [{ $ifNull: ['$$d.ref_det_id', null] }, null] } } } } },
      { $unwind: '$h' },
      { $group: { _id: { a: '$acq_id', s: '$sec', r: '$h.ref_det_id' }, n: { $sum: 1 }, reviewers: { $addToSet: '$h.reviewed_by' } } },
      { $match: { n: { $gt: 1 } } },
      { $count: 'n' },
    ],
  })) as unknown as any[]
  console.log('\noriginais com MAIS DE UMA entrada humana:', dup[0]?.n ?? 0)

  const byDay: Record<string, number> = {}
  for (const r of rows) {
    const at = r.at?.$date ?? r.at
    if (!at) continue
    byDay[String(at).slice(0, 10)] = (byDay[String(at).slice(0, 10)] ?? 0) + 1
  }
  console.log('\n=== ENTRADAS POR DIA ===')
  console.log(JSON.stringify(byDay, null, 1))
}

main().catch(e => { console.error(e); process.exit(1) }).finally(() => prisma.$disconnect())
