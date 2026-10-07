import { FastifyRequest, FastifyReply } from 'fastify'
import prisma from '../../lib/prisma'

const IMAGE_EXTS = new Set(['jpg', 'jpeg', 'png', 'webp', 'gif'])

type ImageLink = { sec: number; ext: string; link: string }
type FileLink = { ext: string; link: string }

export function normalizeToCompact(acqId: string): string {
  if (!acqId) return acqId
  if (/^\d{14}$/.test(acqId)) return acqId
  const match = acqId.match(/(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})/)
  if (match) return match.slice(1).join('')
  const digits = acqId.replace(/\D/g, '')
  return digits.length === 14 ? digits : acqId
}

export function compactToRecorder(compact: string): string | null {
  if (!/^\d{14}$/.test(compact)) return null
  const [y, mo, d, h, mi, s] = [
    compact.slice(0, 4), compact.slice(4, 6), compact.slice(6, 8),
    compact.slice(8, 10), compact.slice(10, 12), compact.slice(12, 14),
  ]
  return `Recorder_${y}-${mo}-${d}_${h}-${mi}-${s}`
}

export async function getCollectionSecondsWithLinks(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: {
      id: true, name: true, description: true,
      gps_spacing_m: true, train_acq_ids: true, test_acq_ids: true,
      curation_locked: true, curation_frame_count: true, curation_generated_at: true,
    },
  })
  if (!collection) return reply.status(404).send({ error: 'Collection not found' })

  const hasSpacingFilter = collection.gps_spacing_m !== null
  const items = await buildItemsMeta(collectionId, hasSpacingFilter)

  return reply.send({
    collectionId: collection.id,
    name: collection.name,
    description: collection.description,
    gps_spacing_m: collection.gps_spacing_m,
    active_item_ids_count: items.reduce((sum, i) => sum + i.active_secs_count, 0),
    train_acq_ids: collection.train_acq_ids,
    test_acq_ids: collection.test_acq_ids,
    curation_locked: collection.curation_locked,
    curation_frame_count: collection.curation_frame_count,
    curation_generated_at: collection.curation_generated_at,
    items,
  })
}

export async function publicGetCollectionView(request: FastifyRequest, reply: FastifyReply) {
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId },
    select: { id: true, name: true, description: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Collection not found' })

  const items = await buildItemsFull(collectionId, false)

  return reply.send({
    collectionId: collection.id,
    name: collection.name,
    description: collection.description,
    items,
  })
}

// Lean metadata fetch — no image links, only file (sec=null) links
async function buildItemsMeta(collectionId: string, hasSpacingFilter: boolean) {
  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId },
    select: { acq_id: true, secs: true, activeSecs: true },
  })

  if (groups.length === 0) return []

  const secsByAcq = new Map<string, Set<number>>()
  const activeSecsCountByAcq = new Map<string, number>()

  for (const group of groups) {
    const canonical = normalizeToCompact(group.acq_id)
    if (!secsByAcq.has(canonical)) secsByAcq.set(canonical, new Set())
    for (const s of group.secs) secsByAcq.get(canonical)!.add(s)

    if (hasSpacingFilter) {
      const active = group.activeSecs.length > 0 ? group.activeSecs.length : group.secs.length
      activeSecsCountByAcq.set(canonical, (activeSecsCountByAcq.get(canonical) ?? 0) + active)
    }
  }

  const searchIds = new Set<string>()
  for (const canonical of secsByAcq.keys()) {
    searchIds.add(canonical)
    const recorder = compactToRecorder(canonical)
    if (recorder) searchIds.add(recorder)
  }

  // Only fetch null-sec links (download files, not images — tiny query)
  const fileLinksDocs = await prisma.links.findMany({
    where: { acq_id: { in: [...searchIds] }, sec: null },
    select: { acq_id: true, ext: true, link: true },
  })

  const filesByAcq = new Map<string, FileLink[]>()
  for (const doc of fileLinksDocs) {
    const canonical = normalizeToCompact(doc.acq_id)
    if (!secsByAcq.has(canonical)) continue
    if (!filesByAcq.has(canonical)) filesByAcq.set(canonical, [])
    const ext = (doc.ext || '').toLowerCase().replace(/^\./, '')
    const existing = filesByAcq.get(canonical)!
    if (!existing.some((f) => f.ext === ext && f.link === doc.link)) {
      existing.push({ ext, link: doc.link })
    }
  }

  return [...secsByAcq.keys()].sort().map((canonical) => {
    const secsSet = secsByAcq.get(canonical)!
    const secs = [...secsSet].sort((a, b) => a - b)
    const files = filesByAcq.get(canonical) || []
    const active_secs_count = hasSpacingFilter
      ? (activeSecsCountByAcq.get(canonical) ?? secs.length)
      : secs.length
    return { acq_id: canonical, secs, active_secs_count, files }
  })
}

// Full fetch including images (used by public view)
async function buildItemsFull(collectionId: string, hasSpacingFilter: boolean) {
  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId },
    select: { acq_id: true, secs: true, activeSecs: true },
  })

  if (groups.length === 0) return []

  const secsByAcq = new Map<string, Set<number>>()
  const activeSecsCountByAcq = new Map<string, number>()

  for (const group of groups) {
    const canonical = normalizeToCompact(group.acq_id)
    if (!secsByAcq.has(canonical)) secsByAcq.set(canonical, new Set())
    for (const s of group.secs) secsByAcq.get(canonical)!.add(s)

    if (hasSpacingFilter) {
      const active = group.activeSecs.length > 0 ? group.activeSecs.length : group.secs.length
      activeSecsCountByAcq.set(canonical, (activeSecsCountByAcq.get(canonical) ?? 0) + active)
    }
  }

  const searchIds = new Set<string>()
  for (const canonical of secsByAcq.keys()) {
    searchIds.add(canonical)
    const recorder = compactToRecorder(canonical)
    if (recorder) searchIds.add(recorder)
  }

  const collectionSecs = [...new Set([...secsByAcq.values()].flatMap((s) => [...s]))]

  const linksDocs = await prisma.links.findMany({
    where: {
      acq_id: { in: [...searchIds] },
      OR: [{ sec: { in: collectionSecs } }, { sec: null }],
    },
    select: { acq_id: true, sec: true, ext: true, link: true },
  })

  const imagesByAcq = new Map<string, ImageLink[]>()
  const filesByAcq = new Map<string, FileLink[]>()

  for (const doc of linksDocs) {
    const canonical = normalizeToCompact(doc.acq_id)
    if (!secsByAcq.has(canonical)) continue

    const ext = (doc.ext || '').toLowerCase().replace(/^\./, '')
    const sec = doc.sec ?? null

    if (sec !== null && IMAGE_EXTS.has(ext)) {
      if (!imagesByAcq.has(canonical)) imagesByAcq.set(canonical, [])
      imagesByAcq.get(canonical)!.push({ sec, ext, link: doc.link })
    } else {
      if (!filesByAcq.has(canonical)) filesByAcq.set(canonical, [])
      const existing = filesByAcq.get(canonical)!
      if (!existing.some((f) => f.ext === ext && f.link === doc.link)) {
        existing.push({ ext, link: doc.link })
      }
    }
  }

  return [...secsByAcq.keys()].sort().map((canonical) => {
    const secsSet = secsByAcq.get(canonical)!
    const secs = [...secsSet].sort((a, b) => a - b)
    const images = (imagesByAcq.get(canonical) || [])
      .filter((img) => secsSet.has(img.sec))
      .sort((a, b) => a.sec - b.sec)
    const files = filesByAcq.get(canonical) || []
    const active_secs_count = hasSpacingFilter
      ? (activeSecsCountByAcq.get(canonical) ?? secs.length)
      : secs.length
    return { acq_id: canonical, secs, active_secs_count, images, files }
  })
}
