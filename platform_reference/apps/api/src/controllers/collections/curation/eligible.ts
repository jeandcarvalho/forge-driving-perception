import { FastifyRequest, FastifyReply } from 'fastify'
import prisma from '../../../lib/prisma'
import { normalizeToCompact } from '../get-collection-seconds-with-links'
import { isHuman } from './shared'

// A2. GET /collections/:collectionId/curation/eligible — leitura pura (não recalcula).
export async function getCurationEligible(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: CURATION_META_SELECT,
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  return sendEligible(reply, collectionId, collection)
}

// Versão pública (leitura) — mesmo payload, sem checagem de dono e sem gerar nada.
// Se o dono ainda não travou o conjunto, `locked: false` + `frames: []` e o front
// mostra "curadoria ainda não iniciada".
export async function publicGetCurationEligible(request: FastifyRequest, reply: FastifyReply) {
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId },
    select: CURATION_META_SELECT,
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  return sendEligible(reply, collectionId, collection)
}

const CURATION_META_SELECT = {
  id: true, curation_locked: true, curation_min_distance_m: true,
  curation_frame_count: true, curation_generated_at: true, curation_generated_by: true,
} as const

type CurationMeta = {
  curation_locked: boolean
  curation_min_distance_m: number | null
  curation_frame_count: number | null
  curation_generated_at: Date | null
  curation_generated_by: string | null
}

async function sendEligible(reply: FastifyReply, collectionId: string, collection: CurationMeta) {
  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId },
    select: { acq_id: true, curationSecs: true, reviewedSecs: true },
  })

  // Frames travados (curationSecs) por acq + flag reviewed.
  type Frame = { acq_id: string; sec: number; reviewed: boolean; yolo_count: number }
  const frames: Frame[] = []
  const acqSecMap = new Map<number, number[]>() // para a query de yolo_count

  for (const g of groups) {
    if (g.curationSecs.length === 0) continue
    const acqC = normalizeToCompact(g.acq_id)
    const reviewedSet = new Set(g.reviewedSecs)
    const n = Number(acqC)
    for (const sec of g.curationSecs) {
      frames.push({ acq_id: acqC, sec, reviewed: reviewedSet.has(sec), yolo_count: 0 })
      if (!isNaN(n)) {
        if (!acqSecMap.has(n)) acqSecMap.set(n, [])
        acqSecMap.get(n)!.push(sec)
      }
    }
  }

  // yolo_count on-read: conta só originais do modelo (0 = candidato a falso negativo).
  if (acqSecMap.size > 0) {
    const docs = await prisma.big1Hz.findMany({
      where: { OR: [...acqSecMap.entries()].map(([acq_id, secs]) => ({ acq_id, sec: { in: secs } })) },
      select: { acq_id: true, sec: true, yolo: true },
    })
    const countByKey = new Map<string, number>()
    for (const d of docs) {
      if (d.acq_id == null) continue
      const originals = (d.yolo ?? []).filter((det) => !isHuman(det as any)).length
      countByKey.set(`${d.acq_id}:${d.sec}`, originals)
    }
    for (const f of frames) {
      f.yolo_count = countByKey.get(`${Number(f.acq_id)}:${f.sec}`) ?? 0
    }
  }

  frames.sort((a, b) => (a.acq_id < b.acq_id ? -1 : a.acq_id > b.acq_id ? 1 : a.sec - b.sec))

  const reviewedCount = frames.filter((f) => f.reviewed).length

  return reply.send({
    locked: collection.curation_locked,
    min_distance_m: collection.curation_min_distance_m,
    frame_count: collection.curation_frame_count ?? frames.length,
    generated_at: collection.curation_generated_at,
    generated_by: collection.curation_generated_by,
    total: frames.length,
    reviewed: reviewedCount,
    frames,
  })
}
