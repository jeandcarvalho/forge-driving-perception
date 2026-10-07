import { FastifyRequest, FastifyReply } from 'fastify'
import prisma from '../../../lib/prisma'
import { normalizeToCompact } from '../get-collection-seconds-with-links'
import { IMAGE_EXTS, collectionRound, recorderVariants, serializeDetections } from './shared'

// B. GET /collections/:collectionId/curation/frame?acq_id=&sec= — frame único.
export async function getCurationFrame(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: { id: true, curation_locked: true, review_round: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  return sendFrame(request, reply, collectionId, collection.curation_locked, collectionRound(collection.review_round))
}

// Versão pública (leitura) — mesmo frame + caixas já curadas, sem checagem de dono.
export async function publicGetCurationFrame(request: FastifyRequest, reply: FastifyReply) {
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId },
    select: { id: true, curation_locked: true, review_round: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  return sendFrame(request, reply, collectionId, collection.curation_locked, collectionRound(collection.review_round))
}

async function sendFrame(
  request: FastifyRequest,
  reply: FastifyReply,
  collectionId: string,
  locked: boolean,
  round: number
) {
  const { acq_id, sec: secRaw } = request.query as { acq_id?: string; sec?: string }

  if (!acq_id || secRaw == null) {
    return reply.status(400).send({ error: 'acq_id e sec são obrigatórios.' })
  }
  const sec = Number(secRaw)
  if (!Number.isInteger(sec)) return reply.status(400).send({ error: 'sec inválido.' })

  if (!locked) return reply.status(409).send({ error: 'Curadoria não travada.' })

  const acqC = normalizeToCompact(acq_id)

  // Valida sec ∈ curationSecs deste grupo (permissão por-coleção).
  const group = await prisma.collectionGroup.findFirst({
    where: { collectionId, acq_id: { in: recorderVariants(acqC) } },
    select: { curationSecs: true, reviewedSecs: true },
  })
  if (!group || !group.curationSecs.includes(sec)) {
    return reply.status(404).send({ error: 'Frame não pertence ao conjunto travado desta coleção.' })
  }

  const n = Number(acqC)
  const doc = await prisma.big1Hz.findFirst({
    where: { acq_id: n, sec },
    select: { yolo: true, can: true },
  })

  // Imagem (câmera única → primeira jpg/png do sec).
  const linkDoc = await prisma.links.findFirst({
    where: { acq_id: { in: recorderVariants(acqC) }, sec },
    select: { ext: true, link: true },
  })
  let image: { ext: string; link: string } | null = null
  if (linkDoc?.link) {
    const ext = (linkDoc.ext || '').toLowerCase().replace(/^\./, '')
    if (IMAGE_EXTS.has(ext)) image = { ext, link: linkDoc.link }
  }

  // Só as entradas humanas DESTA rodada saem daqui: numa coleção round 2 o payload
  // não carrega nenhum vestígio da decisão do anotador 1 (o front fica cego por
  // construção, sem precisar conhecer rodadas).
  const detections = serializeDetections((doc?.yolo as any) ?? [], round)
  const gps =
    doc?.can?.GPS_x != null && doc?.can?.GPS_y != null
      ? { lat: doc.can.GPS_y, lon: doc.can.GPS_x }
      : null

  return reply.send({
    acq_id: acqC,
    sec,
    reviewed: group.reviewedSecs.includes(sec),
    image,
    gps,
    detections,
    yolo_count_model: detections.yolo_count_model,
  })
}
