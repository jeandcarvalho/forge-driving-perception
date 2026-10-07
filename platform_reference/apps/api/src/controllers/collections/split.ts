import { FastifyRequest, FastifyReply } from 'fastify'
import prisma from '../../lib/prisma'

export async function applySplit(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id
  const { collectionId } = request.params as { collectionId: string }
  const body = request.body as { train_acq_ids?: string[]; test_acq_ids?: string[] }

  if (!Array.isArray(body?.train_acq_ids) || !Array.isArray(body?.test_acq_ids)) {
    return reply.status(400).send({ error: 'train_acq_ids e test_acq_ids são obrigatórios.' })
  }

  const { train_acq_ids, test_acq_ids } = body

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: { id: true, gps_spacing_m: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  const testSet = new Set(test_acq_ids)
  const overlap = train_acq_ids.filter((id) => testSet.has(id))
  if (overlap.length > 0) {
    return reply.status(400).send({
      error: `acq_ids duplicados em treino e teste: ${overlap.join(', ')}`,
    })
  }

  const hasFilter = collection.gps_spacing_m !== null

  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId },
    select: { acq_id: true, secs: true, activeSecs: true },
  })

  const validAcqIds = new Set<string>()
  const secsCountByAcq = new Map<string, number>()

  for (const group of groups) {
    validAcqIds.add(group.acq_id)
    const effectiveSecs =
      hasFilter && group.activeSecs.length > 0 ? group.activeSecs : group.secs
    secsCountByAcq.set(group.acq_id, effectiveSecs.length)
  }

  const allSubmitted = [...train_acq_ids, ...test_acq_ids]
  const invalid = allSubmitted.filter((id) => !validAcqIds.has(id))
  if (invalid.length > 0) {
    return reply.status(400).send({
      error: `acq_ids não pertencem a esta coleção: ${invalid.join(', ')}`,
    })
  }

  const trainSet = new Set(train_acq_ids)
  const trainFrames = train_acq_ids.reduce((sum, id) => sum + (secsCountByAcq.get(id) ?? 0), 0)
  const testFrames = test_acq_ids.reduce((sum, id) => sum + (secsCountByAcq.get(id) ?? 0), 0)

  await prisma.collection.update({
    where: { id: collectionId },
    data: { train_acq_ids, test_acq_ids },
  })

  return reply.send({ train_frames: trainFrames, test_frames: testFrames, train_acq_ids, test_acq_ids })
}
