import { FastifyRequest, FastifyReply } from 'fastify'
import prisma from '../../../lib/prisma'
import { normalizeToCompact } from '../get-collection-seconds-with-links'
import {
  Frame, TARGET, LO_BAND, HI_BAND,
  extractGpsAndCoverage, buildCandidates, selectCurationSet,
} from './select'

// A1. POST /collections/:collectionId/curation/generate — gera e trava o conjunto de teste.
//
// Modelo POOL INTEIRO + split por-frame + EXCLUSIVIDADE DE AQUISIÇÃO:
//  - Candidatos = TODAS as aquisições da coleção (sem split manual prévio).
//  - Teste = ~300 frames por espaçamento intra (busca binária em d_intra).
//  - Antivazamento = toda acq que doou ≥1 frame de teste sai INTEIRA do treino
//    (frames restantes dessa acq são descartados, não viram treino).
//  - Treino = todos os frames das acqs NÃO tocadas pelo teste (derivável: acq sem curationSecs).
//  - train_acq_ids/test_acq_ids (split manual antigo) são ZERADOS.
export async function generateCuration(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id
  const email = request.user!.email
  const { collectionId } = request.params as { collectionId: string }
  const { force } = request.query as { force?: string }
  const forceRegen = force === 'true' || force === '1'

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: { id: true, gps_spacing_m: true, curation_locked: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  if (collection.curation_locked && !forceRegen) {
    return reply.status(409).send({
      error: 'Curadoria já travada. Use ?force=true para regenerar (apaga o progresso atual).',
      locked: true,
    })
  }

  const hasFilter = collection.gps_spacing_m !== null

  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId },
    select: { id: true, acq_id: true, secs: true, activeSecs: true },
  })
  if (groups.length === 0) {
    return reply.status(400).send({ error: 'Coleção vazia.' })
  }

  // Pool = TODAS as aquisições (effectiveSecs), chaveado por canonical compacto.
  const poolFrames: Frame[] = []
  const groupIdByAcqC = new Map<string, string>()
  const effSecsByAcqC = new Map<string, number>() // total de frames (effectiveSecs) por acq

  for (const g of groups) {
    const acqC = normalizeToCompact(g.acq_id)
    const eff = hasFilter && g.activeSecs.length > 0 ? g.activeSecs : g.secs
    groupIdByAcqC.set(acqC, g.id)
    effSecsByAcqC.set(acqC, (effSecsByAcqC.get(acqC) ?? 0) + eff.length)
    for (const sec of eff) poolFrames.push({ acqC, sec })
  }

  // Telemetria do pool inteiro: GPS + cobertura yolo.
  const acqSecMap = new Map<number, number[]>()
  for (const f of poolFrames) {
    const n = Number(f.acqC)
    if (isNaN(n)) continue
    if (!acqSecMap.has(n)) acqSecMap.set(n, [])
    acqSecMap.get(n)!.push(f.sec)
  }

  const poolDocs = acqSecMap.size === 0 ? [] : await prisma.big1Hz.findMany({
    where: { OR: [...acqSecMap.entries()].map(([acq_id, secs]) => ({ acq_id, sec: { in: secs } })) },
    select: { acq_id: true, sec: true, can: true, yolo: true },
  })

  // Núcleo puro (mesmo código exercido pelo script de teste).
  const { gpsBySecKey, acqCWithYolo } = extractGpsAndCoverage(poolDocs as any)
  const candidates = buildCandidates(poolFrames, gpsBySecKey, acqCWithYolo)
  const { selected, minDistanceM, underTarget } = selectCurationSet(candidates)

  // Agrupa seleção por acq → curationSecs (só as acqs de teste ganham).
  const selByAcq = new Map<string, number[]>()
  for (const c of selected) {
    if (!selByAcq.has(c.acqC)) selByAcq.set(c.acqC, [])
    selByAcq.get(c.acqC)!.push(c.sec)
  }

  // Contagens derivadas para auditoria/resposta (exclusividade por acq).
  let trainFrameCount = 0
  for (const [acqC, count] of effSecsByAcqC) {
    if (!selByAcq.has(acqC)) trainFrameCount += count // acq intocada pelo teste → treino
  }
  const testFrameCount = selected.length
  const testAcqs = selByAcq.size
  const trainAcqs = groupIdByAcqC.size - testAcqs
  // Frames das acqs de teste que NÃO viraram teste → descartados (margem de exclusividade).
  let discardedFrameCount = 0
  for (const acqC of selByAcq.keys()) {
    discardedFrameCount += (effSecsByAcqC.get(acqC) ?? 0) - (selByAcq.get(acqC)!.length)
  }

  const now = new Date()

  // Transação: zera curadoria em TODOS os grupos, grava curationSecs nas acqs de teste,
  // e atualiza metadados + zera o split manual antigo.
  await prisma.$transaction([
    prisma.collectionGroup.updateMany({
      where: { collectionId },
      data: { curationSecs: [], reviewedSecs: [] },
    }),
    ...[...selByAcq.entries()].map(([acqC, secs]) =>
      prisma.collectionGroup.update({
        where: { id: groupIdByAcqC.get(acqC)! },
        data: { curationSecs: [...secs].sort((a, b) => a - b), reviewedSecs: [] },
      })
    ),
    prisma.collection.update({
      where: { id: collectionId },
      data: {
        curation_locked: true,
        curation_min_distance_m: minDistanceM,
        curation_frame_count: testFrameCount,
        curation_generated_at: now,
        curation_generated_by: email,
        // Split manual antigo aposentado — o split agora é por-frame (curationSecs).
        train_acq_ids: [],
        test_acq_ids: [],
      },
    }),
  ])

  return reply.send({
    locked: true,
    // Teste
    frame_count: testFrameCount,
    test_acqs: testAcqs,
    min_distance_m: minDistanceM,
    under_target: underTarget,
    target: TARGET,
    band: [LO_BAND, HI_BAND],
    // Treino (derivado por exclusividade de acq)
    train_frame_count: trainFrameCount,
    train_acqs: trainAcqs,
    discarded_frame_count: discardedFrameCount,
    // Pool
    candidates_total: candidates.length,
    pool_acqs: groupIdByAcqC.size,
    generated_at: now,
    generated_by: email,
  })
}
