import { FastifyRequest, FastifyReply } from 'fastify'
import { randomUUID } from 'crypto'
import prisma from '../../../lib/prisma'
import { normalizeToCompact } from '../get-collection-seconds-with-links'
import {
  BoxN, RawDet, ROUND_DEFAULT, collectionRound, isHuman, isHumanOfRound, isValidBoxN,
  isValidClass, normalizeClass, recorderVariants, serializeDetections,
} from './shared'

type ActionInput =
  | { type: 'corrected'; ref_idx: number; class?: string; box_n: BoxN }
  | { type: 'rejected'; ref_idx: number }
  | { type: 'kept'; ref_idx: number }
  | { type: 'added'; class: string; box_n: BoxN }
  | { type: 'undo'; target_idx: number }

type Body = {
  acq_id?: string
  sec?: number
  img_w?: number
  img_h?: number
  actions?: ActionInput[]
  mark_done?: boolean
}

// C. POST /collections/:collectionId/curation/frame/action — grava curadoria (append-only).
export async function postCurationAction(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id
  const email = request.user!.email
  const { collectionId } = request.params as { collectionId: string }
  const body = request.body as Body

  const acqRaw = body?.acq_id
  const sec = body?.sec
  const actions = body?.actions
  if (!acqRaw || typeof sec !== 'number' || !Number.isInteger(sec)) {
    return reply.status(400).send({ error: 'acq_id e sec (inteiro) são obrigatórios.' })
  }
  if (!Array.isArray(actions)) {
    return reply.status(400).send({ error: 'actions deve ser um array.' })
  }
  const img_w = typeof body.img_w === 'number' && body.img_w > 0 ? body.img_w : null
  const img_h = typeof body.img_h === 'number' && body.img_h > 0 ? body.img_h : null

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: { id: true, curation_locked: true, review_round: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })
  if (!collection.curation_locked) return reply.status(409).send({ error: 'Curadoria não travada.' })

  // Namespace de escrita: esta coleção só enxerga e só toca entradas da SUA rodada.
  const round = collectionRound(collection.review_round)

  const acqC = normalizeToCompact(acqRaw)

  const group = await prisma.collectionGroup.findFirst({
    where: { collectionId, acq_id: { in: recorderVariants(acqC) } },
    select: { id: true, curationSecs: true, reviewedSecs: true },
  })
  if (!group || !group.curationSecs.includes(sec)) {
    return reply.status(404).send({ error: 'Frame não pertence ao conjunto travado desta coleção.' })
  }

  const n = Number(acqC)
  const doc = await prisma.big1Hz.findFirst({
    where: { acq_id: n, sec },
    select: { id: true, yolo: true },
  })
  if (!doc) return reply.status(404).send({ error: 'Telemetria do frame não encontrada.' })

  // Cópia de trabalho: cada objeto ganha identidade nova, preservando os campos conhecidos.
  const arr: RawDet[] = ((doc.yolo as any as RawDet[]) ?? []).map((d) => ({ ...d }))
  const now = new Date()

  // ── Pré-passe: resolve todo idx→objeto (no snapshot que B viu) e atribui det_id
  //    ANTES de qualquer mutação (evita deslocamento de índice). ──
  type Resolved =
    | { type: 'corrected'; target: RawDet; cls: string | null; box: BoxN }
    | { type: 'rejected'; target: RawDet }
    | { type: 'kept'; target: RawDet }
    | { type: 'added'; cls: string; box: BoxN }
    | { type: 'undo'; target: RawDet }
  const resolved: Resolved[] = []

  const resolveOriginal = (idx: unknown): RawDet | null => {
    if (typeof idx !== 'number' || !Number.isInteger(idx)) return null
    const t = arr[idx]
    if (!t || isHuman(t)) return null
    if (!t.det_id) t.det_id = randomUUID()
    return t
  }

  for (const a of actions) {
    switch (a?.type) {
      case 'corrected': {
        const t = resolveOriginal((a as any).ref_idx)
        if (!t) return reply.status(400).send({ error: 'corrected: ref_idx inválido (não é um original).' })
        if (a.class !== undefined && !isValidClass(a.class)) {
          return reply.status(400).send({ error: `corrected: classe inválida "${a.class}".` })
        }
        if (!isValidBoxN(a.box_n)) return reply.status(400).send({ error: 'corrected: box_n inválido.' })
        const cls = a.class !== undefined ? normalizeClass(a.class) : t.class ?? null
        resolved.push({ type: 'corrected', target: t, cls, box: a.box_n })
        break
      }
      case 'rejected': {
        const t = resolveOriginal((a as any).ref_idx)
        if (!t) return reply.status(400).send({ error: 'rejected: ref_idx inválido (não é um original).' })
        resolved.push({ type: 'rejected', target: t })
        break
      }
      case 'kept': {
        const t = resolveOriginal((a as any).ref_idx)
        if (!t) return reply.status(400).send({ error: 'kept: ref_idx inválido (não é um original).' })
        resolved.push({ type: 'kept', target: t })
        break
      }
      case 'added': {
        if (!isValidClass(a.class)) return reply.status(400).send({ error: `added: classe inválida "${a.class}".` })
        if (!isValidBoxN(a.box_n)) return reply.status(400).send({ error: 'added: box_n inválido.' })
        resolved.push({ type: 'added', cls: normalizeClass(a.class), box: a.box_n })
        break
      }
      case 'undo': {
        const idx = (a as any).target_idx
        if (typeof idx !== 'number' || !Number.isInteger(idx) || !arr[idx] || !isHuman(arr[idx])) {
          return reply.status(400).send({ error: 'undo: target_idx não referencia uma entrada humana.' })
        }
        // Desfazer só alcança a própria rodada — nunca a revisão de outro anotador.
        if (!isHumanOfRound(arr[idx], round)) {
          return reply.status(403).send({ error: 'undo: entrada pertence a outra rodada de revisão.' })
        }
        resolved.push({ type: 'undo', target: arr[idx] })
        break
      }
      default:
        return reply.status(400).send({ error: `Ação de tipo desconhecido: ${(a as any)?.type}` })
    }
  }

  // ── Aplica por identidade (det_id), não por índice. ──
  const removeObj = (o: RawDet) => {
    const i = arr.indexOf(o)
    if (i >= 0) arr.splice(i, 1)
  }
  // Idempotência: recorrigir substitui a humana anterior sobre o mesmo original —
  // mas SÓ a desta rodada. Sem o filtro de round, um segundo anotador revisando o mesmo
  // frame apagaria silenciosamente o veredito do primeiro (o gabarito travado).
  const dropPriorHumanRef = (detId: string) => {
    for (let i = arr.length - 1; i >= 0; i--) {
      if (isHumanOfRound(arr[i], round) && arr[i].ref_det_id === detId) arr.splice(i, 1)
    }
  }
  const boxFields = (box: BoxN): RawDet => {
    const o: RawDet = { x1_n: box.x1, y1_n: box.y1, x2_n: box.x2, y2_n: box.y2 }
    if (img_w && img_h) {
      o.x1 = box.x1 * img_w; o.y1 = box.y1 * img_h
      o.x2 = box.x2 * img_w; o.y2 = box.y2 * img_h
      o.w = o.x2 - o.x1; o.h = o.y2 - o.y1
      o.cx = (o.x1 + o.x2) / 2; o.cy = (o.y1 + o.y2) / 2
      o.area_px = o.w * o.h
    }
    return o
  }
  // Ponto único por onde TODA entrada humana passa (corrected/rejected/kept/added).
  // `collection_id` carimba a leva aqui, e não em cada ramo, para que um tipo de
  // ação novo nasça rastreável sem ninguém lembrar de adicioná-lo.
  const mkHuman = (fields: RawDet): RawDet => ({
    det_id: randomUUID(),
    conf: null,
    reviewed_by: email,
    reviewed_at: now,
    collection_id: collectionId,
    review_round: round,
    ...fields,
  })
  // `reviewed`/`reviewed_by` vivem no ORIGINAL, que é compartilhado por todas as rodadas.
  // Só o round 1 os escreve; fora dele o progresso é derivado das entradas da rodada
  // (serializeDetections) e de reviewedSecs, que já é por-coleção.
  const markReviewed = (o: RawDet) => {
    if (round !== ROUND_DEFAULT) return
    o.reviewed = true
    o.reviewed_by = email
    o.reviewed_at = now
  }

  for (const r of resolved) {
    if (r.type === 'corrected') {
      markReviewed(r.target)
      dropPriorHumanRef(r.target.det_id)
      arr.push(mkHuman({
        class: r.cls, source: 'human_corrected', action: 'corrected',
        ref_det_id: r.target.det_id, ...boxFields(r.box),
      }))
    } else if (r.type === 'rejected') {
      markReviewed(r.target)
      dropPriorHumanRef(r.target.det_id)
      arr.push(mkHuman({
        class: r.target.class ?? null, source: 'human_corrected', action: 'rejected',
        ref_det_id: r.target.det_id,
      }))
    } else if (r.type === 'kept') {
      markReviewed(r.target)
      dropPriorHumanRef(r.target.det_id)
      arr.push(mkHuman({
        class: r.target.class ?? null, source: 'human_corrected', action: 'kept',
        ref_det_id: r.target.det_id,
      }))
    } else if (r.type === 'added') {
      arr.push(mkHuman({
        class: r.cls, source: 'human_added', action: 'added', ref_det_id: null, ...boxFields(r.box),
      }))
    } else if (r.type === 'undo') {
      const ref = r.target.ref_det_id
      removeObj(r.target)
      if (ref && round === ROUND_DEFAULT) {
        // Se foi a última humana da rodada referenciando o original, limpa reviewed.
        const stillRef = arr.some((o) => isHumanOfRound(o, round) && o.ref_det_id === ref)
        if (!stillRef) {
          const orig = arr.find((o) => !isHuman(o) && o.det_id === ref)
          if (orig) {
            orig.reviewed = null
            orig.reviewed_by = null
            orig.reviewed_at = null
          }
        }
      }
    }
  }

  // reviewedSecs: push/remove conforme mark_done.
  const reviewedSet = new Set(group.reviewedSecs)
  if (body.mark_done === true) reviewedSet.add(sec)
  else if (body.mark_done === false) reviewedSet.delete(sec)
  const reviewedSecs = [...reviewedSet].sort((a, b) => a - b)

  await prisma.$transaction([
    prisma.big1Hz.update({ where: { id: doc.id }, data: { yolo: { set: arr as any } } }),
    prisma.collectionGroup.update({ where: { id: group.id }, data: { reviewedSecs } }),
  ])

  // Retorna detecções com idx frescos → front re-renderiza sem novo GET a B.
  const detections = serializeDetections(arr, round)
  return reply.send({
    acq_id: acqC,
    sec,
    reviewed: reviewedSet.has(sec),
    detections,
    yolo_count_model: detections.yolo_count_model,
  })
}
