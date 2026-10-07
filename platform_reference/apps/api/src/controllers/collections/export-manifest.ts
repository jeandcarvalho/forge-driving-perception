import { FastifyRequest, FastifyReply } from 'fastify'
import prisma from '../../lib/prisma'
import { normalizeToCompact, compactToRecorder } from './get-collection-seconds-with-links'
import { IMAGE_EXTS } from './curation/shared'
import { collectionRound, resolveGroundTruth } from './curation/shared'

// GET /public/collections/:collectionId/export/manifest
//   ?split=test|train|discarded|all  &page=1  &per_page=500
//
// Manifest de materialização do dataset: por frame, a imagem (id do Drive), o gabarito
// humano resolvido (só nos frames revisados), as detecções armazenadas e a telemetria.
//
// Rótulo de split, coerente com curation/generate.ts:
//   test      → sec ∈ curationSecs (conjunto travado)
//   train     → todos os secs de aquisições que NÃO doaram frame ao teste
//   discarded → secs restantes de aquisições que doaram (margem de antivazamento)

const DEFAULT_PER_PAGE = 500
const MAX_PER_PAGE = 2000

type Split = 'test' | 'train' | 'discarded'

// https://drive.google.com/file/d/<ID>/view?usp=... → <ID>
export function extractDriveId(link: string): string | null {
  if (!link) return null
  return (
    link.match(/\/file\/d\/([A-Za-z0-9_-]{10,})/)?.[1] ??
    link.match(/[?&]id=([A-Za-z0-9_-]{10,})/)?.[1] ??
    link.match(/googleusercontent\.com\/d\/([A-Za-z0-9_-]{10,})/)?.[1] ??
    null
  )
}

// Resolução não é armazenada em lugar nenhum: deriva de px ÷ normalizado quando há
// detecção com as duas formas. Null quando não dá — o Forge confere no download.
function deriveDims(yolo: any[] | null | undefined): { width: number; height: number } | null {
  for (const det of yolo ?? []) {
    if (det?.action != null) continue
    const { x2, x2_n, y2, y2_n } = det
    if (
      typeof x2 === 'number' && typeof x2_n === 'number' && x2_n > 0.05 &&
      typeof y2 === 'number' && typeof y2_n === 'number' && y2_n > 0.05
    ) {
      return { width: Math.round(x2 / x2_n), height: Math.round(y2 / y2_n) }
    }
  }
  return null
}

export async function publicExportManifest(request: FastifyRequest, reply: FastifyReply) {
  const { collectionId } = request.params as { collectionId: string }
  const query = request.query as { split?: string; page?: string; per_page?: string }

  const splitParam = (query.split ?? 'all').toLowerCase()
  if (!['test', 'train', 'discarded', 'all'].includes(splitParam)) {
    return reply.status(400).send({ error: 'split deve ser test | train | discarded | all.' })
  }
  const page = Math.max(1, Number(query.page ?? '1') || 1)
  const perPage = Math.min(MAX_PER_PAGE, Math.max(1, Number(query.per_page ?? DEFAULT_PER_PAGE) || DEFAULT_PER_PAGE))

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId },
    select: {
      id: true, name: true, description: true, gps_spacing_m: true, framesCount: true,
      curation_locked: true, curation_frame_count: true, curation_min_distance_m: true,
      curation_generated_at: true, curation_generated_by: true,
      review_round: true,
    },
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  // O manifesto é PÚBLICO: sem o round ele exportaria o gabarito (e os vereditos em
  // stored_detections) do round 1 na URL de uma coleção round 2 — vazamento direto
  // para o segundo anotador e para o Forge.
  const round = collectionRound(collection.review_round)

  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId },
    select: { acq_id: true, secs: true, activeSecs: true, curationSecs: true, reviewedSecs: true },
  })

  const hasSpacing = collection.gps_spacing_m !== null

  // Chaves de todos os frames, já rotuladas por split.
  type Key = { acqC: string; sec: number; split: Split; reviewed: boolean }
  const keys: Key[] = []
  const totals = { test: 0, train: 0, discarded: 0 }

  for (const g of groups) {
    const acqC = normalizeToCompact(g.acq_id)
    const effective = hasSpacing && g.activeSecs.length > 0 ? g.activeSecs : g.secs
    const curationSet = new Set(g.curationSecs)
    const reviewedSet = new Set(g.reviewedSecs)
    const donated = curationSet.size > 0

    for (const sec of effective) {
      const split: Split = curationSet.has(sec) ? 'test' : donated ? 'discarded' : 'train'
      totals[split]++
      if (splitParam !== 'all' && splitParam !== split) continue
      keys.push({ acqC, sec, split, reviewed: reviewedSet.has(sec) })
    }
  }

  keys.sort((a, b) => (a.acqC < b.acqC ? -1 : a.acqC > b.acqC ? 1 : a.sec - b.sec))

  const total = keys.length
  const slice = keys.slice((page - 1) * perPage, page * perPage)

  const meta = {
    collection: {
      id: collection.id,
      name: collection.name,
      description: collection.description,
      gps_spacing_m: collection.gps_spacing_m,
      frames_total: collection.framesCount,
      curation: {
        locked: collection.curation_locked,
        frame_count: collection.curation_frame_count,
        min_distance_m: collection.curation_min_distance_m,
        generated_at: collection.curation_generated_at,
        generated_by: collection.curation_generated_by,
      },
    },
    split: splitParam,
    totals,
    page,
    per_page: perPage,
    total,
    has_more: page * perPage < total,
    // As detecções armazenadas NÃO são um baseline avaliável: vieram de um passe com
    // tracking sobre vídeo e com corte de confiança (~0.2). Servem como pré-anotação
    // e para auditoria da curadoria. Para avaliar detector, rode inferência local.
    stored_detections_note:
      'Pré-anotação (tracking sobre vídeo, conf cortada em ~0.2). Não usar como baseline avaliável.',
  }

  if (slice.length === 0) return reply.send({ ...meta, frames: [] })

  // Telemetria + detecções da fatia.
  const byAcqNum = new Map<number, number[]>()
  for (const k of slice) {
    const n = Number(k.acqC)
    if (isNaN(n)) continue
    if (!byAcqNum.has(n)) byAcqNum.set(n, [])
    byAcqNum.get(n)!.push(k.sec)
  }

  const docs = byAcqNum.size === 0 ? [] : await prisma.big1Hz.findMany({
    where: { OR: [...byAcqNum.entries()].map(([acq_id, secs]) => ({ acq_id, sec: { in: secs } })) },
    select: { acq_id: true, sec: true, can: true, block: true, semseg: true, yolo: true, links: true },
  })
  const docByKey = new Map(docs.map((d) => [`${d.acq_id}:${d.sec}`, d]))

  // Links de imagem (fallback: a coleção `links`, quando big_1hz.links vier vazio).
  const searchIds = new Set<string>()
  for (const k of slice) {
    searchIds.add(k.acqC)
    const rec = compactToRecorder(k.acqC)
    if (rec) searchIds.add(rec)
  }
  const linkDocs = await prisma.links.findMany({
    where: { acq_id: { in: [...searchIds] }, sec: { in: [...new Set(slice.map((k) => k.sec))] } },
    select: { acq_id: true, sec: true, ext: true, link: true },
  })
  const linkByKey = new Map<string, { ext: string; link: string }>()
  for (const l of linkDocs) {
    if (l.sec === null) continue
    const ext = (l.ext || '').toLowerCase().replace(/^\./, '')
    if (!IMAGE_EXTS.has(ext)) continue
    const key = `${normalizeToCompact(l.acq_id)}:${l.sec}`
    if (!linkByKey.has(key)) linkByKey.set(key, { ext, link: l.link })
  }

  const frames = slice.map((k) => {
    const doc = docByKey.get(`${Number(k.acqC)}:${k.sec}`)
    const yolo = (doc?.yolo ?? []) as any[]

    const embedded = (doc?.links ?? []).find((l: any) => {
      const ext = (l?.ext || '').toLowerCase().replace(/^\./, '')
      return l?.link && IMAGE_EXTS.has(ext)
    })
    const chosen = embedded
      ? { ext: (embedded.ext || '').toLowerCase().replace(/^\./, ''), link: embedded.link as string }
      : linkByKey.get(`${k.acqC}:${k.sec}`) ?? null

    const driveId = chosen ? extractDriveId(chosen.link) : null
    const dims = deriveDims(yolo)

    const resolved = resolveGroundTruth(yolo, round)

    return {
      acq_id: k.acqC,
      sec: k.sec,
      split: k.split,
      reviewed: k.reviewed,
      image: chosen && driveId
        ? {
            drive_id: driveId,
            ext: chosen.ext,
            source_link: chosen.link,
            url_original: `https://drive.usercontent.google.com/download?id=${driveId}&export=download&confirm=t`,
            url_cdn: `https://lh3.googleusercontent.com/d/${driveId}=w1920-h1080-n`,
            width: dims?.width ?? null,
            height: dims?.height ?? null,
          }
        : null,
      // Gabarito só existe onde houve revisão humana; fora disso é null (não [] —
      // "sem anotação" é diferente de "anotado como vazio").
      ground_truth: k.reviewed ? resolved.ground_truth : null,
      curation_counts: k.reviewed ? resolved.counts : null,
      stored_detections: resolved.originals,
      telemetry: {
        gps: doc?.can?.GPS_x != null && doc?.can?.GPS_y != null
          ? { lat: doc.can.GPS_y, lon: doc.can.GPS_x, alt: doc.can.GPS_z ?? null }
          : null,
        speed: doc?.can?.VehicleSpeed ?? null,
        steering: doc?.can?.SteeringWheelAngle ?? null,
        brake: doc?.can?.BrakeInfoStatus ?? null,
        vehicle: doc?.block?.vehicle ?? null,
        period: doc?.block?.meteo?.period ?? null,
        condition: doc?.block?.meteo?.condition ?? null,
        semseg: doc?.semseg ?? null,
      },
    }
  })

  return reply.send({ ...meta, frames })
}
