import { FastifyRequest, FastifyReply } from 'fastify'
import prisma from '../../lib/prisma'

type DistEntry = { label: string; count: number; pct: number }

function buildDistribution(values: (string | null | undefined)[], matchedTotal: number): DistEntry[] {
  const counts = new Map<string, number>()
  for (const v of values) {
    const label = v?.trim() || 'unknown'
    counts.set(label, (counts.get(label) ?? 0) + 1)
  }
  return [...counts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([label, count]) => ({
      label,
      count,
      pct: matchedTotal > 0 ? Math.round((count / matchedTotal) * 100) : 0,
    }))
}

function speedBucket(speed: number | null | undefined): string {
  if (speed == null) return 'unknown'
  if (speed < 20) return '0–20 km/h'
  if (speed < 40) return '20–40 km/h'
  if (speed < 60) return '40–60 km/h'
  if (speed < 80) return '60–80 km/h'
  if (speed < 100) return '80–100 km/h'
  return '100+ km/h'
}

// All semseg values are in 0–100 scale (% of pixels)
function buildingBucket(v: number | null | undefined): string {
  if (v == null) return 'unknown'
  if (v < 10) return '< 10%'
  if (v < 30) return '10–30%'
  if (v < 60) return '30–60%'
  return '> 60%'
}

function vegetationBucket(v: number | null | undefined): string {
  if (v == null) return 'unknown'
  if (v < 10) return '< 10%'
  if (v < 30) return '10–30%'
  if (v < 60) return '30–60%'
  return '> 60%'
}

function sidewalkBucket(left: number | null | undefined, right: number | null | undefined): string {
  // threshold=2: bimodal distribution — 0–2% is noise, above 2% is real sidewalk presence
  const THRESHOLD = 2
  return ((left ?? 0) > THRESHOLD || (right ?? 0) > THRESHOLD) ? 'Present' : 'None'
}

// Only automotive-relevant YOLO classes
const AUTOMOTIVE_CLASSES = new Set([
  'person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'traffic light', 'stop sign',
])

const STATS_SAMPLE = 1000

export async function getCollectionStats(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: { id: true, gps_spacing_m: true, framesCount: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Collection not found.' })

  return sendStats(reply, collection.id, collection.gps_spacing_m)
}

// Versão pública (leitura) — mesma resposta, sem checagem de dono. Espelha
// /public/collections/:id/view: quem tem o link vê as estatísticas.
export async function publicGetCollectionStats(request: FastifyRequest, reply: FastifyReply) {
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId },
    select: { id: true, gps_spacing_m: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Collection not found.' })

  return sendStats(reply, collection.id, collection.gps_spacing_m)
}

async function sendStats(reply: FastifyReply, collectionId: string, gpsSpacingM: number | null) {
  const hasSpacingFilter = gpsSpacingM !== null

  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId },
    select: { acq_id: true, secs: true, activeSecs: true },
  })

  const allItems: { acq_id: string; sec: number }[] = []
  const uniqueAcqIds = new Set<string>()

  for (const group of groups) {
    uniqueAcqIds.add(group.acq_id)
    const effectiveSecs =
      hasSpacingFilter && group.activeSecs.length > 0 ? group.activeSecs : group.secs
    for (const sec of effectiveSecs) {
      allItems.push({ acq_id: group.acq_id, sec })
    }
  }

  const totalFrames = allItems.length

  if (totalFrames === 0) {
    return reply.send({
      total_frames: 0,
      total_acq_ids: 0,
      gps_spacing_m: gpsSpacingM,
      has_spacing_filter: hasSpacingFilter,
      sampled: false,
      distributions: {
        period: [], condition: [], yolo_class: [], speed_kmh: [],
        semseg_building: [], semseg_vegetation: [], semseg_sidewalk: [],
      },
    })
  }

  const sampled = totalFrames > STATS_SAMPLE
  const items = sampled
    ? Array.from({ length: STATS_SAMPLE }, (_, i) =>
        allItems[Math.floor((i * totalFrames) / STATS_SAMPLE)]
      )
    : allItems

  const acqSecMap = new Map<number, number[]>()
  for (const item of items) {
    const acqId = Number(item.acq_id)
    if (isNaN(acqId)) continue
    if (!acqSecMap.has(acqId)) acqSecMap.set(acqId, [])
    acqSecMap.get(acqId)!.push(item.sec)
  }

  if (acqSecMap.size === 0) {
    return reply.send({
      total_frames: totalFrames,
      total_acq_ids: uniqueAcqIds.size,
      gps_spacing_m: gpsSpacingM,
      has_spacing_filter: hasSpacingFilter,
      sampled,
      distributions: {
        period: [], condition: [], yolo_class: [], speed_kmh: [],
        semseg_building: [], semseg_vegetation: [], semseg_sidewalk: [],
      },
    })
  }

  const telemetry = await prisma.big1Hz.findMany({
    where: {
      OR: [...acqSecMap.entries()].map(([acq_id, secs]) => ({ acq_id, sec: { in: secs } })),
    },
    select: { block: true, can: true, yolo: true, semseg: true },
  })

  const matched = telemetry.length

  // YOLO: dominant automotive class per frame (skip frames with no automotive detection)
  const yoloClassValues: string[] = []
  for (const d of telemetry) {
    if (!d.yolo || d.yolo.length === 0) continue
    const counts = new Map<string, number>()
    for (const det of d.yolo) {
      const cls = det.class?.trim().toLowerCase()
      if (cls && AUTOMOTIVE_CLASSES.has(cls)) {
        counts.set(cls, (counts.get(cls) ?? 0) + 1)
      }
    }
    if (counts.size > 0) {
      yoloClassValues.push([...counts.entries()].sort((a, b) => b[1] - a[1])[0][0])
    }
  }

  return reply.send({
    total_frames: totalFrames,
    total_acq_ids: uniqueAcqIds.size,
    gps_spacing_m: gpsSpacingM,
    has_spacing_filter: hasSpacingFilter,
    sampled,
    distributions: {
      period: buildDistribution(telemetry.map((d) => d.block?.meteo?.period), matched),
      condition: buildDistribution(telemetry.map((d) => d.block?.meteo?.condition), matched),
      yolo_class: buildDistribution(yoloClassValues, yoloClassValues.length),
      speed_kmh: buildDistribution(
        telemetry.map((d) => speedBucket(d.can?.VehicleSpeed)),
        matched
      ),
      semseg_building: buildDistribution(
        telemetry.map((d) => buildingBucket(d.semseg?.building)),
        matched
      ),
      semseg_vegetation: buildDistribution(
        telemetry.map((d) => vegetationBucket(d.semseg?.vegetation)),
        matched
      ),
      semseg_sidewalk: buildDistribution(
        telemetry.map((d) => sidewalkBucket(d.semseg?.sidewalk_left, d.semseg?.sidewalk_right)),
        matched
      ),
    },
  })
}
