import { FastifyRequest, FastifyReply } from 'fastify'
import prisma from '../../lib/prisma'

function haversineMeters(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6_371_000
  const toRad = (d: number) => (d * Math.PI) / 180
  const dLat = toRad(lat2 - lat1)
  const dLon = toRad(lon2 - lon1)
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

function normalizeToCompact(acqId: string): string {
  if (!acqId) return acqId
  if (/^\d{14}$/.test(acqId)) return acqId
  const match = acqId.match(/(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})/)
  if (match) return match.slice(1).join('')
  const digits = acqId.replace(/\D/g, '')
  return digits.length === 14 ? digits : acqId
}

export async function applySpacing(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id
  const { collectionId } = request.params as { collectionId: string }
  const body = request.body as { min_distance_m?: number | null }
  const min_distance_m = body?.min_distance_m ?? null

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: { id: true, framesCount: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  // Clear filter: reset activeSecs to [] on all groups
  if (min_distance_m === null) {
    await prisma.collectionGroup.updateMany({
      where: { collectionId },
      data: { activeSecs: [] },
    })
    await prisma.collection.update({
      where: { id: collectionId },
      data: { gps_spacing_m: null },
    })
    return reply.send({ active: collection.framesCount, total: collection.framesCount, gps_spacing_m: null })
  }

  if (typeof min_distance_m !== 'number' || min_distance_m < 10) {
    return reply.status(400).send({ error: 'min_distance_m deve ser um número >= 10.' })
  }

  // Load groups — data is already grouped by acq_id
  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId },
    select: { id: true, acq_id: true, secs: true },
  })

  // Build acqId (numeric) → secs map for GPS lookup
  const acqSecMap = new Map<number, number[]>()
  for (const group of groups) {
    const canonical = normalizeToCompact(group.acq_id)
    const n = Number(canonical)
    if (isNaN(n)) continue
    acqSecMap.set(n, group.secs)
  }

  const gpsDocs = acqSecMap.size === 0 ? [] : await prisma.big1Hz.findMany({
    where: {
      OR: [...acqSecMap.entries()].map(([acq_id, secs]) => ({ acq_id, sec: { in: secs } })),
    },
    select: { acq_id: true, sec: true, can: true },
  })

  const gpsMap = new Map<string, Map<number, { lat: number; lon: number }>>()
  for (const doc of gpsDocs) {
    if (!doc.acq_id || !doc.can?.GPS_x || !doc.can?.GPS_y) continue
    const key = String(doc.acq_id)
    if (!gpsMap.has(key)) gpsMap.set(key, new Map())
    gpsMap.get(key)!.set(doc.sec, { lat: doc.can.GPS_y, lon: doc.can.GPS_x })
  }

  // Apply Haversine per group and persist activeSecs
  let totalActive = 0
  const totalAll = collection.framesCount

  for (const group of groups) {
    const canonical = normalizeToCompact(group.acq_id)
    const secGps = gpsMap.get(canonical)
    const sorted = [...group.secs].sort((a, b) => a - b)
    const activeSecs: number[] = []
    let lastGps: { lat: number; lon: number } | null = null

    for (const sec of sorted) {
      const gps = secGps?.get(sec)
      if (!gps) {
        activeSecs.push(sec)
        continue
      }
      if (!lastGps) {
        activeSecs.push(sec)
        lastGps = gps
        continue
      }
      if (haversineMeters(lastGps.lat, lastGps.lon, gps.lat, gps.lon) >= min_distance_m) {
        activeSecs.push(sec)
        lastGps = gps
      }
    }

    await prisma.collectionGroup.update({
      where: { id: group.id },
      data: { activeSecs },
    })
    totalActive += activeSecs.length
  }

  await prisma.collection.update({
    where: { id: collectionId },
    data: { gps_spacing_m: min_distance_m },
  })

  return reply.send({ active: totalActive, total: totalAll, gps_spacing_m: min_distance_m })
}
