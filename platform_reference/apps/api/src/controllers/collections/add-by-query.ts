import { PassThrough } from "node:stream";
import { FastifyRequest, FastifyReply } from "fastify";
import { Prisma } from "@prisma/client";
import { type SearchQuery } from "@carcara/shared";
import prisma from "../../lib/prisma";
import { buildMongoMatch } from "../../lib/build-mongo-match";

const MAX_FRAMES = 7_000;

function haversineMeters(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6_371_000;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const dLat = toRad(lat2 - lat1);
  const dLon = toRad(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

export async function addByQuery(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id;
  const { collectionId } = request.params as { collectionId: string };
  const body = request.body as { query?: Record<string, string> };
  const searchQuery = (body?.query ?? {}) as SearchQuery;

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: { id: true, framesCount: true },
  });
  if (!collection) return reply.status(404).send({ error: "Collection not found" });

  const match = buildMongoMatch(searchQuery);
  const isEmptyMatch = !match || Object.keys(match).length === 0;
  if (isEmptyMatch) return reply.send({ inserted: 0, total: 0 });

  const gpsMinDist = searchQuery["gps_min_dist"] ? Number(searchQuery["gps_min_dist"]) : null;
  const applyGpsFilter = gpsMinDist != null && Number.isFinite(gpsMinDist) && gpsMinDist > 0;

  const stream = new PassThrough();
  reply.header("Content-Type", "application/x-ndjson");
  reply.header("Cache-Control", "no-cache");
  reply.header("X-Accel-Buffering", "no");
  reply.send(stream);

  const emit = (data: object) => stream.write(JSON.stringify(data) + "\n");

  try {
    emit({ phase: "querying" });

    const pipeline: Prisma.InputJsonValue[] = [
      { $match: match },
      { $project: { _id: 0, acq_id: 1, sec: 1, ...(applyGpsFilter ? { "can.GPS_x": 1, "can.GPS_y": 1 } : {}) } },
    ];
    const raw = (await prisma.big1Hz.aggregateRaw({ pipeline })) as unknown as any[];

    // Group by acq_id — one upsert per acquisition, not per frame
    const byAcq = new Map<string, Set<number>>();
    if (applyGpsFilter) {
      // First group with GPS coords, then apply Haversine per group
      const gpsGroups = new Map<string, { sec: number; gx: number | null; gy: number | null }[]>();
      for (const doc of raw) {
        if (doc.acq_id == null || doc.sec == null) continue;
        const key = String(doc.acq_id);
        if (!gpsGroups.has(key)) gpsGroups.set(key, []);
        gpsGroups.get(key)!.push({
          sec: Number(doc.sec),
          gx: doc.can?.GPS_x ?? null,
          gy: doc.can?.GPS_y ?? null,
        });
      }
      for (const [key, rows] of gpsGroups.entries()) {
        rows.sort((a, b) => a.sec - b.sec);
        const kept = new Set<number>();
        let lastGps: { lat: number; lon: number } | null = null;
        for (const row of rows) {
          if (row.gx == null || row.gy == null) {
            kept.add(row.sec);
            continue;
          }
          if (!lastGps) {
            kept.add(row.sec);
            lastGps = { lat: row.gy, lon: row.gx };
            continue;
          }
          if (haversineMeters(lastGps.lat, lastGps.lon, row.gy, row.gx) >= gpsMinDist!) {
            kept.add(row.sec);
            lastGps = { lat: row.gy, lon: row.gx };
          }
        }
        if (kept.size > 0) byAcq.set(key, kept);
      }
    } else {
      for (const doc of raw) {
        if (doc.acq_id == null || doc.sec == null) continue;
        const key = String(doc.acq_id);
        if (!byAcq.has(key)) byAcq.set(key, new Set());
        byAcq.get(key)!.add(Number(doc.sec));
      }
    }

    if (byAcq.size === 0) {
      emit({ phase: "done", inserted: 0, total: 0 });
      stream.end();
      return;
    }

    // Fetch existing groups for these acq_ids in one query
    const existing = await prisma.collectionGroup.findMany({
      where: { collectionId, acq_id: { in: [...byAcq.keys()] } },
      select: { id: true, acq_id: true, secs: true },
    });
    const existingByAcq = new Map(existing.map((g) => [g.acq_id, g]));

    // Compute potential new frames (dedup against existing)
    let potentialNew = 0;
    for (const [acq_id, newSecs] of byAcq.entries()) {
      const ex = existingByAcq.get(acq_id);
      if (ex) {
        const existingSet = new Set(ex.secs);
        potentialNew += [...newSecs].filter((s) => !existingSet.has(s)).length;
      } else {
        potentialNew += newSecs.size;
      }
    }

    if (collection.framesCount + potentialNew > MAX_FRAMES) {
      emit({
        phase: "error",
        limit_exceeded: true,
        message: `Collection limit reached: adding ${potentialNew} frames would exceed the ${MAX_FRAMES.toLocaleString()} frame limit (currently ${collection.framesCount}).`,
      });
      stream.end();
      return;
    }

    const totalFrames = [...byAcq.values()].reduce((sum, s) => sum + s.size, 0);
    emit({ phase: "inserting", done: 0, total: totalFrames });

    let inserted = 0;
    let done = 0;

    for (const [acq_id, newSecs] of byAcq.entries()) {
      const ex = existingByAcq.get(acq_id);
      if (ex) {
        const existingSet = new Set(ex.secs);
        const toAdd = [...newSecs].filter((s) => !existingSet.has(s));
        if (toAdd.length > 0) {
          await prisma.collectionGroup.update({
            where: { id: ex.id },
            data: { secs: [...ex.secs, ...toAdd].sort((a, b) => a - b) },
          });
          inserted += toAdd.length;
        }
      } else {
        const secs = [...newSecs].sort((a, b) => a - b);
        await prisma.collectionGroup.create({
          data: { collectionId, acq_id, secs, activeSecs: [] },
        });
        inserted += secs.length;
      }
      done += newSecs.size;
      emit({ phase: "inserting", done, total: totalFrames });
    }

    if (inserted > 0) {
      await prisma.collection.update({
        where: { id: collectionId },
        data: { framesCount: { increment: inserted } },
      });
    }

    emit({ phase: "done", inserted, total: totalFrames });
  } catch (err) {
    emit({ phase: "error", message: String(err) });
  } finally {
    stream.end();
  }
}
