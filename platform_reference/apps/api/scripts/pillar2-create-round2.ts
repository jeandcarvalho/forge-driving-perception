/**
 * Cria a colecao "Pillar2 Second Annotator 63" (review_round = 2) com 63 dos 313
 * frames travados, para uma SEGUNDA anotacao humana independente e CEGA.
 *
 * NAO copia nem toca nenhuma entrada humana: as anotacoes vivem em Big1Hz.yolo e
 * permanecem intactas. O isolamento e o campo `review_round` nas entradas humanas —
 * ver shared.ts. Este script so escreve em Collection e collection_groups.
 *
 * Uso:
 *   npx tsx --env-file=apps/api/.env apps/api/scripts/pillar2-create-round2.ts          # dry-run
 *   npx tsx --env-file=apps/api/.env apps/api/scripts/pillar2-create-round2.ts --commit
 */
import { writeFileSync } from 'fs'
import prisma from '../src/lib/prisma'
import { normalizeToCompact } from '../src/controllers/collections/get-collection-seconds-with-links'
import { isHuman, RawDet } from '../src/controllers/collections/curation/shared'
import { SEED, TARGET_FRAMES, selectSecondRound, type Frame } from './pillar2-select'

const SOURCE_COLLECTION = 'All perception in one'
export const ROUND2_NAME = 'Pillar2 Second Annotator 63'
const OUT = 'apps/api/scripts/out/pillar2'

async function main() {
  const commit = process.argv.includes('--commit')

  const src = await prisma.collection.findFirst({
    where: { name: SOURCE_COLLECTION },
    select: { id: true, name: true, userId: true, curation_locked: true, curation_min_distance_m: true, gps_spacing_m: true },
  })
  if (!src) throw new Error(`Coleção "${SOURCE_COLLECTION}" não encontrada.`)
  if (!src.curation_locked) throw new Error('Coleção de origem não está travada — abortado.')

  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId: src.id },
    select: { acq_id: true, curationSecs: true },
  })
  const withCur = groups.filter((g) => g.curationSecs.length > 0)
  const total = withCur.reduce((a, g) => a + g.curationSecs.length, 0)
  console.log(`Origem: ${src.name} (${src.id}) — ${total} frames / ${withCur.length} aquisições`)

  // Classes das PROPOSALS (originais do modelo). Nenhuma entrada humana é lida.
  const docs = await prisma.big1Hz.findMany({
    where: { OR: withCur.map((g) => ({ acq_id: Number(normalizeToCompact(g.acq_id)), sec: { in: g.curationSecs } })) },
    select: { acq_id: true, sec: true, yolo: true },
  })
  const propByKey = new Map<string, string[]>()
  for (const d of docs) {
    const arr = ((d.yolo as any) ?? []) as RawDet[]
    propByKey.set(`${d.acq_id}:${d.sec}`, arr.filter((x) => !isHuman(x)).map((x) => String(x.class ?? 'null')))
  }

  const all: Frame[] = []
  for (const g of withCur) {
    const acqC = normalizeToCompact(g.acq_id)
    for (const sec of g.curationSecs) {
      all.push({ acqC, sec, classes: propByKey.get(`${Number(acqC)}:${sec}`) ?? [] })
    }
  }
  if (all.length !== total) throw new Error('Inconsistência ao montar o pool.')

  const { selected, acqCount, repairs } = selectSecondRound(all, TARGET_FRAMES)
  console.log(`\nSELEÇÃO: ${selected.length} frames / ${acqCount} aquisições distintas  (seed="${SEED}")`)
  if (repairs.length) console.log('reparos de classe:', repairs)

  // Auditoria de proposals/classes (não usada como critério de dificuldade).
  const clsCount: Record<string, number> = {}
  let proposals = 0
  const perAcq: Record<string, number> = {}
  for (const f of selected) {
    proposals += f.classes.length
    perAcq[f.acqC] = (perAcq[f.acqC] ?? 0) + 1
    for (const c of f.classes) clsCount[c] = (clsCount[c] ?? 0) + 1
  }
  const framesNoProposal = selected.filter((f) => f.classes.length === 0).length
  console.log(`proposals YOLO no subconjunto = ${proposals}  (frames sem proposal: ${framesNoProposal})`)
  console.log('classes das proposals:', Object.entries(clsCount).sort((a, b) => b[1] - a[1]))
  console.log('máx. frames por aquisição:', Math.max(...Object.values(perAcq)))

  const bySec = new Map<string, number[]>()
  for (const f of selected) {
    if (!bySec.has(f.acqC)) bySec.set(f.acqC, [])
    bySec.get(f.acqC)!.push(f.sec)
  }

  const manifest = {
    generated_at: new Date().toISOString(),
    seed: SEED,
    algorithm: 'round-robin sobre aquisições ordenadas por sha256(seed:acq); frame por sha256(seed:acq:sec); reparo car/truck',
    source_collection: { id: src.id, name: src.name, frames: total, acquisitions: withCur.length },
    target_frames: TARGET_FRAMES,
    selected_frames: selected.length,
    selected_acquisitions: acqCount,
    repairs,
    audit: { proposals, frames_without_proposal: framesNoProposal, proposal_classes: clsCount },
    frames: selected.map((f) => ({ acq_id: f.acqC, sec: f.sec, proposal_classes: f.classes })),
  }
  writeFileSync(`${OUT}/round2-selection.json`, JSON.stringify(manifest, null, 2))
  console.log(`\nmanifesto -> ${OUT}/round2-selection.json`)

  if (!commit) {
    console.log('\nDRY-RUN. Nada foi escrito. Rode com --commit para criar a coleção.')
    return
  }

  const existing = await prisma.collection.findFirst({ where: { name: ROUND2_NAME, userId: src.userId }, select: { id: true } })
  if (existing) {
    console.log(`\nJá existe "${ROUND2_NAME}" (${existing.id}). Nada a fazer — o script não sobrescreve.`)
    return
  }

  const created = await prisma.collection.create({
    data: {
      userId: src.userId,
      name: ROUND2_NAME,
      description: `Segunda anotação independente (round 2) de ${selected.length} dos ${total} frames de "${src.name}". Seed: ${SEED}. Anotador cego ao round 1.`,
      gps_spacing_m: src.gps_spacing_m,
      framesCount: selected.length,
      train_acq_ids: [],
      test_acq_ids: [...bySec.keys()].sort(),
      curation_locked: true,
      curation_min_distance_m: src.curation_min_distance_m,
      curation_frame_count: selected.length,
      curation_generated_at: new Date(),
      curation_generated_by: `pillar2-second-annotator (seed=${SEED})`,
      review_round: 2,
      source_collection_id: src.id,
    },
    select: { id: true },
  })

  for (const [acqC, secs] of [...bySec.entries()].sort()) {
    const sorted = [...secs].sort((a, b) => a - b)
    await prisma.collectionGroup.create({
      data: { collectionId: created.id, acq_id: acqC, secs: sorted, activeSecs: sorted, curationSecs: sorted, reviewedSecs: [] },
    })
  }

  console.log(`\nCOLEÇÃO CRIADA: ${ROUND2_NAME}`)
  console.log(`id = ${created.id}`)
  console.log(`UI: /collections/${created.id}/curation`)
}
main().finally(() => prisma.$disconnect())
