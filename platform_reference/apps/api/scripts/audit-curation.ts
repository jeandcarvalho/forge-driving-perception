/**
 * Auditoria de integridade da curadoria manual de uma coleção.
 * Percorre TODOS os frames do conjunto travado que têm entrada humana e/ou estão em
 * reviewedSecs, e checa as invariantes do modelo append-only (ver YOLO_CURATION_PLAN.md).
 *
 * Uso: npx tsx --env-file=apps/api/.env apps/api/scripts/audit-curation.ts "<nome ou id da coleção>"
 */
import prisma from '../src/lib/prisma'
import { normalizeToCompact } from '../src/controllers/collections/get-collection-seconds-with-links'
import { isHuman, isValidClass, RawDet, serializeDetections } from '../src/controllers/collections/curation/shared'

const ACTIONS = new Set(['kept', 'corrected', 'added', 'rejected'])

type Problem = { where: string; msg: string }
const problems: Problem[] = []
const warn: Problem[] = []
const bad = (where: string, msg: string) => problems.push({ where, msg })

function checkFrame(where: string, arr: RawDet[]) {
  const ser = serializeDetections(arr)

  // det_id único no array
  const seen = new Map<string, number>()
  arr.forEach((d, i) => {
    if (!d?.det_id) return
    if (seen.has(d.det_id)) bad(where, `det_id duplicado ${d.det_id} (idx ${seen.get(d.det_id)} e ${i})`)
    seen.set(d.det_id, i)
  })

  // Humanas por original referenciado (deve haver no máx. 1 — dropPriorHumanRef)
  const perRef = new Map<string, number>()

  for (const h of ser.human) {
    const raw = arr[h.idx]
    if (!ACTIONS.has(h.action)) bad(where, `idx ${h.idx}: action desconhecida "${h.action}"`)
    if (raw.conf !== null && raw.conf !== undefined) bad(where, `idx ${h.idx}: humana com conf=${raw.conf} (deveria ser null)`)
    if (!raw.reviewed_by) bad(where, `idx ${h.idx}: humana sem reviewed_by`)
    if (!raw.reviewed_at) bad(where, `idx ${h.idx}: humana sem reviewed_at`)
    if (!raw.det_id) bad(where, `idx ${h.idx}: humana sem det_id`)
    if (h.action !== 'rejected' && !isValidClass(h.class)) bad(where, `idx ${h.idx}: classe inválida "${h.class}"`)

    const expectedSource = h.action === 'added' ? 'human_added' : 'human_corrected'
    if (raw.source !== expectedSource) bad(where, `idx ${h.idx}: source "${raw.source}" ≠ "${expectedSource}" p/ action ${h.action}`)

    if (h.action === 'added') {
      if (raw.ref_det_id != null) bad(where, `idx ${h.idx}: added com ref_det_id`)
    } else {
      if (!raw.ref_det_id) { bad(where, `idx ${h.idx}: ${h.action} sem ref_det_id`); continue }
      perRef.set(raw.ref_det_id, (perRef.get(raw.ref_det_id) ?? 0) + 1)
      if (h.ref_idx == null) bad(where, `idx ${h.idx}: ref_det_id ${raw.ref_det_id} não resolve p/ nenhuma entrada`)
      else if (isHuman(arr[h.ref_idx])) bad(where, `idx ${h.idx}: ref aponta p/ outra humana (idx ${h.ref_idx})`)
      else if (arr[h.ref_idx].reviewed !== true) bad(where, `idx ${h.idx}: original idx ${h.ref_idx} referenciado mas reviewed≠true`)
    }

    // Geometria: corrected/added exigem box válido; rejected/kept não têm geometria.
    if (h.action === 'corrected' || h.action === 'added') {
      const b = h.box_n
      if (!b) { bad(where, `idx ${h.idx}: ${h.action} sem box_n`); continue }
      if (!(b.x1 >= 0 && b.x1 < b.x2 && b.x2 <= 1 && b.y1 >= 0 && b.y1 < b.y2 && b.y2 <= 1)) {
        bad(where, `idx ${h.idx}: box_n fora de [0,1] ou invertido: ${JSON.stringify(b)}`)
      }
      // px derivado deve ser coerente com o normalizado (se gravado)
      if (typeof raw.x1 === 'number' && typeof raw.w === 'number') {
        const w = raw.x2 - raw.x1
        if (Math.abs(w - raw.w) > 1e-6) bad(where, `idx ${h.idx}: w=${raw.w} ≠ x2-x1=${w}`)
        const imgW = raw.x1 / (b.x1 || 1e-9)
        const imgW2 = raw.x2 / (b.x2 || 1e-9)
        if (b.x1 > 0.01 && Math.abs(imgW - imgW2) > 2) bad(where, `idx ${h.idx}: px/normalizado incoerente (img_w implícita ${imgW.toFixed(1)} vs ${imgW2.toFixed(1)})`)
      }
    } else if (h.box_n) {
      bad(where, `idx ${h.idx}: ${h.action} não deveria ter geometria própria`)
    }
  }

  for (const [ref, n] of perRef) if (n > 1) bad(where, `original det_id ${ref} tem ${n} humanas referenciando (esperado 1)`)

  // Originais marcados reviewed sem nenhuma humana apontando (órfão)
  for (const m of ser.model) {
    if (m.reviewed && !(m.det_id && perRef.has(m.det_id))) {
      bad(where, `idx ${m.idx}: original reviewed=true sem entrada humana referenciando`)
    }
    if (m.reviewed && !arr[m.idx].reviewed_by) bad(where, `idx ${m.idx}: original reviewed sem reviewed_by`)
    // Geometria original intacta?
    if (m.box_n == null && m.class != null) warn.push({ where, msg: `idx ${m.idx}: original "${m.class}" sem box_n (provável não-match do backfill)` })
  }

  return ser
}

async function main() {
  const key = process.argv[2] ?? 'All perception in one'
  const col = await prisma.collection.findFirst({
    where: /^[0-9a-f]{24}$/i.test(key) ? { id: key } : { name: key },
    select: {
      id: true, name: true, curation_locked: true, curation_frame_count: true,
      curation_min_distance_m: true, curation_generated_at: true, curation_generated_by: true,
    },
  })
  if (!col) { console.log(`Coleção não encontrada: ${key}`); return }

  console.log(`Coleção: "${col.name}" (${col.id})`)
  console.log(`  travada: ${col.curation_locked} | frames travados: ${col.curation_frame_count} | d_intra: ${col.curation_min_distance_m?.toFixed(1)} m`)
  console.log(`  gerado em ${col.curation_generated_at?.toISOString()} por ${col.curation_generated_by}`)

  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId: col.id },
    select: { acq_id: true, secs: true, curationSecs: true, reviewedSecs: true },
  })

  const curationTotal = groups.reduce((s, g) => s + g.curationSecs.length, 0)
  const reviewedTotal = groups.reduce((s, g) => s + g.reviewedSecs.length, 0)
  console.log(`  grupos: ${groups.length} | curationSecs: ${curationTotal} | reviewedSecs: ${reviewedTotal}`)
  if (col.curation_frame_count != null && col.curation_frame_count !== curationTotal) {
    bad('coleção', `curation_frame_count=${col.curation_frame_count} ≠ soma curationSecs=${curationTotal}`)
  }

  // reviewedSecs ⊆ curationSecs, e curationSecs ⊆ secs
  const lockedByAcq = new Map<number, Set<number>>()
  for (const g of groups) {
    const cs = new Set(g.curationSecs)
    const ss = new Set(g.secs)
    for (const s of g.curationSecs) if (!ss.has(s)) bad(`acq ${g.acq_id}`, `curationSec ${s} não está em secs`)
    for (const s of g.reviewedSecs) if (!cs.has(s)) bad(`acq ${g.acq_id}`, `reviewedSec ${s} fora de curationSecs`)
    const n = Number(normalizeToCompact(g.acq_id))
    if (!isNaN(n) && cs.size) lockedByAcq.set(n, cs)
  }

  // ── Varredura GLOBAL: qualquer frame do big_1hz com entrada humana ──
  const raw = (await prisma.big1Hz.findRaw({
    filter: { yolo: { $elemMatch: { action: { $exists: true, $ne: null } } } },
    options: { projection: { acq_id: 1, sec: 1, yolo: 1 } },
  })) as any as Array<{ acq_id: number; sec: number; yolo: RawDet[] }>

  console.log(`\nFrames no banco com curadoria humana (varredura global): ${raw.length}`)

  const reviewedIndex = new Map<number, Set<number>>()
  for (const g of groups) {
    const n = Number(normalizeToCompact(g.acq_id))
    if (!isNaN(n) && g.reviewedSecs.length) reviewedIndex.set(n, new Set(g.reviewedSecs))
  }

  let totKept = 0, totCorr = 0, totAdd = 0, totRej = 0, totModel = 0
  const rows: string[] = []
  const framesWithHuman = new Set<string>()

  for (const d of raw) {
    const where = `${d.acq_id}/${d.sec}`
    framesWithHuman.add(where)
    const locked = lockedByAcq.get(d.acq_id)
    const inThisCollection = locked?.has(d.sec) ?? false
    if (!inThisCollection) {
      warn.push({ where, msg: 'frame com curadoria humana FORA do conjunto travado desta coleção (outra coleção?)' })
      continue
    }
    const ser = checkFrame(where, d.yolo ?? [])
    const c = { kept: 0, corrected: 0, added: 0, rejected: 0 } as Record<string, number>
    for (const h of ser.human) c[h.action] = (c[h.action] ?? 0) + 1
    totKept += c.kept; totCorr += c.corrected; totAdd += c.added; totRej += c.rejected
    totModel += ser.yolo_count_model

    const done = reviewedIndex.get(d.acq_id)?.has(d.sec) ?? false
    const pend = ser.model.filter((m) => !m.reviewed).length
    if (done && pend > 0) warn.push({ where, msg: `marcado como revisado mas ${pend} original(is) sem ação humana` })
    if (!done) warn.push({ where, msg: 'tem curadoria humana mas NÃO está em reviewedSecs (não marcado como concluído)' })
    rows.push(
      `  ${where.padEnd(20)} orig=${String(ser.yolo_count_model).padStart(2)} ` +
      `kept=${c.kept} corr=${c.corrected} add=${c.added} rej=${c.rejected} ` +
      `pendentes=${pend} ${done ? '✔ concluído' : '… em aberto'}`,
    )
  }

  console.log(rows.join('\n'))
  console.log(`\nTotais nos frames curados: originais=${totModel} kept=${totKept} corrected=${totCorr} added=${totAdd} rejected=${totRej}`)

  // Frames marcados como revisados que NÃO têm nenhuma entrada humana
  for (const [acq, secs] of reviewedIndex) {
    for (const s of secs) {
      if (!framesWithHuman.has(`${acq}/${s}`)) {
        warn.push({ where: `${acq}/${s}`, msg: 'está em reviewedSecs mas não tem nenhuma entrada humana (frame vazio marcado concluído)' })
      }
    }
  }

  console.log(`\n${'='.repeat(60)}`)
  if (problems.length === 0) console.log('✅ NENHUMA violação de invariante encontrada.')
  else {
    console.log(`❌ ${problems.length} PROBLEMA(S):`)
    for (const p of problems) console.log(`  [${p.where}] ${p.msg}`)
  }
  if (warn.length) {
    console.log(`\nℹ️  ${warn.length} observação(ões):`)
    for (const p of warn.slice(0, 40)) console.log(`  [${p.where}] ${p.msg}`)
    if (warn.length > 40) console.log(`  … +${warn.length - 40}`)
  }
}

main().then(() => prisma.$disconnect()).catch(async (e) => { console.error(e); await prisma.$disconnect(); process.exit(1) })
