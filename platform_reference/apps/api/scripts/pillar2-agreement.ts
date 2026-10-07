/**
 * Concordancia inter-anotador: round 1 (gabarito) x round 2 (segundo anotador).
 * SOMENTE LEITURA. Recusa-se a rodar enquanto a segunda anotacao nao estiver completa
 * (use --force para um parcial exploratorio).
 *
 * Uso: npx tsx --env-file=apps/api/.env apps/api/scripts/pillar2-agreement.ts [--force]
 */
import { writeFileSync, mkdirSync } from 'fs'
import prisma from '../src/lib/prisma'
import { normalizeToCompact } from '../src/controllers/collections/get-collection-seconds-with-links'
import { normalizeClass, resolveGroundTruth, type BoxN, type ResolvedOriginal, type GtBox } from '../src/controllers/collections/curation/shared'
import { iou } from '../src/controllers/collections/curation/stats'

const ROUND2_NAME = 'Pillar2 Second Annotator 63'
const OUT = 'apps/api/scripts/out/pillar2'

// Corte que separa "empurrei a caixa" de "refiz a caixa" — o MESMO de curation/stats.ts.
const IOU_MINOR = 0.7
// Corte do matching one-to-one dos objetos adicionados (sem proposal para ancorar).
const IOU_MATCH_ADDED = 0.5

const VERDICTS = ['kept', 'corrected-minor', 'corrected-redrawn', 'class-changed', 'rejected'] as const
type V = typeof VERDICTS[number]

/**
 * Dobra o veredito bruto (kept/corrected/rejected) nas 5 categorias do experimento.
 * Precedencia: erro de CLASSE domina erro de CAIXA — mesma convencao de curation/stats.ts.
 */
function verdict5(o: ResolvedOriginal): V | null {
  if (o.verdict === 'kept') return 'kept'
  if (o.verdict === 'rejected') return 'rejected'
  if (o.verdict !== 'corrected' || !o.correction) return null // unreviewed
  const a = o.class, b = o.correction.class
  if (typeof a === 'string' && typeof b === 'string' && normalizeClass(a) !== normalizeClass(b)) return 'class-changed'
  const ob = o.box_n, cb = o.correction.box_n
  return ob && cb && iou(ob, cb) >= IOU_MINOR ? 'corrected-minor' : 'corrected-redrawn'
}

// Caixa final do anotador para um original (kept => caixa do proprio original).
function finalBox(o: ResolvedOriginal): BoxN | null {
  return o.verdict === 'corrected' ? o.correction?.box_n ?? null : o.box_n
}
function finalClass(o: ResolvedOriginal): string | null {
  const c = o.verdict === 'corrected' ? o.correction?.class ?? null : o.class
  return typeof c === 'string' ? normalizeClass(c) : null
}

const round3 = (n: number) => Math.round(n * 1000) / 1000
function quantile(sorted: number[], q: number): number | null {
  if (!sorted.length) return null
  const pos = (sorted.length - 1) * q
  const lo = Math.floor(pos), hi = Math.ceil(pos)
  return lo === hi ? sorted[lo] : sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
}

/** Matching guloso one-to-one por IoU decrescente. `sameClass` restringe a mesma classe. */
function matchAdded(a: GtBox[], b: GtBox[], thr: number, sameClass: boolean) {
  const pairs: Array<{ i: number; j: number; iou: number }> = []
  a.forEach((x, i) => b.forEach((y, j) => {
    if (!x.box_n || !y.box_n) return
    if (sameClass && normalizeClass(String(x.class)) !== normalizeClass(String(y.class))) return
    const v = iou(x.box_n, y.box_n)
    if (v >= thr) pairs.push({ i, j, iou: v })
  }))
  pairs.sort((p, q) => q.iou - p.iou || p.i - q.i || p.j - q.j)
  const ua = new Set<number>(), ub = new Set<number>()
  const matched: typeof pairs = []
  for (const p of pairs) {
    if (ua.has(p.i) || ub.has(p.j)) continue
    ua.add(p.i); ub.add(p.j); matched.push(p)
  }
  return { matched, unmatchedA: a.length - ua.size, unmatchedB: b.length - ub.size }
}

async function main() {
  const force = process.argv.includes('--force')

  const col = await prisma.collection.findFirst({
    where: { name: ROUND2_NAME },
    select: { id: true, review_round: true, source_collection_id: true },
  })
  if (!col) throw new Error(`Colecao "${ROUND2_NAME}" nao encontrada.`)

  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId: col.id }, select: { acq_id: true, curationSecs: true, reviewedSecs: true },
  })
  const withCur = groups.filter((g) => g.curationSecs.length > 0)
  const totalFrames = withCur.reduce((a, g) => a + g.curationSecs.length, 0)
  const reviewedFrames = withCur.reduce((a, g) => a + g.reviewedSecs.filter((s) => g.curationSecs.includes(s)).length, 0)

  if (reviewedFrames < totalFrames && !force) {
    console.log(`Segunda anotacao incompleta: ${reviewedFrames}/${totalFrames} frames.`)
    console.log('Rode second-annotator-status.ts. Use --force para um parcial exploratorio.')
    process.exitCode = 1
    return
  }

  const docs = await prisma.big1Hz.findMany({
    where: { OR: withCur.map((g) => ({ acq_id: Number(normalizeToCompact(g.acq_id)), sec: { in: g.curationSecs } })) },
    select: { acq_id: true, sec: true, yolo: true },
  })

  // ── Proposal-level ──
  const cm: Record<string, Record<string, number>> = {}
  for (const a of VERDICTS) { cm[a] = {}; for (const b of VERDICTS) cm[a][b] = 0 }
  const margA: Record<string, number> = {}, margB: Record<string, number> = {}
  for (const v of VERDICTS) { margA[v] = 0; margB[v] = 0 }
  let comparable = 0, exact = 0, onlyA = 0, onlyB = 0
  let classComparable = 0, classAgree = 0
  let bothClassChangedN = 0, bothClassChangedAgree = 0
  const classDisagreements: Array<{ frame: string; a: string; b: string; va: string; vb: string }> = []
  const ious: number[] = []
  let addedA = 0, addedB = 0
  let mGeom = 0, mGeomClassAgree = 0, umA = 0, umB = 0, mSameClass = 0
  const addedClassDisagree: Array<{ frame: string; a: string; b: string; iou: number }> = []

  for (const d of docs) {
    const frameKey = `${d.acq_id}:${d.sec}`
    const r1 = resolveGroundTruth(d.yolo as any, 1)
    const r2 = resolveGroundTruth(d.yolo as any, 2)
    const byId2 = new Map(r2.originals.map((o) => [o.det_id, o]))

    for (const o1 of r1.originals) {
      const o2 = o1.det_id ? byId2.get(o1.det_id) : undefined
      const v1 = verdict5(o1), v2 = o2 ? verdict5(o2) : null
      if (v1 && !v2) onlyA++
      if (!v1 && v2) onlyB++
      if (!v1 || !v2 || !o2) continue

      comparable++
      cm[v1][v2]++
      margA[v1]++; margB[v2]++
      if (v1 === v2) exact++

      // Classe final — so onde ambos mantiveram o objeto (rejected nao tem classe final).
      if (v1 !== 'rejected' && v2 !== 'rejected') {
        const c1 = finalClass(o1), c2 = finalClass(o2)
        if (c1 && c2) {
          classComparable++
          if (c1 === c2) classAgree++
          else classDisagreements.push({ frame: frameKey, a: c1, b: c2, va: v1, vb: v2 })
        }
        if (v1 === 'class-changed' && v2 === 'class-changed') {
          bothClassChangedN++
          if (c1 === c2) bothClassChangedAgree++
        }
      }

      // Geometria — so onde AMBOS redesenharam a caixa.
      if (o1.verdict === 'corrected' && o2.verdict === 'corrected') {
        const b1 = finalBox(o1), b2 = finalBox(o2)
        if (b1 && b2) ious.push(iou(b1, b2))
      }
    }

    // ── Objetos adicionados (sem ref_det_id) — avaliados a parte, fora do kappa ──
    const a1 = r1.ground_truth.filter((g) => g.origin === 'added')
    const a2 = r2.ground_truth.filter((g) => g.origin === 'added')
    addedA += a1.length; addedB += a2.length

    // Primario: matching GEOMETRICO (sem exigir classe), para que a classe possa ser
    // MEDIDA nos pares. Exigir mesma classe no matching tornaria o agreement de classe
    // 100% por construcao — nao mediria nada.
    const mg = matchAdded(a1, a2, IOU_MATCH_ADDED, false)
    mGeom += mg.matched.length; umA += mg.unmatchedA; umB += mg.unmatchedB
    for (const p of mg.matched) {
      const ca = normalizeClass(String(a1[p.i].class)), cb = normalizeClass(String(a2[p.j].class))
      if (ca === cb) mGeomClassAgree++
      else addedClassDisagree.push({ frame: frameKey, a: ca, b: cb, iou: round3(p.iou) })
    }
    // Secundario: matching com a restricao de mesma classe (a regra do enunciado).
    mSameClass += matchAdded(a1, a2, IOU_MATCH_ADDED, true).matched.length
  }

  // Cohen's kappa sobre os 5 vereditos + erro-padrao (aproximacao normal de amostra grande).
  const po = comparable ? exact / comparable : 0
  let pe = 0
  for (const v of VERDICTS) pe += (margA[v] / (comparable || 1)) * (margB[v] / (comparable || 1))
  const kappa = pe < 1 ? (po - pe) / (1 - pe) : null
  const se = comparable && pe < 1 ? Math.sqrt((po * (1 - po)) / (comparable * (1 - pe) ** 2)) : null
  const ci = kappa != null && se != null
    ? { lo: round3(kappa - 1.96 * se), hi: round3(kappa + 1.96 * se), method: 'aproximacao normal de amostra grande (kappa +- 1.96*EP)' }
    : null

  const sorted = [...ious].sort((a, b) => a - b)
  const geomPct = (n: number) => (ious.length ? round3((n / ious.length) * 100) : null)

  const report = {
    generated_at: new Date().toISOString(),
    collection_round2: col.id, collection_round1: col.source_collection_id,
    complete: reviewedFrames === totalFrames,
    frames: { total: totalFrames, reviewed_round2: reviewedFrames },
    thresholds: {
      iou_minor: IOU_MINOR, iou_match_added: IOU_MATCH_ADDED,
      note: 'class-changed tem precedencia sobre corrected-minor/redrawn (mesma convencao de curation/stats.ts)',
    },
    proposal_level: {
      comparable_n: comparable, decided_only_annotator1: onlyA, decided_only_annotator2: onlyB,
      exact_agreement: exact, percent_agreement: comparable ? round3((exact / comparable) * 100) : null,
      cohens_kappa: kappa == null ? null : round3(kappa),
      kappa_se: se == null ? null : round3(se),
      kappa_ci95: ci,
      p_observed: round3(po), p_expected: round3(pe),
      verdict_order: VERDICTS,
      marginals_annotator1: margA, marginals_annotator2: margB,
      confusion_matrix_5x5: cm,
    },
    class_level: {
      comparable_n: classComparable, agreement: classAgree,
      percent: classComparable ? round3((classAgree / classComparable) * 100) : null,
      class_changes_annotator1: margA['class-changed'], class_changes_annotator2: margB['class-changed'],
      both_class_changed_n: bothClassChangedN, both_class_changed_agree: bothClassChangedAgree,
      disagreements: classDisagreements,
    },
    geometry: {
      both_corrected_n: ious.length,
      mean_iou: ious.length ? round3(ious.reduce((a, b) => a + b, 0) / ious.length) : null,
      median_iou: ious.length ? round3(quantile(sorted, 0.5)!) : null,
      q1_iou: ious.length ? round3(quantile(sorted, 0.25)!) : null,
      q3_iou: ious.length ? round3(quantile(sorted, 0.75)!) : null,
      min_iou: ious.length ? round3(sorted[0]) : null,
      max_iou: ious.length ? round3(sorted[sorted.length - 1]) : null,
      n_iou_ge_050: ious.filter((v) => v >= 0.5).length,
      pct_iou_ge_050: geomPct(ious.filter((v) => v >= 0.5).length),
      n_iou_ge_070: ious.filter((v) => v >= 0.7).length,
      pct_iou_ge_070: geomPct(ious.filter((v) => v >= 0.7).length),
      distribution: {
        '<0.5': ious.filter((v) => v < 0.5).length,
        '0.5-0.7': ious.filter((v) => v >= 0.5 && v < 0.7).length,
        '0.7-0.9': ious.filter((v) => v >= 0.7 && v < 0.9).length,
        '>=0.9': ious.filter((v) => v >= 0.9).length,
      },
    },
    added_objects: {
      annotator1: addedA, annotator2: addedB,
      matched_geometric: mGeom, unmatched_annotator1: umA, unmatched_annotator2: umB,
      class_agreement_in_matched: mGeomClassAgree,
      class_agreement_pct: mGeom ? round3((mGeomClassAgree / mGeom) * 100) : null,
      class_disagreements: addedClassDisagree,
      matched_same_class_constrained: mSameClass,
      match_rule: `one-to-one guloso por IoU decrescente, IoU >= ${IOU_MATCH_ADDED}; matching PRIMARIO so por geometria (a classe e medida, nao imposta); "matched_same_class_constrained" repete o matching exigindo mesma classe`,
      note: 'fora do Cohen kappa dos vereditos das proposals',
    },
  }

  mkdirSync(OUT, { recursive: true })
  writeFileSync(`${OUT}/agreement-report.json`, JSON.stringify(report, null, 2))
  console.log(JSON.stringify(report, null, 2))
  console.log(`\nrelatorio -> ${OUT}/agreement-report.json`)
}
main().finally(() => prisma.$disconnect())
