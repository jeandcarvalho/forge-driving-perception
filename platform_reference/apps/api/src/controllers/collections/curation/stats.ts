import { FastifyRequest, FastifyReply } from 'fastify'
import prisma from '../../../lib/prisma'
import { normalizeToCompact } from '../get-collection-seconds-with-links'
import { collectionRound, normalizeClass, resolveGroundTruth, type BoxN } from './shared'

// Acima disso a correção é um empurrãozinho na caixa; abaixo, o curador refez a caixa.
const IOU_MINOR = 0.7

export function iou(a: BoxN, b: BoxN): number {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1))
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1))
  const inter = ix * iy
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter
  return union > 0 ? inter / union : 0
}

// Estatísticas da curadoria: YOLO base × gabarito humano, contadas SÓ sobre os frames
// já revisados (o resto do conjunto ainda não tem veredito). Leitura pura.
export async function getCurationStats(request: FastifyRequest, reply: FastifyReply) {
  const userId = request.user!.id
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId, userId },
    select: { id: true, review_round: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  return sendCurationStats(reply, collectionId, collectionRound(collection.review_round))
}

// Gêmea pública (somente leitura, sem checagem de dono).
export async function publicGetCurationStats(request: FastifyRequest, reply: FastifyReply) {
  const { collectionId } = request.params as { collectionId: string }

  const collection = await prisma.collection.findFirst({
    where: { id: collectionId },
    select: { id: true, review_round: true },
  })
  if (!collection) return reply.status(404).send({ error: 'Coleção não encontrada.' })

  return sendCurationStats(reply, collectionId, collectionRound(collection.review_round))
}

async function sendCurationStats(reply: FastifyReply, collectionId: string, round: number) {
  const groups = await prisma.collectionGroup.findMany({
    where: { collectionId },
    select: { acq_id: true, curationSecs: true, reviewedSecs: true },
  })

  let totalFrames = 0
  let reviewedFrames = 0
  const acqSecMap = new Map<number, number[]>() // só os revisados

  for (const g of groups) {
    if (g.curationSecs.length === 0) continue
    totalFrames += g.curationSecs.length
    const curationSet = new Set(g.curationSecs)
    const reviewed = g.reviewedSecs.filter((s) => curationSet.has(s))
    reviewedFrames += reviewed.length

    const n = Number(normalizeToCompact(g.acq_id))
    if (isNaN(n) || reviewed.length === 0) continue
    if (!acqSecMap.has(n)) acqSecMap.set(n, [])
    acqSecMap.get(n)!.push(...reviewed)
  }

  let modelBoxes = 0
  let confirmed = 0
  let boxMinor = 0 // mesma classe, só empurrei a caixa
  let boxMajor = 0 // mesma classe, refiz a caixa (ou sem original p/ comparar)
  let classChanged = 0 // o modelo errou o QUE é
  let rejected = 0
  let added = 0

  if (acqSecMap.size > 0) {
    const docs = await prisma.big1Hz.findMany({
      where: { OR: [...acqSecMap.entries()].map(([acq_id, secs]) => ({ acq_id, sec: { in: secs } })) },
      select: { yolo: true },
    })
    for (const d of docs) {
      // Dobra canônica model × humano (fonte única em shared.ts).
      const { originals, counts } = resolveGroundTruth(d.yolo as any, round)
      modelBoxes += counts.model
      confirmed += counts.kept
      rejected += counts.rejected
      added += counts.added

      // `corrected` ainda precisa da comparação original × correção para separar
      // erro de classe de erro de caixa (e caixa empurrada de caixa refeita).
      for (const o of originals) {
        if (o.verdict !== 'corrected' || !o.correction) continue
        const sameClass =
          typeof o.class === 'string' &&
          typeof o.correction.class === 'string' &&
          normalizeClass(o.class) === normalizeClass(o.correction.class)
        if (!sameClass) { classChanged++; continue }
        const a = o.box_n
        const b = o.correction.box_n
        if (a && b && iou(a, b) >= IOU_MINOR) boxMinor++
        else boxMajor++
      }
    }
  }

  // ── Avaliação do detector contra o gabarito humano, no IoU de corte ──
  // Detecções do modelo nos frames revisados (toda original recebeu veredito).
  const detections = confirmed + boxMinor + boxMajor + classChanged + rejected
  // TP: classe correta E bbox dentro do corte de IoU (intacta = IoU 1 por definição).
  const tp = confirmed + boxMinor
  // FP decomposto no estilo TIDE: localização, classificação e fundo.
  const fpLocalization = boxMajor // classe correta, IoU < corte
  const fpClassification = classChanged // classe incorreta
  const fpBackground = rejected // não havia instância ali
  const fp = fpLocalization + fpClassification + fpBackground
  // Gabarito = instâncias reais: as que o modelo cobriu (bem ou mal) + as anotadas do zero.
  const groundTruth = detections - fpBackground + added
  // FN: instância do gabarito sem detecção válida — nunca detectada, ou coberta só por
  // uma detecção que virou FP (erro de localização/classe).
  const fnUndetected = added
  const fnFromErrors = fpLocalization + fpClassification
  const fn = fnUndetected + fnFromErrors

  const ratio = (n: number, total: number) =>
    total > 0 ? Math.round((n / total) * 1000) / 10 : null
  const precision = ratio(tp, detections)
  const recall = ratio(tp, groundTruth)
  const f1 =
    precision != null && recall != null && precision + recall > 0
      ? Math.round(((2 * precision * recall) / (precision + recall)) * 10) / 10
      : null

  return reply.send({
    total_frames: totalFrames,
    reviewed_frames: reviewedFrames,
    iou_threshold: IOU_MINOR,

    detections,
    ground_truth: groundTruth,
    tp,
    fp,
    fn,
    precision,
    recall,
    f1,

    // Decomposição dos erros
    fp_localization: fpLocalization,
    fp_classification: fpClassification,
    fp_background: fpBackground,
    fn_undetected: fnUndetected,
    fn_from_errors: fnFromErrors,

    // Contagens cruas das ações de curadoria
    kept: confirmed,
    corrected_box_minor: boxMinor,
    corrected_box_major: boxMajor,
    corrected_class: classChanged,
    rejected,
    added,
  })
}
