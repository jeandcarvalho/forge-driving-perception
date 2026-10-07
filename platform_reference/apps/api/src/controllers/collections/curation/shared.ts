import { compactToRecorder } from '../get-collection-seconds-with-links'

export const IMAGE_EXTS = new Set(['jpg', 'jpeg', 'png', 'webp', 'gif'])

// acq_id em CollectionGroup/Links pode estar compacto (14 díg.) ou "Recorder_...".
// Retorna as duas variantes (sem nulos) para busca. Em Big1Hz o acq_id é Int.
export function recorderVariants(acqCompact: string): string[] {
  const out = [acqCompact]
  const rec = compactToRecorder(acqCompact)
  if (rec) out.push(rec)
  return out
}

// 80 classes COCO (YOLO11x). Ground truth humana precisa de classe válida.
export const COCO_CLASSES = new Set([
  'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
  'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
  'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
  'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
  'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
  'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
  'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
  'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
  'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
])

export function isValidClass(cls: unknown): cls is string {
  return typeof cls === 'string' && COCO_CLASSES.has(cls.trim().toLowerCase())
}

export function normalizeClass(cls: string): string {
  return cls.trim().toLowerCase()
}

export type BoxN = { x1: number; y1: number; x2: number; y2: number }

export function isValidBoxN(b: unknown): b is BoxN {
  if (!b || typeof b !== 'object') return false
  const { x1, y1, x2, y2 } = b as Record<string, unknown>
  const nums = [x1, y1, x2, y2]
  if (!nums.every((n) => typeof n === 'number' && Number.isFinite(n))) return false
  const bx = b as BoxN
  return bx.x1 >= 0 && bx.x1 < bx.x2 && bx.x2 <= 1 && bx.y1 >= 0 && bx.y1 < bx.y2 && bx.y2 <= 1
}

// Uma detecção crua do array Big1Hz.yolo (campos livres, mutável no RMW).
export type RawDet = Record<string, any>

// Classificação canônica é por `action` (NÃO por `source`): toda entrada humana grava
// action (kept/corrected/added/rejected); originais do modelo nunca gravam action.
// Manter action-based — source é redundante e pode assumir menos valores que action.
export function isHuman(det: RawDet): boolean {
  return det?.action != null
}

// ─────────────────────────────────────────────────────────────────────────────
// Rodadas de revisao (review_round)
//
// As entradas humanas vivem DENTRO de Big1Hz.yolo, que e global por (acq_id, sec) —
// nao por colecao. Duplicar a colecao NAO isola um segundo anotador: dropPriorHumanRef
// apagaria a entrada do primeiro. O round e o namespace que separa os anotadores.
//
// Entradas gravadas antes da existencia do campo nao tem `review_round` → sao round 1.
// ─────────────────────────────────────────────────────────────────────────────
export const ROUND_DEFAULT = 1

export function roundOf(det: RawDet): number {
  const r = det?.review_round
  return typeof r === 'number' && Number.isInteger(r) ? r : ROUND_DEFAULT
}

// Normaliza o round de uma colecao (null legado == round 1).
export function collectionRound(r: number | null | undefined): number {
  return typeof r === 'number' && Number.isInteger(r) && r > 0 ? r : ROUND_DEFAULT
}

// Entrada humana PERTENCENTE a rodada pedida. Toda leitura e toda escrita passam por
// aqui: e o unico ponto que impede um anotador de ver ou apagar o outro.
export function isHumanOfRound(det: RawDet, round: number): boolean {
  return isHuman(det) && roundOf(det) === round
}

// Extrai box normalizado de uma detecção crua (px absoluto é derivado, não canônico).
export function boxNOf(det: RawDet): BoxN | null {
  const { x1_n, y1_n, x2_n, y2_n } = det
  if ([x1_n, y1_n, x2_n, y2_n].every((n) => typeof n === 'number')) {
    return { x1: x1_n, y1: y1_n, x2: x2_n, y2: y2_n }
  }
  return null
}

export type SerializedDetections = {
  model: Array<{
    idx: number
    det_id: string | null
    class: string | null
    conf: number | null
    box_n: BoxN | null
    track_id: number | null
    dist_m: number | null
    reviewed: boolean
    rejected: boolean
    kept: boolean
    superseded_by: number | null // idx da correção humana, se houver
  }>
  human: Array<{
    idx: number
    det_id: string | null
    class: string | null
    action: string
    ref_idx: number | null // idx do original referenciado (null p/ "added")
    box_n: BoxN | null
  }>
  yolo_count_model: number
}

// Fonte única de verdade do `idx`: posição real no array bruto `yolo`.
// Reusado por B (GET frame) e pela resposta do PATCH (C).
export function serializeDetections(
  yolo: RawDet[] | null | undefined,
  round: number = ROUND_DEFAULT,
): SerializedDetections {
  const arr = yolo ?? []

  // Mapa det_id → idx (para resolver ref_det_id → ref_idx nas humanas).
  const detIdToIdx = new Map<string, number>()
  arr.forEach((det, idx) => {
    if (det?.det_id) detIdToIdx.set(det.det_id, idx)
  })

  // Estado derivado dos originais: varre humanas DESTA RODADA que referenciam cada
  // original. Entradas de outra rodada sao invisiveis — e assim que o anotador 2 fica
  // cego ao gabarito do anotador 1 sem que o front precise saber que rodadas existem.
  const rejectedRefs = new Set<string>()
  const keptRefs = new Set<string>()
  const supersededBy = new Map<string, number>() // det_id original → idx da correção
  const reviewedRefs = new Set<string>() // originais com veredito NESTA rodada

  arr.forEach((det, idx) => {
    if (!isHumanOfRound(det, round)) return
    const ref = det.ref_det_id
    if (!ref) return
    reviewedRefs.add(ref)
    if (det.action === 'rejected') rejectedRefs.add(ref)
    else if (det.action === 'kept') keptRefs.add(ref)
    else if (det.action === 'corrected') supersededBy.set(ref, idx)
  })

  const model: SerializedDetections['model'] = []
  const human: SerializedDetections['human'] = []

  arr.forEach((det, idx) => {
    if (isHuman(det)) {
      if (!isHumanOfRound(det, round)) return // entrada de outra rodada: nao vaza
      human.push({
        idx,
        det_id: det.det_id ?? null,
        class: det.class ?? null,
        action: det.action,
        ref_idx: det.ref_det_id ? detIdToIdx.get(det.ref_det_id) ?? null : null,
        box_n: boxNOf(det),
      })
    } else {
      const did = det.det_id ?? null
      model.push({
        idx,
        det_id: did,
        class: det.class ?? null,
        conf: typeof det.conf === 'number' ? det.conf : null,
        box_n: boxNOf(det),
        track_id: typeof det.track_id === 'number' ? det.track_id : null,
        dist_m: typeof det.dist_m === 'number' ? det.dist_m : null,
        // `det.reviewed` e um flag COMPARTILHADO no original, escrito pelo round 1.
        // Fora do round 1 ele revelaria o progresso do primeiro anotador, entao o
        // estado e derivado apenas das entradas humanas desta rodada.
        reviewed: round === ROUND_DEFAULT
          ? det.reviewed === true
          : did != null && reviewedRefs.has(did),
        rejected: did ? rejectedRefs.has(did) : false,
        kept: did ? keptRefs.has(did) : false,
        superseded_by: did ? supersededBy.get(did) ?? null : null,
      })
    }
  })

  return { model, human, yolo_count_model: model.length }
}

// ─────────────────────────────────────────────────────────────────────────────
// resolveGroundTruth — dobra originais + entradas humanas no GABARITO FINAL.
//
// Fonte única de verdade: quem quiser o gabarito (export, métricas, Forge) passa
// por aqui. Não reimplemente a dobra em outro lugar.
//
// Invariantes garantidas por action.ts (POST .../curation/frame/action):
//  - existe NO MÁXIMO uma entrada humana por original (dropPriorHumanRef);
//  - `kept`/`rejected` não carregam caixa — a caixa é a do original;
//  - `corrected`/`added` carregam caixa própria (x1_n..y2_n).
// ─────────────────────────────────────────────────────────────────────────────

export type GtOrigin = 'kept' | 'corrected' | 'added'
export type Verdict = GtOrigin | 'rejected' | 'unreviewed'

export type GtBox = {
  class: string | null
  box_n: BoxN | null
  origin: GtOrigin
  det_id: string | null
  ref_det_id: string | null // original de onde veio (null p/ "added")
}

export type ResolvedOriginal = {
  det_id: string | null
  class: string | null
  conf: number | null
  box_n: BoxN | null
  track_id: number | null
  dist_m: number | null
  verdict: Verdict
  // Só em `corrected`: o que o humano gravou por cima (para auditoria de erro).
  correction: { class: string | null; box_n: BoxN | null } | null
}

export type ResolvedFrame = {
  ground_truth: GtBox[]
  originals: ResolvedOriginal[]
  counts: {
    model: number
    kept: number
    corrected: number
    rejected: number
    added: number
    unreviewed: number
  }
}

export function resolveGroundTruth(
  yolo: RawDet[] | null | undefined,
  round: number = ROUND_DEFAULT,
): ResolvedFrame {
  const arr = yolo ?? []

  // Entrada humana DESTA RODADA que referencia cada original (no máximo uma, por
  // invariante). Cada rodada dobra para o seu proprio gabarito, independente da outra.
  const humanByRef = new Map<string, RawDet>()
  const added: RawDet[] = []
  for (const det of arr) {
    if (!isHumanOfRound(det, round)) continue
    if (det.action === 'added') { added.push(det); continue }
    if (det.ref_det_id) humanByRef.set(det.ref_det_id, det)
  }

  const ground_truth: GtBox[] = []
  const originals: ResolvedOriginal[] = []
  const counts = { model: 0, kept: 0, corrected: 0, rejected: 0, added: 0, unreviewed: 0 }

  for (const det of arr) {
    if (isHuman(det)) continue
    counts.model++

    const origBox = boxNOf(det)
    const human = det.det_id ? humanByRef.get(det.det_id) : undefined
    const action = human?.action as Verdict | undefined
    const verdict: Verdict =
      action === 'kept' || action === 'corrected' || action === 'rejected' ? action : 'unreviewed'

    let correction: ResolvedOriginal['correction'] = null

    if (verdict === 'kept') {
      counts.kept++
      // Caixa canônica é a do ORIGINAL — `kept` não grava caixa própria.
      ground_truth.push({
        class: human!.class ?? det.class ?? null,
        box_n: origBox,
        origin: 'kept',
        det_id: human!.det_id ?? null,
        ref_det_id: det.det_id ?? null,
      })
    } else if (verdict === 'corrected') {
      counts.corrected++
      correction = { class: human!.class ?? null, box_n: boxNOf(human!) }
      ground_truth.push({
        class: human!.class ?? null,
        box_n: boxNOf(human!),
        origin: 'corrected',
        det_id: human!.det_id ?? null,
        ref_det_id: det.det_id ?? null,
      })
    } else if (verdict === 'rejected') {
      counts.rejected++ // falso positivo: não entra no gabarito
    } else {
      counts.unreviewed++ // original sem veredito → frame não revisado
    }

    originals.push({
      det_id: det.det_id ?? null,
      class: det.class ?? null,
      conf: typeof det.conf === 'number' ? det.conf : null,
      box_n: origBox,
      track_id: typeof det.track_id === 'number' ? det.track_id : null,
      dist_m: typeof det.dist_m === 'number' ? det.dist_m : null,
      verdict,
      correction,
    })
  }

  for (const det of added) {
    counts.added++
    ground_truth.push({
      class: det.class ?? null,
      box_n: boxNOf(det),
      origin: 'added',
      det_id: det.det_id ?? null,
      ref_det_id: null,
    })
  }

  return { ground_truth, originals, counts }
}
