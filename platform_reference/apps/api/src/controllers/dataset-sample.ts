import { FastifyRequest, FastifyReply } from 'fastify'
import { Prisma } from '@prisma/client'
import { type SearchQuery } from '@carcara/shared'
import prisma from '../lib/prisma'
import { buildMongoMatch } from '../lib/build-mongo-match'
import { haversineMeters } from '../lib/geo'
import { normalizeToCompact } from './collections/get-collection-seconds-with-links'
import { extractDriveId } from './collections/export-manifest'

// GET /api/dataset/sample
//   ?mode=random|query  &n=1000  &seed=42  &dedup_m=100
//   &exclude_collection=<id>  &exclude_mode=acqs|acqs_all|frames|none
//   &<filtros de busca>            (mesmos da /api/search, só em mode=query)
//
// Amostragem REPRODUTÍVEL sobre o acervo, para montar os braços do experimento de
// seleção de dados:
//   braço 1 (aleatório):    mode=random
//   braço 2 (busca):        mode=query&<filtros>
//   braço 3 (busca+dedup):  mode=query&<filtros>&dedup_m=100
//
// Mesmo `seed` ⇒ mesma amostra. A ordem é canônica (acq_id, sec) antes do embaralhamento,
// então o resultado não depende da ordem que o Mongo devolveu.

const MAX_N = 20_000

// PRNG determinístico (mulberry32) — não usar Math.random: a amostra precisa ser
// reproduzível por terceiros a partir do seed publicado no paper.
function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function seededShuffle<T>(arr: T[], seed: number): T[] {
  const rnd = mulberry32(seed)
  const out = [...arr]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

type Candidate = { acq_id: number; sec: number; lat: number | null; lon: number | null; link: string | null }

export async function datasetSample(request: FastifyRequest, reply: FastifyReply) {
  const q = request.query as SearchQuery & {
    mode?: string
    n?: string
    seed?: string
    dedup_m?: string
    exclude_collection?: string
    exclude_mode?: string
    spread?: string
  }

  const mode = (q.mode ?? 'query').toLowerCase()
  if (mode !== 'random' && mode !== 'query') {
    return reply.status(400).send({ error: 'mode deve ser random ou query.' })
  }
  const n = Math.min(MAX_N, Math.max(1, Number(q.n ?? '1000') || 1000))
  const seed = Number(q.seed ?? '42') || 42
  const dedupM = q.dedup_m != null ? Number(q.dedup_m) : null
  if (dedupM != null && (!Number.isFinite(dedupM) || dedupM < 0)) {
    return reply.status(400).send({ error: 'dedup_m deve ser um número >= 0.' })
  }
  const excludeMode = (q.exclude_mode ?? 'acqs').toLowerCase()
  if (!['acqs', 'acqs_all', 'frames', 'none'].includes(excludeMode)) {
    return reply.status(400).send({ error: 'exclude_mode deve ser acqs | acqs_all | frames | none.' })
  }
  const spread = q.spread === 'acq' || q.spread === 'true' || q.spread === '1'
  if (q.spread != null && !spread && !['false', '0', 'none'].includes(String(q.spread))) {
    return reply.status(400).send({ error: 'spread deve ser acq (rodízio entre aquisições) ou none.' })
  }

  // ── Exclusão anti-vazamento: tira do pool o conjunto de teste de uma coleção ──
  // `acqs` (padrão) é a regra estrita, a mesma de curation/generate.ts: a aquisição
  // inteira que doou frame ao teste sai do pool de treino.
  //
  // `acqs_all` é MAIS forte e serve a outra pergunta. `acqs` tira as 127 gravações
  // que doaram frame ao gabarito, mas deixa no pool as demais gravações da coleção
  // — as que a curadoria escolheu e não doaram nada. O braço "aleatório" sorteado
  // assim herda frames de gravação curada: no braço n4364 de 16/08, 1.371 dos 4.364
  // (31%) saíram de gravação da coleção, e eles carregam a densidade da curadoria
  // (7,55 obj/frame contra 2,24 do acervo intocado). Isso não vaza para a avaliação
  // — o gabarito continua fora — mas contamina a COMPARAÇÃO: mede curadoria contra
  // uma mistura que já é 31% curadoria, e o contraste sai encolhido.
  //
  // Use `acqs` quando a pergunta for "posso treinar com isto?" (antivazamento) e
  // `acqs_all` quando for "o que a curadoria acrescenta sobre o acervo?" (braço de
  // controle). Os dois pools são diferentes: 42.831 contra 29.671 frames.
  const excludedAcqs = new Set<number>()
  const excludedFrames = new Set<string>()
  if (q.exclude_collection && excludeMode !== 'none') {
    const groups = await prisma.collectionGroup.findMany({
      where: { collectionId: q.exclude_collection },
      select: { acq_id: true, curationSecs: true },
    })
    if (groups.length === 0) {
      return reply.status(404).send({ error: 'exclude_collection não encontrada (ou vazia).' })
    }
    for (const g of groups) {
      const num = Number(normalizeToCompact(g.acq_id))
      if (isNaN(num)) continue
      // `acqs_all` não olha para curationSecs: toda gravação que a coleção toca sai.
      if (excludeMode === 'acqs_all') {
        excludedAcqs.add(num)
        continue
      }
      if (g.curationSecs.length === 0) continue
      if (excludeMode === 'acqs') excludedAcqs.add(num)
      else for (const sec of g.curationSecs) excludedFrames.add(`${num}:${sec}`)
    }
  }

  // ── Pool de candidatos: frames com imagem (sem imagem não serve para treino) ──
  const match: Record<string, any> = mode === 'query' ? buildMongoMatch(q) : {}
  match['links.0.link'] = { $exists: true }

  const pipeline: Prisma.InputJsonValue[] = [
    { $match: match },
    { $sort: { acq_id: 1, sec: 1 } },
    {
      $project: {
        _id: 0,
        acq_id: 1,
        sec: 1,
        lat: '$can.GPS_y',
        lon: '$can.GPS_x',
        link: { $arrayElemAt: ['$links.link', 0] },
      },
    },
  ]

  const raw = (await prisma.big1Hz.aggregateRaw({ pipeline })) as unknown as Candidate[]

  let candidates = raw.filter((c) => {
    if (c.acq_id == null) return false
    if (excludedAcqs.has(c.acq_id)) return false
    if (excludedFrames.has(`${c.acq_id}:${c.sec}`)) return false
    return true
  })
  const poolAfterFilter = candidates.length

  // ── Dedup geográfico (a segunda alavanca da plataforma) ──
  // Greedy por aquisição, na ordem temporal: mesma regra do filtro de espaçamento.
  let poolAfterDedup = poolAfterFilter
  if (dedupM != null && dedupM > 0) {
    const kept: Candidate[] = []
    const lastByAcq = new Map<number, { lat: number; lon: number }>()
    for (const c of candidates) {
      if (c.lat == null || c.lon == null) { kept.push(c); continue }
      const prev = lastByAcq.get(c.acq_id)
      if (!prev || haversineMeters(prev.lat, prev.lon, c.lat, c.lon) >= dedupM) {
        kept.push(c)
        lastByAcq.set(c.acq_id, { lat: c.lat, lon: c.lon })
      }
    }
    candidates = kept
    poolAfterDedup = kept.length
  }

  // ── Diversidade de aquisição (a terceira alavanca) ──
  // O dedup acima espaça frames DENTRO de uma aquisição. Ele não impede que uma
  // única gravação domine a amostra: numa consulta por `car`, 23 dos frames
  // sorteados saíam da mesma gravação, contra 2,5 do gabarito (que a curadoria
  // monta espaçando a 1.593 m e por isso cobre 127 gravações em 313 frames).
  //
  // O rodízio dá uma volta por aquisição, pegando um frame de cada por vez, até
  // completar N. O teto por aquisição sai sozinho: ceil(N / nº de aquisições).
  const shuffled = seededShuffle(candidates, seed)
  let sampled: Candidate[]

  if (spread) {
    const byAcq = new Map<number, Candidate[]>()
    for (const c of shuffled) {
      const list = byAcq.get(c.acq_id)
      if (list) list.push(c)
      else byAcq.set(c.acq_id, [c])
    }
    // A ordem das aquisições também sai do seed: sem isso, as primeiras voltas
    // favoreceriam sempre as mesmas gravações (a ordem do Mongo).
    const acqOrder = seededShuffle([...byAcq.keys()], seed)

    sampled = []
    for (let round = 0; sampled.length < n; round++) {
      let progressed = false
      for (const acq of acqOrder) {
        const list = byAcq.get(acq)!
        if (round >= list.length) continue
        sampled.push(list[round])
        progressed = true
        if (sampled.length >= n) break
      }
      if (!progressed) break // todas as aquisições esgotadas
    }
  } else {
    sampled = shuffled.slice(0, n)
  }

  // Devolve em ordem canônica — embaralhar serve para escolher, não para entregar.
  sampled.sort((a, b) => (a.acq_id - b.acq_id) || (a.sec - b.sec))

  const acqsInPool = new Set(candidates.map((c) => c.acq_id)).size
  const perAcq = new Map<number, number>()
  for (const c of sampled) perAcq.set(c.acq_id, (perAcq.get(c.acq_id) ?? 0) + 1)

  const frames = sampled.map((c) => {
    const driveId = c.link ? extractDriveId(c.link) : null
    return {
      acq_id: String(c.acq_id),
      sec: c.sec,
      gps: c.lat != null && c.lon != null ? { lat: c.lat, lon: c.lon } : null,
      image: driveId
        ? {
            drive_id: driveId,
            source_link: c.link,
            url_original: `https://drive.usercontent.google.com/download?id=${driveId}&export=download&confirm=t`,
            url_cdn: `https://lh3.googleusercontent.com/d/${driveId}=w1920-h1080-n`,
          }
        : null,
    }
  })

  return reply.send({
    mode,
    seed,
    n_requested: n,
    n_returned: frames.length,
    dedup_m: dedupM,
    exclude_collection: q.exclude_collection ?? null,
    exclude_mode: q.exclude_collection ? excludeMode : 'none',
    excluded_acqs: excludedAcqs.size,
    excluded_frames: excludedFrames.size,
    // Frames que casaram o match e têm imagem (em mode=random, o acervo inteiro).
    pool_matched: raw.length,
    pool_after_exclusion: poolAfterFilter,
    pool_after_dedup: poolAfterDedup,
    // Diversidade: quantas gravações distintas a amostra cobre, e o quanto a
    // maior delas domina. É a estatística que o dedup por GPS não move.
    spread: spread ? 'acq' : 'none',
    acqs_in_pool: acqsInPool,
    acqs_sampled: perAcq.size,
    frames_per_acq_max: perAcq.size ? Math.max(...perAcq.values()) : 0,
    frames_per_acq_mean: perAcq.size
      ? Math.round((frames.length / perAcq.size) * 100) / 100
      : 0,
    frames,
  })
}
