/** Somente leitura. Fatos do corpus indexado — para conferir os números da Figure 1
 *  e caracterizar a coleção do Pillar 1. Não escreve nada.
 */
import prisma from '../src/lib/prisma'

async function main() {
  const total = await prisma.big1Hz.count()
  console.log('Big1Hz docs (frames de telemetria indexados):', total)

  const acqs = (await prisma.big1Hz.aggregateRaw({
    pipeline: [{ $group: { _id: '$acq_id' } }, { $count: 'n' }],
  })) as unknown as any[]
  console.log('acq_id distintos:', acqs[0]?.n)

  const days = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $group: { _id: { $substrCP: [{ $toString: '$acq_id' }, 0, 8] }, n: { $sum: 1 }, acqs: { $addToSet: '$acq_id' } } },
      { $project: { _id: 1, n: 1, acqs: { $size: '$acqs' } } },
      { $sort: { _id: 1 } },
    ],
  })) as unknown as any[]
  console.log('\ndias de campo:', days.length)
  console.log(JSON.stringify(days, null, 1))

  const veh = (await prisma.big1Hz.aggregateRaw({
    pipeline: [{ $group: { _id: '$block.vehicle', n: { $sum: 1 }, acqs: { $addToSet: '$acq_id' } } },
               { $project: { _id: 1, n: 1, acqs: { $size: '$acqs' } } }, { $sort: { n: -1 } }],
  })) as unknown as any[]
  console.log('\nveículos no corpus:', JSON.stringify(veh, null, 1))

  const per = (await prisma.big1Hz.aggregateRaw({
    pipeline: [{ $group: { _id: '$block.meteo.period', n: { $sum: 1 } } }, { $sort: { n: -1 } }],
  })) as unknown as any[]
  console.log('períodos:', JSON.stringify(per))

  const withYolo = (await prisma.big1Hz.aggregateRaw({
    pipeline: [{ $match: { $expr: { $gt: [{ $size: { $ifNull: ['$yolo', []] } }, 0] } } }, { $count: 'n' }],
  })) as unknown as any[]
  console.log('\nframes com >=1 deteccao yolo:', withYolo[0]?.n)

  // conf mínima/máxima das detecções armazenadas (corte do pipeline de indexação)
  const conf = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $unwind: '$yolo' },
      { $group: { _id: null, min: { $min: '$yolo.conf' }, max: { $max: '$yolo.conf' }, n: { $sum: 1 } } },
    ],
  })) as unknown as any[]
  console.log('conf das deteccoes armazenadas:', JSON.stringify(conf))

  // links de imagem
  const linkExt = (await prisma.links.aggregateRaw({
    pipeline: [{ $group: { _id: '$ext', n: { $sum: 1 }, acqs: { $addToSet: '$acq_id' } } },
               { $project: { _id: 1, n: 1, acqs: { $size: '$acqs' } } }, { $sort: { n: -1 } }],
  })) as unknown as any[]
  console.log('\nLinks por extensão:', JSON.stringify(linkExt, null, 1))

  const framesWithImg = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $match: { $expr: { $gt: [{ $size: { $ifNull: ['$link_files', []] } }, 0] } } },
      { $count: 'n' },
    ],
  })) as unknown as any[]
  console.log('frames com link_files:', framesWithImg[0]?.n)

  const one = await prisma.big1Hz.findFirst({ select: { acq_id: true, sec: true, block: true, link_files: true } })
  console.log('\namostra de um doc (block + link_files):')
  console.log(JSON.stringify(one, null, 1).slice(0, 2500))

  const fp = await prisma.filesProcessed.count()
  console.log('\nFilesProcessed docs:', fp)
  const fpSample = await prisma.filesProcessed.findFirst()
  console.log(JSON.stringify(fpSample, null, 1).slice(0, 3000))
}

main().catch(e => { console.error(e); process.exit(1) }).finally(() => prisma.$disconnect())
