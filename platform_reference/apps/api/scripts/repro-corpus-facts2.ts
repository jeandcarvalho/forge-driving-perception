/** Somente leitura. Parte 2: links de imagem, fps/frame_idx do yolo, FilesProcessed (rig). */
import prisma from '../src/lib/prisma'

async function main() {
  const jpgDistinct = (await prisma.links.aggregateRaw({
    pipeline: [{ $match: { ext: 'jpg' } }, { $group: { _id: { a: '$acq_id', s: '$sec' } } }, { $count: 'n' }],
  })) as unknown as any[]
  console.log('Links jpg — pares (acq_id,sec) distintos:', jpgDistinct[0]?.n)

  const framesWithJpg = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $match: { links: { $elemMatch: { ext: 'jpg' } } } },
      { $count: 'n' },
    ],
  })) as unknown as any[]
  console.log('Big1Hz com links[ext=jpg]:', framesWithJpg[0]?.n ?? 0)

  const anyLinks = (await prisma.big1Hz.aggregateRaw({
    pipeline: [{ $match: { $expr: { $gt: [{ $size: { $ifNull: ['$links', []] } }, 0] } } }, { $count: 'n' }],
  })) as unknown as any[]
  console.log('Big1Hz com algum links[]:', anyLinks[0]?.n ?? 0)

  // fps / frame_idx das deteccoes: revela o video nativo
  const fps = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $unwind: '$yolo' },
      { $group: { _id: '$yolo.fps', n: { $sum: 1 } } }, { $sort: { n: -1 } },
    ],
  })) as unknown as any[]
  console.log('\nyolo.fps distintos:', JSON.stringify(fps))

  const px = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $unwind: '$yolo' },
      { $match: { 'yolo.x2': { $ne: null } } },
      { $group: { _id: null, maxx2: { $max: '$yolo.x2' }, maxy2: { $max: '$yolo.y2' }, minx1: { $min: '$yolo.x1' }, miny1: { $min: '$yolo.y1' } } },
    ],
  })) as unknown as any[]
  console.log('extremos de pixel das caixas armazenadas:', JSON.stringify(px))

  const fp = await prisma.filesProcessed.findMany({ take: 2 })
  console.log('\nFilesProcessed (2 docs):')
  console.log(JSON.stringify(fp, null, 1).slice(0, 4000))
  console.log('FilesProcessed total:', await prisma.filesProcessed.count())

  const rt = (await prisma.filesProcessed.aggregateRaw({
    pipeline: [{ $group: { _id: '$Road Type', n: { $sum: 1 } } }, { $sort: { n: -1 } }],
  })) as unknown as any[]
  console.log('Road Type:', JSON.stringify(rt))
}

main().catch(e => { console.error(e); process.exit(1) }).finally(() => prisma.$disconnect())
