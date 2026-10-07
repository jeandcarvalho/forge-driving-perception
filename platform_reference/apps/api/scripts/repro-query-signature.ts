/** Somente leitura. Testa hipoteses sobre a query que gerou a colecao do Pillar 1.
 *  Nao escreve nada. Exporta GPS + presenca de yolo por frame para o teste de espacamento.
 */
import prisma from '../src/lib/prisma'
import { writeFileSync } from 'fs'

async function main() {
  const acqsWithYolo = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $match: { $expr: { $gt: [{ $size: { $ifNull: ['$yolo', []] } }, 0] } } },
      { $group: { _id: '$acq_id' } }, { $count: 'n' },
    ],
  })) as unknown as any[]
  console.log('acq_ids com >=1 frame com deteccao:', acqsWithYolo[0]?.n)

  const acqsWithJpg = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $match: { links: { $elemMatch: { ext: 'jpg' } } } },
      { $group: { _id: '$acq_id' } }, { $count: 'n' },
    ],
  })) as unknown as any[]
  console.log('acq_ids com >=1 frame com jpg:', acqsWithJpg[0]?.n)

  const both = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $match: { links: { $elemMatch: { ext: 'jpg' } }, $expr: { $gt: [{ $size: { $ifNull: ['$yolo', []] } }, 0] } } },
      { $count: 'n' },
    ],
  })) as unknown as any[]
  console.log('frames com jpg E >=1 deteccao:', both[0]?.n)

  // dump enxuto para o teste de espacamento offline
  const rows = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $project: {
          _id: 0, acq_id: 1, sec: 1,
          gx: '$can.GPS_x', gy: '$can.GPS_y',
          nd: { $size: { $ifNull: ['$yolo', []] } },
          jpg: { $cond: [{ $gt: [{ $size: { $ifNull: ['$links', []] } }, 0] }, 1, 0] },
        } },
    ],
  })) as unknown as any[]
  console.log('linhas exportadas:', rows.length)
  const out = 'corpus-gps-yolo.json'
  writeFileSync(out, JSON.stringify(rows), 'utf-8')
  console.log('escrito', out)
}

main().catch(e => { console.error(e); process.exit(1) }).finally(() => prisma.$disconnect())
