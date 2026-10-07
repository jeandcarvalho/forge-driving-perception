/** Somente leitura. Exporta acq_id, sec, classes distintas do yolo, n de deteccoes,
 *  velocidade, periodo, condicao — para testar hipoteses de query offline. */
import prisma from '../src/lib/prisma'
import { writeFileSync } from 'fs'

async function main() {
  const rows = (await prisma.big1Hz.aggregateRaw({
    pipeline: [
      { $project: {
          _id: 0, acq_id: 1, sec: 1,
          gx: '$can.GPS_x', gy: '$can.GPS_y',
          v: '$can.VehicleSpeed',
          cls: { $setUnion: [{ $ifNull: ['$yolo.class', []] }, []] },
          nd: { $size: { $ifNull: ['$yolo', []] } },
          per: '$block.meteo.period',
          jpg: { $cond: [{ $gt: [{ $size: { $ifNull: ['$links', []] } }, 0] }, 1, 0] },
        } },
    ],
  })) as unknown as any[]
  console.log('linhas:', rows.length)
  const out = 'corpus-frame-classes.json'
  writeFileSync(out, JSON.stringify(rows), 'utf-8')
  console.log('escrito', out)
}
main().catch(e => { console.error(e); process.exit(1) }).finally(() => prisma.$disconnect())
