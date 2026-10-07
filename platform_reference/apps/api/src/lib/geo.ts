// Utilitários geográficos reaproveitados por spacing/curadoria.

export type LatLon = { lat: number; lon: number }

export function haversineMeters(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6_371_000
  const toRad = (d: number) => (d * Math.PI) / 180
  const dLat = toRad(lat2 - lat1)
  const dLon = toRad(lon2 - lon1)
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRad(lat1)) * Math.cos(toRad(lat2)) * Math.sin(dLon / 2) ** 2
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
}

// Menor distância (m) de um ponto a um conjunto de pontos. Infinity se o conjunto é vazio.
export function minDistanceToSet(point: LatLon, set: LatLon[]): number {
  let min = Infinity
  for (const p of set) {
    const d = haversineMeters(point.lat, point.lon, p.lat, p.lon)
    if (d < min) min = d
  }
  return min
}
