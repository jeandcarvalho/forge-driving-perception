// ============================
// Constantes de domínio — usadas no front (chips/filtros) e no back (match)
// ============================

export const VEHICLE_NORMALIZATION: Record<string, string> = {
  captur: "Captur",
  "daf cf 410": "DAF CF 410",
  renegade: "Renegade",
};

export const VEHICLES = ["Captur", "DAF CF 410", "Renegade"] as const;
export const PERIODS = ["day", "night", "dawn", "dusk"] as const;
export const CONDITIONS = ["clear", "rain", "fog", "cloudy"] as const;

export const BRAKE_DB_MAP: Record<string, string> = {
  not_pressed: "Not pressed",
  pressed: "Pressed",
};

export const SWA_CHIPS = [
  { label: "Straight", range: "-15..15" },
  { label: "Slight left", range: "15..45" },
  { label: "Slight right", range: "-45..-15" },
  { label: "Left", range: "45..180" },
  { label: "Right", range: "-180..-45" },
] as const;

export const SEMSEG_THRESH = {
  low: "0..0.2",
  mid: "0.2..0.5",
  high: "0.5..1",
} as const;

export const MAX_SECS_WITH_LINKS_PER_ACQ = 5;

// ============================
// Helpers de parsing — usados no front e no back
// ============================

export function splitList(v?: string | null): string[] {
  if (!v) return [];
  return v
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

export function parseRangeToken(tok: string): [number | null, number | null] {
  const [a, b] = tok.split("..");
  const min = a === "" || a === undefined ? null : Number(a);
  const max = b === "" || b === undefined ? null : Number(b);
  return [
    Number.isFinite(min as number) ? (min as number) : null,
    Number.isFinite(max as number) ? (max as number) : null,
  ];
}

export function toRangeParam(
  min: number | "",
  max: number | ""
): string | null {
  if (min === "" && max === "") return null;
  return `${min}..${max}`;
}

export function approxEq(a: number, b: number, eps = 0.01): boolean {
  return Math.abs(a - b) < eps;
}
