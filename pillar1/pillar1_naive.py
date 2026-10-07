"""Pilar 1 com o braco de controle NAIVE: sorteio uniforme do acervo inteiro.

Motivacao (25/08/2026). A pergunta do Pilar 1 e "curadoria bate sorteio no mesmo
acervo?". O braco `acqs_all` de 19/08 responde outra: tira do pool as 171
gravacoes que a colecao tocou e sorteia nas 113 restantes. Isso encolhe o
universo de 80.584 para 29.671 frames e, pior, deixa o controle concentrado num
unico dia de coleta (101 de 106 gravacoes em 2024-02-03). Trocamos um vies por
outro maior.

O baseline natural e sortear 4.364 frames dos 80.584 do acervo, sem exclusao.
Um frame nao escolhido de uma gravacao que a colecao tocou continua sendo um
sorteio legitimo do acervo: a densidade daquela gravacao e propriedade do
ACERVO, que a curadoria encontrou, nao criou. Remove-la do pool do controle e
tirar do sorteio exatamente a parte onde mora o sinal.

Este script so LE. Nao escreve braco, split nem rotulo. Table 4 (redundancia),
cobertura de gravacao e os diagnosticos de vies saem so da telemetria GPS, de
graca. Table 3 (densidade) precisa do professor e mora em tables3_naive.py.
"""
import json
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
HERE = Path(__file__).resolve().parent
R = 6371008.8

SEEDS_FULL = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
SEEDS_NOCUR = [42, 43, 44]


def nn_within_acq(by_acq):
    """Distancia ao vizinho mais proximo DENTRO da mesma gravacao (metros)."""
    nn = []
    for pts in by_acq.values():
        n = len(pts)
        if n < 2:
            continue
        a = np.radians(np.array(pts))
        la, lo = a[:, 0], a[:, 1]
        dla = la[:, None] - la[None, :]
        dlo = lo[:, None] - lo[None, :]
        h = np.sin(dla / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlo / 2) ** 2
        d = 2 * R * np.arcsin(np.sqrt(np.clip(h, 0, 1)))
        np.fill_diagonal(d, np.inf)
        nn.extend(d.min(axis=1).tolist())
    return np.array(nn)


def stats(label, items, get_gps, get_acq, curated_acqs, curated_keys):
    by = defaultdict(list)
    seen = []
    keys = set()
    n = 0
    for it in items:
        acq = str(get_acq(it))
        seen.append(acq)
        keys.add((acq, it["sec"]))
        g = get_gps(it)
        if g and g.get("lat") is not None:
            by[acq].append((g["lat"], g["lon"]))
            n += 1
    arr = nn_within_acq(by)
    days = Counter(a[:8] for a in seen)
    top_day, top_day_n = days.most_common(1)[0]
    acq_days = Counter(a[:8] for a in set(seen))
    return {
        "label": label,
        "frames": len(seen),
        "frames_gps": n,
        "acqs": len(set(seen)),
        "fr_per_acq": len(seen) / len(set(seen)),
        "min": float(arr.min()),
        "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "lt10_pct": float(100 * (arr < 10).mean()),
        "lt10_n": int((arr < 10).sum()),
        "lt25_pct": float(100 * (arr < 25).mean()),
        "lt25_n": int((arr < 25).sum()),
        "days": len(days),
        "top_day": top_day,
        "top_day_pct": 100 * top_day_n / len(seen),
        "top_day_acqs": acq_days.most_common(1)[0][1],
        "from_curated_acq_pct": 100 * sum(1 for a in seen if a in curated_acqs) / len(seen),
        "overlap_curated_frames": len(keys & curated_keys),
    }


def agg(rows, key):
    v = [r[key] for r in rows]
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def main():
    man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    col_frames = man["frames"]
    curated_acqs = {str(f["acq_id"]) for f in col_frames}
    curated_keys = {(str(f["acq_id"]), f["sec"]) for f in col_frames}

    def gps_col(f):
        return (f.get("telemetry") or {}).get("gps")

    rows = stats("colecao curada", col_frames, gps_col, lambda f: f["acq_id"],
                 curated_acqs, curated_keys)

    def arm(path, label):
        d = json.loads(path.read_text(encoding="utf-8"))
        return stats(label, d["frames"], lambda f: f.get("gps"),
                     lambda f: f["acq_id"], curated_acqs, curated_keys)

    old = arm(ROOT / "arms" / "random-n4364-s42-acqsall.json", "aleatorio acqs_all s42 (controle 19/08)")
    old_leaky = arm(ROOT / "arms" / "random-n4364-s42.json", "aleatorio acqs s42 (controle 16/08)")
    full = [arm(ROOT / "arms" / f"random-n4364-s{s}-full.json", f"aleatorio acervo s{s}") for s in SEEDS_FULL]
    nocur = [arm(ROOT / "arms" / f"random-n4364-s{s}-nocur.json", f"acervo sem os curados s{s}") for s in SEEDS_NOCUR]

    hdr = (f"{'braco':<40}{'frames':>7}{'acqs':>6}{'fr/acq':>8}{'P05':>7}{'med':>8}"
           f"{'<10m':>9}{'<25m':>9}{'dias':>6}{'maior dia':>11}")
    print("=" * len(hdr))
    print("TABLE 4 + COBERTURA - redundancia intra-gravacao e diversidade")
    print("=" * len(hdr))
    print(hdr)

    def line(r):
        print(f"{r['label']:<40}{r['frames']:>7}{r['acqs']:>6}{r['fr_per_acq']:>8.1f}"
              f"{r['p05']:>7.1f}{r['median']:>8.1f}{r['lt10_pct']:>8.1f}%{r['lt25_pct']:>8.1f}%"
              f"{r['days']:>6}{r['top_day_pct']:>10.1f}%")

    line(rows)
    print("-" * len(hdr))
    line(old_leaky)
    line(old)
    print("-" * len(hdr))
    for r in full:
        line(r)
    print("-" * len(hdr))
    for r in nocur:
        line(r)

    col = rows
    print()
    print("=" * len(hdr))
    print(f"MEDIA +- DP sobre {len(SEEDS_FULL)} sementes (braco NAIVE, acervo inteiro)")
    print("=" * len(hdr))
    spec = [
        ("acqs", "{:.1f}", " gravacoes cobertas"),
        ("fr_per_acq", "{:.2f}", " frames por gravacao"),
        ("median", "{:.1f}", " m  mediana do vizinho"),
        ("p05", "{:.1f}", " m  P05"),
        ("lt10_pct", "{:.2f}", " %  vizinho < 10 m"),
        ("lt25_pct", "{:.2f}", " %  vizinho < 25 m"),
        ("days", "{:.1f}", " dias de coleta"),
        ("top_day_pct", "{:.1f}", " %  no maior dia"),
        ("from_curated_acq_pct", "{:.1f}", " %  de gravacao que a colecao tocou"),
        ("overlap_curated_frames", "{:.1f}", " frames em comum com a colecao"),
    ]
    for key, fmt, unit in spec:
        m, s = agg(full, key)
        cell = fmt.format(m) + " +- " + fmt.format(s)
        print(f"  {cell:>22}{unit}   [colecao: {fmt.format(col[key])}]")

    print()
    print("  razoes colecao / aleatorio-naive (media das sementes):")
    for key, name in [("lt10_pct", "vizinho < 10 m"), ("lt25_pct", "vizinho < 25 m")]:
        m, _ = agg(full, key)
        r = "inf" if col[key] == 0 else f"{m / col[key]:.0f}x menos"
        print(f"    {name:<18} colecao {col[key]:.2f}%  x  aleatorio {m:.2f}%  ->  {r}")

    print()
    print("=" * len(hdr))
    print("DIAGNOSTICO DE VIES - o que cada desenho de controle faz com o pool")
    print("=" * len(hdr))
    print(f"{'braco':<40}{'pool':>9}{'acq pool':>10}{'dias':>6}{'maior dia':>11}{'de acq curada':>15}")
    diag = [(old_leaky, 42831, 157), (old, 29671, 113)] + [(x, 80584, 284) for x in full[:3]]
    for r, pool, apool in diag:
        print(f"{r['label']:<40}{pool:>9}{apool:>10}{r['days']:>6}"
              f"{r['top_day_pct']:>10.1f}%{r['from_curated_acq_pct']:>14.1f}%")

    out = {"collection": col, "old_acqs_all": old, "old_acqs": old_leaky, "full": full, "nocur": nocur}
    (HERE / "pillar1_naive.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nresultados -> {HERE / 'pillar1_naive.json'}")


if __name__ == "__main__":
    main()
