"""Redundancia intra-gravacao na COLECAO COMPLETA (4364) vs aleatorio pareado (4364)."""
import json, math
from collections import defaultdict
from pathlib import Path
import numpy as np

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
col = man["frames"]
arm = json.loads((ROOT / "arms" / "random-n4364-s42.json").read_text(encoding="utf-8"))["frames"]
R = 6371008.8


def analyse(items, get_gps, get_acq, label):
    by_acq = defaultdict(list)
    for it in items:
        g = get_gps(it)
        if g and g.get("lat") is not None:
            by_acq[get_acq(it)].append((g["lat"], g["lon"]))
    nn, pairs_lt, total_pairs = [], defaultdict(int), 0
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
        iu = np.triu_indices(n, 1)
        vals = d[iu]
        total_pairs += vals.size
        for t in (10, 25, 50):
            pairs_lt[t] += int((vals < t).sum())
    arr = np.array(nn)
    print(f"\n{label}  (n={len(items)}, gravacoes={len(by_acq)})")
    print(f"  vizinho + proximo: min={arr.min():7.1f} m  p05={np.percentile(arr,5):7.1f}  "
          f"mediana={np.median(arr):7.1f}  media={arr.mean():7.1f}")
    for t in (10, 25, 50):
        print(f"  frames com vizinho < {t:3} m: {int((arr<t).sum()):5} ({100*(arr<t).mean():5.1f}%)  "
              f"| pares < {t:3} m: {pairs_lt[t]:6} ({100*pairs_lt[t]/total_pairs:6.3f}%)")
    return arr


analyse(col, lambda f: f["telemetry"]["gps"], lambda f: f["acq_id"], "COLECAO COMPLETA (4364)")
analyse(arm, lambda f: f["gps"], lambda f: f["acq_id"], "ALEATORIO (n4364 s42)")

# diversidade, para registro
for items, gk, ak, lb in ((col, lambda f: f["telemetry"]["gps"], lambda f: f["acq_id"], "colecao"),
                          (arm, lambda f: f["gps"], lambda f: f["acq_id"], "aleatorio")):
    acqs = defaultdict(int)
    for it in items:
        acqs[ak(it)] += 1
    print(f"\n{lb}: {len(acqs)} gravacoes · {len(items)/len(acqs):.1f} frames/gravacao · maior {max(acqs.values())}")
