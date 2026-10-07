"""Item 4: a redundancia (Table 4) tinha o mesmo furo que a densidade (Table 3)?

O braco `random-n4364-s42` foi sorteado com exclude_mode=acqs, que so tira as 127
gravacoes doadoras. Sobraram no braco 1.371 frames (31%) vindos das 44 gravacoes
NAO-doadoras da propria colecao — frames que a curadoria escolheu. A densidade
comprovadamente inflou por causa disso. Aqui a pergunta e se o vizinho <25 m
tambem inflou.

So le. Nao escreve braco, split nem rotulo.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
ACQS = Path(os.environ.get("FORGE_ACQS_JSON", "acqs.json"))  # per-acquisition GPS list exported from the private database; not distributed
R = 6371008.8


def nn_stats(by_acq):
    """Vizinho mais proximo DENTRO da gravacao — a unidade de redundancia."""
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


def report(label, by_acq, total_frames):
    arr = nn_stats(by_acq)
    if arr.size == 0:
        print(f"{label:<34} (sem par dentro de gravacao)")
        return
    print(
        f"{label:<34}{total_frames:>7}{len(by_acq):>7}{arr.min():>8.1f}"
        f"{np.percentile(arr, 5):>8.1f}{np.median(arr):>10.1f}"
        f"{100 * (arr < 10).mean():>10.1f}%{100 * (arr < 25).mean():>9.1f}%"
    )


def main():
    d = json.loads(ACQS.read_text(encoding="utf-8"))
    touched, donor = set(d["touched"]), set(d["donor"])

    arm = json.loads((ROOT / "arms" / "random-n4364-s42.json").read_text(encoding="utf-8"))["frames"]
    cats = {"intocada": defaultdict(list), "nao-doadora": defaultdict(list), "todo o braco": defaultdict(list)}
    counts = defaultdict(int)
    for f in arm:
        g = f.get("gps")
        if not g or g.get("lat") is None:
            continue
        acq = f["acq_id"]
        cat = "intocada" if acq not in touched else ("doadora" if acq in donor else "nao-doadora")
        if cat == "doadora":  # nao deveria existir: exclude_mode=acqs ja tirou
            print(f"  !! frame de gravacao doadora no braco: {acq}")
            continue
        cats[cat][acq].append((g["lat"], g["lon"]))
        cats["todo o braco"][acq].append((g["lat"], g["lon"]))
        counts[cat] += 1
        counts["todo o braco"] += 1

    print("Braco random-n4364-s42 (exclude_mode=acqs) decomposto\n")
    print(f"{'fatia':<34}{'frames':>7}{'acqs':>7}{'min':>8}{'P05':>8}{'mediana':>10}{'<10m':>11}{'<25m':>10}")
    for k in ("todo o braco", "intocada", "nao-doadora"):
        report(k, cats[k], counts[k])

    # A colecao, para referencia na mesma regua.
    man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    col = defaultdict(list)
    ncol = 0
    for f in man["frames"]:
        g = (f.get("telemetry") or {}).get("gps")
        if g and g.get("lat") is not None:
            col[f["acq_id"]].append((g["lat"], g["lon"]))
            ncol += 1
    print()
    report("COLECAO COMPLETA (referencia)", col, ncol)


if __name__ == "__main__":
    main()
