"""Espacamento real da colecao + formula da taxa de quase-duplicata (Pillar 1).

So leitura do manifest. Nao roda modelo nenhum.
Reproduz a metrica de paper/spacing_full.py com o denominador AUDITADO.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
OUT = Path(__file__).resolve().parent
R = 6371008.8  # raio medio da Terra usado em spacing_full.py

man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
frames = man["frames"]

# ── 1. espacamento CONSECUTIVO no tempo, dentro da gravacao ──────────────────
by_acq = defaultdict(list)
no_gps = 0
for f in frames:
    g = (f.get("telemetry") or {}).get("gps") or {}
    if g.get("lat") is None or g.get("lon") is None:
        no_gps += 1
        continue
    by_acq[f["acq_id"]].append((f["sec"], g["lat"], g["lon"]))


def hav(lat1, lon1, lat2, lon2):
    p = np.radians
    dla, dlo = p(lat2 - lat1), p(lon2 - lon1)
    a = np.sin(dla / 2) ** 2 + np.cos(p(lat1)) * np.cos(p(lat2)) * np.sin(dlo / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(min(a, 1.0)))


consec = []
sec_gaps = []
for acq, rows in by_acq.items():
    rows.sort()
    for (s1, la1, lo1), (s2, la2, lo2) in zip(rows, rows[1:]):
        consec.append(hav(la1, lo1, la2, lo2))
        sec_gaps.append(s2 - s1)
consec = np.array(consec)
sec_gaps = np.array(sec_gaps)

spacing = {
    "frames_total": len(frames),
    "frames_sem_gps": no_gps,
    "frames_com_gps": sum(len(v) for v in by_acq.values()),
    "acquisitions": len(by_acq),
    "pares_consecutivos": int(consec.size),
    "dist_consecutiva_m": {
        "min": round(float(consec.min()), 2),
        "p01": round(float(np.percentile(consec, 1)), 2),
        "p05": round(float(np.percentile(consec, 5)), 2),
        "p25": round(float(np.percentile(consec, 25)), 2),
        "mediana": round(float(np.median(consec)), 2),
        "p75": round(float(np.percentile(consec, 75)), 2),
        "p95": round(float(np.percentile(consec, 95)), 2),
        "max": round(float(consec.max()), 2),
        "media": round(float(consec.mean()), 2),
    },
    "fracao_consecutivos_abaixo_de": {
        str(t): round(float((consec < t).mean()) * 100, 2) for t in (5, 10, 25, 30, 50)
    },
    "gap_em_segundos": {
        "min": int(sec_gaps.min()), "mediana": float(np.median(sec_gaps)),
        "moda_aprox": int(np.bincount(sec_gaps[sec_gaps < 200]).argmax()),
        "max": int(sec_gaps.max()),
        "fracao_gap_igual_1s": round(float((sec_gaps == 1).mean()) * 100, 2),
    },
}

# ── 2. taxa de quase-duplicata, exatamente como spacing_full.py ─────────────
def near_dup(items, get_gps, get_acq, thresholds=(10, 25, 50)):
    grp = defaultdict(list)
    dropped_no_gps = 0
    for it in items:
        g = get_gps(it)
        if not g or g.get("lat") is None:
            dropped_no_gps += 1
            continue
        grp[get_acq(it)].append((g["lat"], g["lon"]))

    acqs_total = len(grp)
    acqs_singleton = sum(1 for v in grp.values() if len(v) < 2)
    frames_singleton = sum(len(v) for v in grp.values() if len(v) < 2)

    nn, pairs_lt, total_pairs = [], defaultdict(int), 0
    for pts in grp.values():
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
        for t in thresholds:
            pairs_lt[t] += int((vals < t).sum())

    arr = np.array(nn)
    return {
        "itens_entrada": len(items),
        "descartados_sem_gps": dropped_no_gps,
        "acquisitions_com_gps": acqs_total,
        "acquisitions_com_1_frame_excluidas": acqs_singleton,
        "frames_em_acq_de_1_frame_excluidos": frames_singleton,
        "DENOMINADOR_frames_avaliados": int(arr.size),
        "nn_m": {
            "min": round(float(arr.min()), 2),
            "p05": round(float(np.percentile(arr, 5)), 2),
            "mediana": round(float(np.median(arr)), 2),
            "media": round(float(arr.mean()), 2),
            "max": round(float(arr.max()), 2),
        },
        "por_frame": {
            str(t): {
                "frames_com_vizinho_abaixo": int((arr < t).sum()),
                "pct": round(float((arr < t).mean()) * 100, 2),
            } for t in thresholds
        },
        "por_par": {
            str(t): {
                "pares_abaixo": pairs_lt[t],
                "total_pares": total_pairs,
                "pct": round(100 * pairs_lt[t] / total_pairs, 4),
            } for t in thresholds
        },
    }


col = near_dup(frames, lambda f: (f.get("telemetry") or {}).get("gps"), lambda f: f["acq_id"])

arms = {}
for name in ("random-n4364-s42.json", "random-n4364-s42-full.json"):
    p = ROOT / "arms" / name
    if p.exists():
        items = json.loads(p.read_text(encoding="utf-8"))["frames"]
        arms[name] = near_dup(items, lambda f: f.get("gps"), lambda f: f["acq_id"])

# ── 3. o subconjunto de teste (313) ─────────────────────────────────────────
test = [f for f in frames if f["split"] == "test"]
test_nd = near_dup(test, lambda f: (f.get("telemetry") or {}).get("gps"), lambda f: f["acq_id"])
test_acqs = {f["acq_id"] for f in test}

out = {
    "_fonte": str(ROOT / "manifest.json"),
    "_metrica": (
        "taxa de quase-duplicata = |{frames f : d_hav(f, NN_mesma_acq(f)) < t}| / "
        "|{frames com GPS em acquisitions com >= 2 frames com GPS}|"
    ),
    "espacamento_consecutivo_colecao_4364": spacing,
    "quase_duplicata_colecao_4364": col,
    "quase_duplicata_test_313": {**test_nd, "acquisitions_no_split_test": len(test_acqs)},
    "quase_duplicata_bracos_aleatorios": arms,
}
(OUT / "pillar1-spacing-nearduplicate.json").write_text(
    json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
