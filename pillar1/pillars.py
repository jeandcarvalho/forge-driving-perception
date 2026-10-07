"""Fecha os numeros dos dois pilares do conference, a partir do que esta em disco."""
import json, math, itertools
from pathlib import Path
import numpy as np

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"

man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
frames = man["frames"]
train = [f for f in frames if f["split"] == "train"]
test = [f for f in frames if f["split"] == "test"]
arm = json.loads((ROOT / "arms" / "random-n1406-s42.json").read_text(encoding="utf-8"))["frames"]

R = 6371.0088  # km


def latlon(items, getter):
    pts = []
    for it in items:
        g = getter(it)
        if g and g.get("lat") is not None and g.get("lon") is not None:
            pts.append((g["lat"], g["lon"]))
    return np.array(pts)


def spatial(pts, acqs, label):
    lat = np.radians(pts[:, 0]); lon = np.radians(pts[:, 1])
    # distancia media par-a-par (haversine), amostrada se grande
    n = len(pts)
    idx = np.arange(n)
    sl, cl = np.sin(lat), np.cos(lat)
    # matriz completa (n<=1500 -> ok)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + cl[:, None] * cl[None, :] * np.sin(dlon / 2) ** 2
    d = 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    iu = np.triu_indices(n, 1)
    mean_pair = d[iu].mean()
    # dispersao em torno do centroide
    clat, clon = pts[:, 0].mean(), pts[:, 1].mean()
    cy = np.radians(clat)
    dx = (pts[:, 1] - clon) * 111.32 * math.cos(cy)
    dy = (pts[:, 0] - clat) * 110.574
    rms = float(np.sqrt((dx ** 2 + dy ** 2).mean()))
    # cobertura por celula
    def cells(size_m):
        s = size_m / 1000.0
        ky = np.floor(dy / s).astype(int); kx = np.floor(dx / s).astype(int)
        return len(set(zip(kx.tolist(), ky.tolist())))
    uniq_acq = len(set(acqs))
    per_acq = n / uniq_acq
    from collections import Counter
    c = Counter(acqs)
    print(f"{label:28} n={n:5}  acqs={uniq_acq:4}  fr/acq={per_acq:6.1f} (max {max(c.values()):4})  "
          f"dist_par={mean_pair:6.2f} km  rms_centroide={rms:6.2f} km  "
          f"celulas 100m={cells(100):5}  1km={cells(1000):4}")
    return dict(n=n, acqs=uniq_acq, per_acq=per_acq, max_acq=max(c.values()),
                mean_pair=mean_pair, rms=rms, c100=cells(100), c1000=cells(1000))


print("=== PILAR 1B — diversidade espacial (mesmo N=1406) ===")
col = spatial(latlon(train, lambda f: f["telemetry"]["gps"]), [f["acq_id"] for f in train], "colecao curada (train)")
rnd = spatial(latlon(arm, lambda f: f["gps"]), [f["acq_id"] for f in arm], "aleatorio (n1406 s42)")
tst = spatial(latlon(test, lambda f: f["telemetry"]["gps"]), [f["acq_id"] for f in test], "gabarito (test 313)")

print()
print("=== PILAR 2 — audit sobre os 313 do gabarito ===")


def iou(a, b):
    x1 = max(a["x1"], b["x1"]); y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"]); y2 = min(a["y2"], b["y2"])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ar = (a["x2"] - a["x1"]) * (a["y2"] - a["y1"])
    br = (b["x2"] - b["x1"]) * (b["y2"] - b["y1"])
    return inter / (ar + br - inter)


kept = rejected = minor = major = klass = 0
dets = 0
for f in test:
    for d in f.get("stored_detections", []):
        dets += 1
        v = d["verdict"]
        if v == "kept":
            kept += 1
        elif v == "rejected":
            rejected += 1
        elif v == "corrected":
            cor = d["correction"]
            if cor["class"] != d["class"]:
                klass += 1
            elif iou(d["box_n"], cor["box_n"]) >= 0.7:
                minor += 1
            else:
                major += 1
added = sum(1 for f in test for g in f["ground_truth"] if g["origin"] == "added")
gt = dets - rejected + added
tp = kept + minor
fp = major + klass + rejected
fn = added + major + klass
print(f"frames revisados      {len(test)}")
print(f"deteccoes propostas   {dets}")
print(f"  kept                {kept:5}  ({100*kept/dets:5.1f}%)")
print(f"  corrigida (leve)    {minor:5}  ({100*minor/dets:5.1f}%)")
print(f"  corrigida (refeita) {major:5}  ({100*major/dets:5.1f}%)")
print(f"  classe trocada      {klass:5}  ({100*klass/dets:5.1f}%)")
print(f"  rejeitada           {rejected:5}  ({100*rejected/dets:5.1f}%)")
print(f"adicionadas na mao    {added:5}")
print(f"gabarito final        {gt:5}   (added = {100*added/gt:.1f}% do gabarito)")
print(f"tp={tp} fp={fp} fn={fn}")
print(f"precisao={100*tp/(tp+fp):.1f}%  recall={100*tp/(tp+fn):.1f}%")

print()
print("=== contexto: composicao do gabarito por classe ===")
from collections import Counter
cc = Counter(g["class"] for f in test for g in f["ground_truth"])
for k, v in cc.most_common():
    print(f"  {k:16} {v:5}")
print(f"  TOTAL            {sum(cc.values()):5}")
