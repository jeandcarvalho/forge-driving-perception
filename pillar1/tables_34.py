"""Table 3 (densidade) e Table 4 (redundancia) — colecao x braco aleatorio LIMPO.

O que mudou em 19/08/2026: o braco `random-n4364-s42` foi sorteado com
`exclude_mode=acqs`, que so tira as 127 gravacoes DOADORAS. Sobravam no pool as
44 gravacoes nao-doadoras da propria colecao, e 1.371 dos 4.364 frames (31%) do
"aleatorio" saiam de gravacao curada. O braco novo usa `exclude_mode=acqs_all`:
pool de 29.671 frames em 113 gravacoes que a colecao nunca tocou.

A colecao NAO muda — ela e a colecao. Muda o braco de controle.

So le e conta. Nao escreve braco, split nem rotulo. Os 313 frames do gabarito
sao lidos como imagem para contar deteccao do professor; nenhuma anotacao humana
e tocada.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
HERE = Path(__file__).resolve().parent
CACHE = HERE / "collection_density.json"
CONF, IOU, IMGSZ = 0.25, 0.7, 640
R = 6371008.8

ARM_OLD = "random-n4364-s42"
ARM_NEW = "random-n4364-s42-acqsall"


# -- densidade ---------------------------------------------------------------

def count_label_dir(d: Path) -> tuple[int, int, int]:
    """(frames, objetos, vazios) de um diretorio de rotulos YOLO."""
    frames = objs = empty = 0
    for f in d.glob("*.txt"):
        n = len([ln for ln in f.read_text().splitlines() if ln.strip()])
        frames += 1
        objs += n
        empty += n == 0
    return frames, objs, empty


def collection_density(force: bool) -> dict:
    """Colecao completa sob o mesmo rotulador.

    Cacheia porque roda o professor em 2.958 imagens (test + discarded); os 1.406
    rotulos do split train ja estao em disco e sao reaproveitados.
    """
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text(encoding="utf-8"))

    from ultralytics import YOLO

    man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    by_split = {"test": [], "train": [], "discarded": []}
    for f in man["frames"]:
        p = ROOT / "images" / f["acq_id"] / f"{f['sec']:06d}.jpg"
        if p.exists():
            by_split[f["split"]].append(p)

    train_dir = ROOT / "labels" / "auto" / "yolo11x_c025"
    tr_frames, tr_objs, tr_empty = count_label_dir(train_dir)
    if tr_frames != len(by_split["train"]):
        raise SystemExit(
            f"train: {tr_frames} rotulos para {len(by_split['train'])} frames — "
            "rode `forge label --split train` antes"
        )

    model = YOLO(str(HERE.parent / "yolo11x.pt"))
    per = {"train": [tr_frames, tr_objs, tr_empty]}
    for split in ("test", "discarded"):
        paths = by_split[split]
        listfile = HERE / f"_measure_{split}.txt"
        listfile.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
        objs = empty = 0
        for r in model.predict(source=str(listfile), conf=CONF, iou=IOU, imgsz=IMGSZ,
                               device=0, batch=8, stream=True, verbose=False):
            n = 0 if r.boxes is None else len(r.boxes)
            objs += n
            empty += n == 0
        per[split] = [len(paths), objs, empty]
        listfile.unlink(missing_ok=True)

    out = {
        "frames": sum(v[0] for v in per.values()),
        "objects": sum(v[1] for v in per.values()),
        "empty": sum(v[2] for v in per.values()),
        "per_split": per,
    }
    CACHE.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


# -- redundancia -------------------------------------------------------------

def nn_within_acq(by_acq: dict) -> np.ndarray:
    """Distancia ao vizinho mais proximo DENTRO da mesma gravacao."""
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


def gps_by_acq(items, get_gps, get_acq):
    by = defaultdict(list)
    n = 0
    for it in items:
        g = get_gps(it)
        if g and g.get("lat") is not None:
            by[get_acq(it)].append((g["lat"], g["lon"]))
            n += 1
    return by, n


def spacing_row(label, by_acq, n_frames) -> dict:
    arr = nn_within_acq(by_acq)
    return {
        "label": label, "frames": n_frames, "acqs": len(by_acq),
        "min": float(arr.min()), "p05": float(np.percentile(arr, 5)),
        "median": float(np.median(arr)),
        "lt10_n": int((arr < 10).sum()), "lt10_pct": float(100 * (arr < 10).mean()),
        "lt25_n": int((arr < 25).sum()), "lt25_pct": float(100 * (arr < 25).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force-collection", action="store_true",
                    help="reroda o professor na colecao em vez de usar o cache")
    args = ap.parse_args()

    man = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    arms = {}
    for name in (ARM_OLD, ARM_NEW):
        p = ROOT / "arms" / f"{name}.json"
        if p.exists():
            arms[name] = json.loads(p.read_text(encoding="utf-8"))

    # -- TABLE 3 --
    col = collection_density(args.force_collection)
    rows = [("colecao", col["frames"], col["objects"], col["empty"])]
    for name in (ARM_OLD, ARM_NEW):
        d = ROOT / "labels" / "auto" / f"{name}_yolo11x_c025"
        if d.exists():
            rows.append((f"aleatorio {name}", *count_label_dir(d)))

    print("=" * 80)
    print("TABLE 3 - densidade (YOLO11x @0.25, mesmo instrumento nos dois lados)")
    print("=" * 80)
    print(f"{'braco':<36}{'frames':>7}{'objetos':>9}{'obj/frame':>11}{'vazios':>8}{'% vazio':>9}")
    for lbl, fr, ob, em in rows:
        print(f"{lbl:<36}{fr:>7}{ob:>9}{ob / fr:>11.2f}{em:>8}{100 * em / fr:>8.1f}%")
    cf, co, ce = rows[0][1], rows[0][2], rows[0][3]
    for lbl, fr, ob, em in rows[1:]:
        r_col, r_arm = ce / cf, em / fr
        raz = "inf" if r_col == 0 else f"{r_arm / r_col:.0f}x"
        print(f"\n  colecao / {lbl}:")
        print(f"    densidade    {co / cf:.2f} vs {ob / fr:.2f}  ->  {(co / cf) / (ob / fr):.2f}x")
        print(f"    frame vazio  {100 * r_col:.1f}% vs {100 * r_arm:.1f}%  ->  {raz} menos")
    print("\n  composicao da colecao: " + " | ".join(
        f"{k} {v[1]} obj ({v[2]} vazios) em {v[0]} frames" for k, v in col["per_split"].items()))

    # -- TABLE 4 --
    print()
    print("=" * 80)
    print("TABLE 4 - redundancia intra-gravacao (vizinho mais proximo na mesma gravacao)")
    print("=" * 80)
    by, n = gps_by_acq(man["frames"],
                       lambda f: (f.get("telemetry") or {}).get("gps"),
                       lambda f: f["acq_id"])
    out = [spacing_row("colecao", by, n)]
    for name in (ARM_OLD, ARM_NEW):
        if name not in arms:
            continue
        by, n = gps_by_acq(arms[name]["frames"], lambda f: f.get("gps"), lambda f: f["acq_id"])
        out.append(spacing_row(f"aleatorio {name}", by, n))
    print(f"{'braco':<36}{'frames':>7}{'acqs':>6}{'min':>7}{'P05':>7}{'mediana':>9}"
          f"{'<10m':>16}{'<25m':>16}")
    for r in out:
        print(f"{r['label']:<36}{r['frames']:>7}{r['acqs']:>6}{r['min']:>7.1f}{r['p05']:>7.1f}"
              f"{r['median']:>9.1f}{r['lt10_pct']:>9.1f}% ({r['lt10_n']:>4})"
              f"{r['lt25_pct']:>9.1f}% ({r['lt25_n']:>4})")
    base = out[0]
    for r in out[1:]:
        f10 = "inf" if base["lt10_pct"] == 0 else f"{r['lt10_pct'] / base['lt10_pct']:.0f}x"
        f25 = "inf" if base["lt25_pct"] == 0 else f"{r['lt25_pct'] / base['lt25_pct']:.0f}x"
        print(f"\n  colecao / {r['label']}:  <10m {f10} menos  |  <25m {f25} menos")

    # -- cobertura de gravacoes (Pilar 1C) --
    print()
    print("=" * 80)
    print("COBERTURA DE GRAVACOES (Pilar 1C)")
    print("=" * 80)
    for r in out:
        print(f"{r['label']:<36}{r['acqs']:>5} gravacoes  |  "
              f"{r['frames'] / r['acqs']:.1f} frames por gravacao")

    (HERE / "tables_34.json").write_text(json.dumps(
        {"table3": [{"label": l, "frames": f, "objects": o, "empty": e} for l, f, o, e in rows],
         "table4": out}, indent=2), encoding="utf-8")
    print(f"\nresultados -> {HERE / 'tables_34.json'}")


if __name__ == "__main__":
    main()
