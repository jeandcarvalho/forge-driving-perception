"""Metricas de CONTEUDO do Pillar 1: detections/frame e empty-frame rate.

Bracos: forge / naive_random / random_30m, N=4.364, sementes 42-44 nos random
(as MESMAS tres que a Table 3 antiga usou — labels/auto/random-n4364-s{42,43,44}
-full_yolo11x_c025).

Instrumento, identico ao de tables_34.py e measure_density.py:
    YOLO11x (yolo11x.pt), conf=0.25, iou=0.7, imgsz=640, device=0, batch=8,
    stream=True. Deteccao = `len(r.boxes)` do ultralytics, sem filtro de classe
    (as 80 do COCO). Frame vazio = `n == 0`.

Inferencia UMA VEZ por imagem distinta da uniao dos bracos, e agregacao por
braco a partir desse cache. Isso garante, por construcao, que os tres bracos sao
medidos com exatamente o mesmo instrumento — inclusive nos frames que aparecem
em mais de um braco.

Nao sobrescreve nada antigo: escreve so os arquivos `pillar1-content-*`.
"""

from __future__ import annotations

import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

import os
DATA = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
LISTS = Path(os.environ.get("FORGE_P1_FRAMELISTS", "pillar1-content-framelists-s42-51.json"))  # private: contains Drive ids
IMAGES = DATA / "images"
OUT_DIR = Path(os.environ.get("FORGE_OUT", "out"))
CACHE = OUT_DIR / "pillar1-content-detcounts.json"
WEIGHTS = HERE.parent / "yolo11x.pt"
OLD_LABELS = DATA / "labels" / "auto"

CONF, IOU, IMGSZ, BATCH, DEVICE = 0.25, 0.7, 640, 8, 0
CHECKPOINT = 1000  # imagens entre gravacoes do cache


def path_of(acq_id: str, sec: int) -> Path:
    return IMAGES / acq_id / f"{sec:06d}.jpg"


def key_of(p: Path) -> str:
    return f"{p.parent.name}/{int(p.stem)}"


def infer_union(paths: list[Path], counts: dict[str, int]) -> dict[str, int]:
    """{'<acq>/<sec>': n_deteccoes} — uma passada, mesmo corte do professor.

    Grava o cache a cada CHECKPOINT imagens: a rodada leva ~17 min e o shell que
    a invoca tem teto de 10, entao a inferencia precisa ser retomavel. Rodar de
    novo so infere o que ainda nao esta no cache.
    """
    from ultralytics import YOLO

    listfile = OUT_DIR / "_pillar1_content_union.txt"
    listfile.write_text("\n".join(str(p) for p in paths), encoding="utf-8")
    model = YOLO(str(WEIGHTS))
    t0 = time.time()
    n = 0
    for r in model.predict(source=str(listfile), conf=CONF, iou=IOU, imgsz=IMGSZ,
                           device=DEVICE, batch=BATCH, stream=True, verbose=False):
        counts[key_of(Path(r.path))] = 0 if r.boxes is None else len(r.boxes)
        n += 1
        if n % CHECKPOINT == 0:
            CACHE.write_text(json.dumps(counts), encoding="utf-8")
            el = time.time() - t0
            print(f"  {n}/{len(paths)}  {n/el:.1f} img/s  eta {(len(paths)-n)/(n/el)/60:.1f} min "
                  f"(cache: {len(counts)})", flush=True)
    listfile.unlink(missing_ok=True)
    CACHE.write_text(json.dumps(counts), encoding="utf-8")
    print(f"  inferencia: {n} imagens em {(time.time()-t0)/60:.1f} min")
    return counts


def quant(v: list[int], q: float) -> float:
    s = sorted(v)
    idx = q * (len(s) - 1)
    lo, hi = int(idx), min(len(s) - 1, int(idx) + 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def row(arm: str, seed, spacing: int, frames: list[dict], counts: dict[str, int]) -> dict:
    v = [counts[f'{f["acq_id"]}/{f["sec"]}'] for f in frames
         if f'{f["acq_id"]}/{f["sec"]}' in counts]
    missing = len(frames) - len(v)
    empty = sum(1 for x in v if x == 0)
    return {
        "arm": arm, "seed": seed, "spacing_m": spacing,
        "n_frames": len(v), "n_missing_image": missing,
        "detections": sum(v),
        "det_per_frame": sum(v) / len(v),
        "empty_n": empty, "empty_pct": 100 * empty / len(v),
        "det_mean": sum(v) / len(v), "det_median": quant(v, 0.5),
        "det_p25": quant(v, 0.25), "det_p75": quant(v, 0.75),
        "det_min": min(v), "det_max": max(v),
    }


def agg(rows: list[dict], key: str) -> dict:
    v = [r[key] for r in rows]
    return {"mean": st.mean(v), "sd": st.stdev(v) if len(v) > 1 else 0.0,
            "min": min(v), "max": max(v)}


def count_label_dir(d: Path) -> tuple[int, int, int]:
    """Contagem pelo caminho ANTIGO (diretorio de rotulos), para conferencia."""
    frames = objs = empty = 0
    for f in d.glob("*.txt"):
        n = len([ln for ln in f.read_text().splitlines() if ln.strip()])
        frames += 1
        objs += n
        empty += n == 0
    return frames, objs, empty


def main() -> None:
    d = json.loads(LISTS.read_text(encoding="utf-8"))
    arms = d["arms"]

    union: dict[str, Path] = {}
    for a in arms:
        for f in a["frames"]:
            p = path_of(f["acq_id"], f["sec"])
            union.setdefault(f'{f["acq_id"]}/{f["sec"]}', p)
    on_disk = {k: p for k, p in union.items() if p.is_file()}
    print(f"uniao de frames: {len(union)}   com imagem em disco: {len(on_disk)}   "
          f"faltando: {len(union) - len(on_disk)}")
    if len(union) != len(on_disk):
        print("ATENCAO: ha frames sem imagem; rode pillar1_content_fetch.py antes.")

    counts = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.is_file() else {}
    falta = [k for k in on_disk if k not in counts]
    print(f"cache: {len(counts)} contagens ja feitas; faltando inferir {len(falta)}")
    if falta:
        print(f"YOLO11x conf={CONF} iou={IOU} imgsz={IMGSZ} batch={BATCH} device={DEVICE}")
        infer_union([on_disk[k] for k in falta], counts)
    print(f"cache -> {CACHE}")
    if len(falta) and any(k not in counts for k in on_disk):
        restante = sum(1 for k in on_disk if k not in counts)
        print(f"INCOMPLETO: faltam {restante} imagens. Rode de novo para continuar.")
        return

    rows = [row(a["arm"], a["seed"], a["spacing_m"], a["frames"], counts) for a in arms]

    # ── conferencia contra o caminho de contagem ANTIGO (labels/*.txt) ──
    check = []
    for seed in (42, 43, 44):
        dirp = OLD_LABELS / f"random-n4364-s{seed}-full_yolo11x_c025"
        if not dirp.is_dir():
            continue
        fr, ob, em = count_label_dir(dirp)
        mine = next(r for r in rows if r["arm"] == "naive_random" and r["seed"] == seed)
        check.append({
            "seed": seed, "old_dir": dirp.name,
            "old_frames": fr, "old_objects": ob, "old_empty": em,
            "new_frames": mine["n_frames"], "new_detections": mine["detections"],
            "new_empty": mine["empty_n"],
            "delta_objects": mine["detections"] - ob, "delta_empty": mine["empty_n"] - em,
        })

    summary = []
    for arm in ("naive_random", "random_30m"):
        sub = [r for r in rows if r["arm"] == arm]
        s = {"arm": arm, "spacing_m": sub[0]["spacing_m"], "seeds": len(sub)}
        for k in ("n_frames", "detections", "det_per_frame", "empty_n", "empty_pct",
                  "det_median", "det_p25", "det_p75", "det_max"):
            s[k] = agg(sub, k)
        summary.append(s)

    out = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "instrument": {"weights": WEIGHTS.name, "conf": CONF, "iou": IOU, "imgsz": IMGSZ,
                       "batch": BATCH, "device": DEVICE,
                       "detection": "len(r.boxes), 80 classes COCO, sem filtro",
                       "empty_frame": "len(r.boxes) == 0"},
        "n_target": 4364, "seeds": [42, 43, 44, 45, 46, 47, 48, 49, 50, 51],
        "union_frames": len(union), "union_inferred": len(on_disk),
        "rows": rows, "summary": summary, "reproduction_check": check,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "pillar1-content-arms-s42-51.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    cols = list(rows[0].keys())
    (OUT_DIR / "pillar1-content-arms-s42-51.csv").write_text(
        "\n".join([",".join(cols)] + [",".join(str(r[c]) for c in cols) for r in rows]),
        encoding="utf-8")

    print("\n=== CONFERENCIA contra labels/auto antigos (naive_random) ===")
    for c in check:
        print(f"  s{c['seed']}: objetos antigo={c['old_objects']} novo={c['new_detections']} "
              f"(delta {c['delta_objects']:+})   vazios antigo={c['old_empty']} "
              f"novo={c['new_empty']} (delta {c['delta_empty']:+})")

    print("\n=== POR BRACO / SEED ===")
    print(f"{'braco':<14}{'seed':>5}{'N':>7}{'det':>8}{'det/frame':>11}{'vazios':>8}{'% vazio':>9}"
          f"{'med':>5}{'p25':>5}{'p75':>5}{'max':>5}")
    for r in rows:
        print(f"{r['arm']:<14}{str(r['seed']):>5}{r['n_frames']:>7}{r['detections']:>8}"
              f"{r['det_per_frame']:>11.3f}{r['empty_n']:>8}{r['empty_pct']:>8.2f}%"
              f"{r['det_median']:>5.0f}{r['det_p25']:>5.0f}{r['det_p75']:>5.0f}{r['det_max']:>5.0f}")

    print(f"\n=== AGREGADO ({summary[0]['seeds']} sementes) ===")
    for s in summary:
        print(f"  {s['arm']:<14} det/frame {s['det_per_frame']['mean']:.3f} +- {s['det_per_frame']['sd']:.3f} "
              f"[{s['det_per_frame']['min']:.3f}, {s['det_per_frame']['max']:.3f}]   "
              f"vazio {s['empty_pct']['mean']:.2f}% +- {s['empty_pct']['sd']:.2f} "
              f"[{s['empty_pct']['min']:.2f}, {s['empty_pct']['max']:.2f}]")

    print(f"\n-> {OUT_DIR / 'pillar1-content-arms-s42-51.json'}")
    print(f"-> {OUT_DIR / 'pillar1-content-arms-s42-51.csv'}")


if __name__ == "__main__":
    main()
