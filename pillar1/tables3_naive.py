"""Table 3 (densidade) com o braco de controle NAIVE, sobre varias sementes.

Le os rotulos do professor (yolo11x @0.25) ja em disco. NAO roda GPU e NAO
escreve nada alem do proprio JSON de resultado. A linha da colecao vem do cache
`collection_density.json`, medida com o MESMO instrumento.

Braços esperados em labels/auto/:
  random-n4364-s{42..}-full_yolo11x_c025   sorteio uniforme dos 80.584 do acervo
  random-n4364-s42-acqsall_yolo11x_c025    controle de 19/08 (pool 29.671)
  random-n4364-s42_yolo11x_c025            controle de 16/08 (pool 42.831)
"""
import json
import statistics as st
from pathlib import Path

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
HERE = Path(__file__).resolve().parent
AUTO = ROOT / "labels" / "auto"


def count_label_dir(d: Path):
    frames = objs = empty = 0
    for f in d.glob("*.txt"):
        n = len([ln for ln in f.read_text().splitlines() if ln.strip()])
        frames += 1
        objs += n
        empty += n == 0
    return frames, objs, empty


def row(label, d: Path):
    fr, ob, em = count_label_dir(d)
    return {"label": label, "frames": fr, "objects": ob, "empty": em,
            "per_frame": ob / fr, "empty_pct": 100 * em / fr}


def main():
    col = json.loads((HERE / "collection_density.json").read_text(encoding="utf-8"))
    c = {"label": "colecao curada", "frames": col["frames"], "objects": col["objects"],
         "empty": col["empty"], "per_frame": col["objects"] / col["frames"],
         "empty_pct": 100 * col["empty"] / col["frames"]}

    old = [(n, AUTO / f"{n}_yolo11x_c025") for n in
           ("random-n4364-s42", "random-n4364-s42-acqsall")]
    full = sorted(AUTO.glob("random-n4364-s*-full_yolo11x_c025"))

    rows_old = [row(f"aleatorio {n}", p) for n, p in old if p.is_dir()]
    rows_full = [row(f"aleatorio acervo {p.name.split('-')[2]}", p) for p in full
                 if len(list(p.glob('*.txt'))) >= 4364]

    hdr = f"{'braco':<40}{'frames':>7}{'objetos':>9}{'obj/frame':>11}{'vazios':>8}{'% vazio':>9}"
    print("=" * len(hdr))
    print("TABLE 3 - densidade (YOLO11x @0.25, mesmo instrumento nos dois lados)")
    print("=" * len(hdr))
    print(hdr)
    for r in [c] + rows_old + rows_full:
        print(f"{r['label']:<40}{r['frames']:>7}{r['objects']:>9}{r['per_frame']:>11.2f}"
              f"{r['empty']:>8}{r['empty_pct']:>8.1f}%")

    if not rows_full:
        print("\n(sem braco -full rotulado ainda)")
        return

    def agg(key):
        v = [r[key] for r in rows_full]
        return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)

    m_dens, s_dens = agg("per_frame")
    m_emp, s_emp = agg("empty_pct")
    print()
    print("=" * len(hdr))
    print(f"MEDIA +- DP sobre {len(rows_full)} sementes (braco NAIVE)")
    print("=" * len(hdr))
    print(f"  obj/frame      colecao {c['per_frame']:.2f}   x   aleatorio {m_dens:.2f} +- {s_dens:.2f}"
          f"   ->  {c['per_frame'] / m_dens:.1f}x mais objeto por frame")
    print(f"  frame vazio    colecao {c['empty_pct']:.1f}%  x   aleatorio {m_emp:.1f}% +- {s_emp:.1f}"
          f"   ->  {m_emp / c['empty_pct']:.0f}x menos frame vazio")
    print(f"  telas vazias que o sorteio abriria: {m_dens and round(m_emp * 4364 / 100)} de 4.364"
          f"   (colecao: {c['empty']})")

    (HERE / "tables3_naive.json").write_text(json.dumps(
        {"collection": c, "old": rows_old, "full": rows_full,
         "mean": {"per_frame": m_dens, "per_frame_sd": s_dens,
                  "empty_pct": m_emp, "empty_pct_sd": s_emp}}, indent=2), encoding="utf-8")
    print(f"\nresultados -> {HERE / 'tables3_naive.json'}")


if __name__ == "__main__":
    main()
