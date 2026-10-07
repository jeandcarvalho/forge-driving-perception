"""YOLO11x PURO, quadro a quadro, nos 313 frames do gabarito travado.

Partes A e B do refazimento de P2/P3. Somente leitura das fontes; escreve só
nesta pasta. Não toca Mongo, não usa `yolo[]` do banco, não usa as 505
`stored_detections`, não usa o run de 16/08 (ele só aparece depois, como
checagem cruzada, em `analyze_raw_vs_gt.py`).

O que é "puro" aqui, e é verificado no código, não só declarado:
  - `model.predict` (nunca `model.track`), uma chamada por JPEG, `batch=1`;
  - sem `persist`, sem tracker: depois da rodada `model.predictor` não pode ter
    atributo `trackers` (é onde o ultralytics pendura o BYTETracker/BoT-SORT);
  - sem pós-processamento: a caixa gravada é `boxes.xyxy` do resultado, em px
    do frame original 1920x1080, sem arredondar.

Parâmetros — de onde vem cada um:
  pesos   yolo11x.pt, sha256 travado = o do run 20260816T220716Z-infer
  imgsz   640   = config.INFER_IMGSZ = default do ultralytics
  NMS IoU 0.70  = config.INFER_NMS_IOU = default do ultralytics
  conf    0.001 = config.INFER_CONF (cauda inteira, para AP). O conjunto de
          PROPOSTAS é um corte pós-hoc dessa cauda: conf >= 0.25 (default do
          `predict` do ultralytics = config.LABEL_CONF). 0.20 fica como
          sensibilidade, porque é o piso medido das propostas antigas.
  classes todas as 80 COCO (classes=None); o escopo de avaliação é decidido
          na análise, não na inferência.

Uso:
    .venv\\Scripts\\python.exe paper\\repro\\raw_yolo11x_313\\run_raw_yolo11x.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

FORGE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(FORGE / "src"))

from forge.api.models import Frame  # noqa: E402
from forge.config import INFER_CONF, INFER_IMGSZ, INFER_NMS_IOU, LABEL_CONF  # noqa: E402
from forge.core.classes import class_id, is_valid_class  # noqa: E402
from forge.data.layout import DatasetLayout  # noqa: E402
from forge.data.writers.coco import frame_image_id  # noqa: E402

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
LAYOUT = DatasetLayout(root=ROOT)
OUT = Path(__file__).resolve().parent

WEIGHTS = FORGE / "yolo11x.pt"
WEIGHTS_SHA256 = "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"

PROPOSAL_CONF = LABEL_CONF          # 0.25
SENSITIVITY_CONF = 0.20
#: Diagnóstico da Parte D: qual imgsz o pipeline antigo usou? Não entra em P2/P3.
DIAGNOSTIC_IMGSZ = (1280,)


class Problem(Exception):
    pass


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(FORGE), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return None


def test_frames() -> list[Frame]:
    man = json.loads(LAYOUT.manifest_path.read_text(encoding="utf-8"))
    frames = [Frame.model_validate(f) for f in man["frames"] if f["split"] == "test"]
    frames.sort(key=lambda f: (f.acq_id, f.sec))
    if len(frames) != 313 or not all(f.reviewed for f in frames):
        raise Problem(f"esperado 313 frames de teste revisados, veio {len(frames)}")
    return frames


def verified_images(frames: list[Frame]) -> dict[str, tuple[Path, str]]:
    """Cada JPEG tem de bater com o checksums.sha256 do pull."""
    expected: dict[str, str] = {}
    for line in LAYOUT.checksums_path.read_text(encoding="utf-8").splitlines():
        h, _, rel = line.partition("  ")
        expected[rel.strip()] = h.strip()
    out = {}
    for f in frames:
        p = LAYOUT.image_path(f.acq_id, f.sec)
        rel = p.relative_to(ROOT).as_posix()
        got = sha256(p)
        if expected.get(rel) != got:
            raise Problem(f"{rel}: sha256 {got} != checksums.sha256 {expected.get(rel)}")
        out[f.key] = (p, got)
    return out


def infer(model, frames, images, *, imgsz: int) -> tuple[list[dict], dict]:
    """Uma chamada `predict` por frame. Devolve a cauda inteira (conf >= 0.001)."""
    rows: list[dict] = []
    names = model.names
    t0 = time.perf_counter()
    for f in frames:
        path, _ = images[f.key]
        res = model.predict(source=str(path), conf=INFER_CONF, iou=INFER_NMS_IOU, imgsz=imgsz,
                            device=0, batch=1, half=False, augment=False, agnostic_nms=False,
                            classes=None, max_det=300, verbose=False)
        if len(res) != 1:
            raise Problem(f"{f.key}: predict devolveu {len(res)} resultados")
        r = res[0]
        h, w = r.orig_shape
        if (w, h) != (1920, 1080):
            raise Problem(f"{f.key}: imagem {w}x{h}")
        b = r.boxes
        xyxy = b.xyxy.cpu().tolist() if b is not None else []
        conf = b.conf.cpu().tolist() if b is not None else []
        cls = b.cls.cpu().tolist() if b is not None else []
        dets = []
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls, strict=True):
            name = names[int(k)]
            dets.append({"class": name, "class_id": class_id(name), "conf": float(c),
                         "box_px": [x1, y1, x2, y2]})
        # id local estável: posto por confiança decrescente dentro do frame
        dets.sort(key=lambda d: (-d["conf"], d["class_id"], d["box_px"][0], d["box_px"][1]))
        for i, d in enumerate(dets):
            x1, y1, x2, y2 = d["box_px"]
            rows.append({
                "proposal_id": f"raw11x:{f.acq_id}:{f.sec}:{i:03d}",
                "acq_id": f.acq_id, "sec": f.sec, "rank_in_frame": i,
                "image_path": str(path),
                "class": d["class"], "class_id": d["class_id"], "conf": d["conf"],
                "box_px": [x1, y1, x2, y2],
                "box_n": [x1 / w, y1 / h, x2 / w, y2 / h],
                "image_w": w, "image_h": h,
            })
    elapsed = time.perf_counter() - t0
    return rows, {"seconds": round(elapsed, 2), "images": len(frames)}


def predictor_args(model) -> dict:
    a = vars(model.predictor.args)
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in sorted(a.items())
            if isinstance(v, (str, int, float, bool, type(None), list, tuple, Path))}


def letterbox_shape(model, path: Path) -> list[int]:
    import cv2
    im = cv2.imread(str(path))
    return list(model.predictor.pre_transform([im])[0].shape)


def coco_results(rows: list[dict]) -> list[dict]:
    """Mesmo formato do `forge infer` (COCO results, category_id = índice YOLO)."""
    out = []
    for r in rows:
        x1, y1, x2, y2 = r["box_px"]
        out.append({"image_id": frame_image_id(r["acq_id"], r["sec"]),
                    "category_id": r["class_id"], "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": r["conf"]})
    return out


def write_manifest(rows: list[dict], frames: list[Frame], path_json: Path, path_csv: Path,
                   conf_min: float, meta: dict) -> dict:
    props = [r for r in rows if r["conf"] >= conf_min]
    by_frame: dict[str, list[dict]] = {f.key: [] for f in frames}
    for r in props:
        by_frame[f"{r['acq_id']}:{r['sec']}"].append(r)
    doc = {
        "_meta": {**meta, "conf_min": conf_min, "proposals": len(props),
                  "frames": len(frames),
                  "empty_frames": sum(1 for v in by_frame.values() if not v)},
        "frames": [{"acq_id": f.acq_id, "sec": f.sec,
                    "image_path": str(LAYOUT.image_path(f.acq_id, f.sec)),
                    "proposals": by_frame[f.key]} for f in frames],
    }
    path_json.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    cols = ["proposal_id", "acq_id", "sec", "rank_in_frame", "image_path", "class", "class_id",
            "conf", "x1_n", "y1_n", "x2_n", "y2_n", "x1_px", "y1_px", "x2_px", "y2_px"]
    with path_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in props:
            w.writerow([r["proposal_id"], r["acq_id"], r["sec"], r["rank_in_frame"],
                        r["image_path"], r["class"], r["class_id"], f"{r['conf']:.6f}",
                        *[f"{v:.6f}" for v in r["box_n"]], *[f"{v:.3f}" for v in r["box_px"]]])
    return doc["_meta"]


def main() -> int:
    import torch
    import ultralytics
    from ultralytics import YOLO

    for p in (OUT / "manifest.json", OUT / "raw_predictions_tail.json"):
        if p.exists():
            raise Problem(f"{p} já existe — não sobrescrevo resultado")

    if sha256(WEIGHTS) != WEIGHTS_SHA256:
        raise Problem(f"{WEIGHTS}: sha256 diferente do travado")
    frames = test_frames()
    images = verified_images(frames)
    src_manifest_sha = sha256(LAYOUT.manifest_path)
    print(f"313 frames, 313 JPEGs conferidos com checksums.sha256; pesos {WEIGHTS_SHA256[:12]}")

    model = YOLO(str(WEIGHTS))
    bad = sorted({n for n in model.names.values() if not is_valid_class(n)})
    if bad or len(model.names) != 80:
        raise Problem(f"cabeça fora do COCO-80: {bad}")

    rows, timing = infer(model, frames, images, imgsz=INFER_IMGSZ)
    if hasattr(model.predictor, "trackers"):
        raise Problem("predictor tem `trackers` — algum tracker foi registrado")
    args = predictor_args(model)
    lb = letterbox_shape(model, images[frames[0].key][0])
    print(f"imgsz {INFER_IMGSZ}: {len(rows)} predicoes na cauda, {timing['seconds']} s")

    # Repetição idêntica: mede determinismo da GPU nesta máquina.
    rows_rep, _ = infer(model, frames, images, imgsz=INFER_IMGSZ)
    same = (len(rows_rep) == len(rows) and all(
        a["proposal_id"] == b["proposal_id"] and a["class"] == b["class"]
        and a["conf"] == b["conf"] and a["box_px"] == b["box_px"] for a, b in zip(rows, rows_rep)))
    max_dconf = max((abs(a["conf"] - b["conf"]) for a, b in zip(rows, rows_rep)), default=0.0)
    print(f"repeticao bit a bit identica: {same} (max |dconf| {max_dconf:.2e})")

    diag = {}
    for s in DIAGNOSTIC_IMGSZ:
        drows, dt = infer(model, frames, images, imgsz=s)
        d = OUT / f"diagnostic_imgsz{s}"
        d.mkdir(exist_ok=True)
        (d / "raw_predictions_tail.json").write_text(json.dumps(drows), encoding="utf-8")
        diag[s] = {"predictions_tail": len(drows), **dt,
                   "file": str(d / "raw_predictions_tail.json")}
        print(f"diagnostico imgsz {s}: {len(drows)} predicoes")

    (OUT / "raw_predictions_tail.json").write_text(json.dumps(rows), encoding="utf-8")
    (OUT / "predictions_coco_tail.json").write_text(json.dumps(coco_results(rows)),
                                                    encoding="utf-8")

    env = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "forge_git_commit": git_commit(),
    }
    meta_common = {"weights": str(WEIGHTS), "weights_sha256": WEIGHTS_SHA256,
                   "imgsz": INFER_IMGSZ, "nms_iou": INFER_NMS_IOU,
                   "inference_conf": INFER_CONF, "mode": "predict, 1 call per JPEG, no tracker"}
    m25 = write_manifest(rows, frames, OUT / "manifest.json", OUT / "manifest.csv",
                         PROPOSAL_CONF, meta_common)
    m20 = write_manifest(rows, frames, OUT / "manifest_conf020_sensitivity.json",
                         OUT / "manifest_conf020_sensitivity.csv", SENSITIVITY_CONF, meta_common)

    config = {
        "part_A_parameters": {
            "weights_path": str(WEIGHTS), "weights_sha256": WEIGHTS_SHA256,
            "weights_bytes": WEIGHTS.stat().st_size,
            "task": model.task, "n_classes": len(model.names), "classes": "all 80 COCO",
            "imgsz": INFER_IMGSZ, "letterbox_input_shape_hwc": lb,
            "nms_iou": INFER_NMS_IOU, "inference_conf": INFER_CONF,
            "proposal_conf": PROPOSAL_CONF, "sensitivity_conf": SENSITIVITY_CONF,
            "device": "cuda:0", "batch": 1, "half": False, "augment": False,
            "agnostic_nms": False, "max_det": 300,
            "image_transform": ("none outside ultralytics: JPEG read by cv2 (BGR), ultralytics "
                                "LetterBox to imgsz with stride-32 rect padding, BGR->RGB, /255"),
            "tracker": None, "persist": False,
            "sources_in_code": {
                "INFER_CONF/INFER_NMS_IOU/INFER_IMGSZ": "src/forge/config.py:19-21",
                "LABEL_CONF (0.25, = ultralytics predict default)": "src/forge/config.py:30",
                "weights": "src/forge/config.py:29 TEACHER_WEIGHTS; sha from runs/20260816T220716Z-infer/meta.json",
            },
            "predictor_args_as_run": args,
        },
        "environment": env,
        "determinism_repeat_identical": same, "determinism_max_abs_dconf": max_dconf,
        "timing_imgsz640": timing,
        "diagnostic_runs_not_for_paper": diag,
        "inputs": {"manifest_json": str(LAYOUT.manifest_path),
                   "manifest_json_sha256": src_manifest_sha,
                   "images_verified_against": str(LAYOUT.checksums_path),
                   "images_sha256_list_sha256": hashlib.sha256("\n".join(
                       f"{k} {images[k][1]}" for k in sorted(images)).encode()).hexdigest()},
        "outputs": {"tail_predictions": len(rows), "proposals_conf025": m25,
                    "proposals_conf020": m20},
        "image_sha256": {k: v[1] for k, v in sorted(images.items())},
    }
    (OUT / "config.json").write_text(json.dumps(config, indent=1, ensure_ascii=False),
                                     encoding="utf-8")

    if sha256(LAYOUT.manifest_path) != src_manifest_sha:
        raise Problem("manifest.json de origem mudou durante a rodada")
    hashes = {p.relative_to(OUT).as_posix(): sha256(p) for p in sorted(OUT.rglob("*"))
              if p.is_file() and p.suffix in {".json", ".csv"} and p.name != "SHA256SUMS.txt"}
    (OUT / "SHA256SUMS.txt").write_text(
        "".join(f"{h}  {n}\n" for n, h in hashes.items()), encoding="utf-8")
    print(f"propostas conf>=0.25: {m25['proposals']} (frames vazios {m25['empty_frames']}); "
          f"conf>=0.20: {m20['proposals']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Problem as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        raise SystemExit(2)
