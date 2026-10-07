"""P3: VLM decomposed (v2) sobre EXATAMENTE as 969 propostas do P2.

Conjunto congelado: raw YOLO11x, conf >= 0.25, só as 6 classes do gabarito
(car, truck, bus, motorcycle, bicycle, person). Nada de train/airplane, nada de
tracking, nada de v1/v3.

O prompt, o schema, a renderização (cena + recorte) e a derivação do veredito
são os de `forge.vlm.phases.run_phase1(..., v2=True)`, chamados sem alteração —
um frame por vez só para poder gravar checkpoint. Retomável: rodar de novo pula
o que já tem veredito e refaz o que falhou.

A coluna `truth` gravada em cada linha é o veredito GEOMÉTRICO contra os 731
(vínculo τ=0.10, correta = mesma classe e IoU >= 0.70; ver
REPORT-RAW-YOLO11X-P2P3.md §3). Serve para auditoria; a pontuação é refeita em
`score_p2p3_969.py` a partir da geometria, não desta coluna.

Uso:
    .venv\\Scripts\\python.exe paper\\repro\\raw_yolo11x_313\\run_vlm_v2_969.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
FORGE = HERE.parents[2]
sys.path.insert(0, str(FORGE / "src"))
sys.path.insert(0, str(HERE))

from analyze_raw_p2p3 import geometric_verdict, link, load_frames, load_gt, Box  # noqa: E402
from forge.core.geometry import BoxN  # noqa: E402
from forge.vlm.client import VlmClient, VlmError  # noqa: E402
from forge.vlm.phases import Phase1Row, phase1_prompt_v2, run_phase1  # noqa: E402
from forge.vlm.prepare import PHASE1_SCHEMA_V2, BoxTask  # noqa: E402

import os
ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"
SIX = ("car", "truck", "bus", "motorcycle", "bicycle", "person")
CONF_MIN = 0.25
EXPECTED = 969
FROZEN = HERE / "p2p3_frozen_969.json"
OUT = HERE / "vlm_v2_969"
CKPT = OUT / "phase1.jsonl"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def freeze() -> list[dict]:
    """Lê o manifest do P2 e congela o conjunto. Se já existe, confere que é igual."""
    man = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    props = [p for f in man["frames"] for p in f["proposals"]
             if p["class"] in SIX and p["conf"] >= CONF_MIN]
    if len(props) != EXPECTED:
        raise SystemExit(f"ERRO: esperado {EXPECTED} propostas, veio {len(props)}")
    doc = {"_meta": {"source": "manifest.json", "source_sha256": sha256_bytes(
               (HERE / "manifest.json").read_bytes()),
               "rule": "class in 6 GT classes AND conf >= 0.25", "classes": list(SIX),
               "count": len(props)},
           "proposals": props}
    blob = json.dumps(doc, indent=1, ensure_ascii=False).encode("utf-8")
    if FROZEN.exists():
        if FROZEN.read_bytes() != blob:
            raise SystemExit("ERRO: p2p3_frozen_969.json existente difere do manifest atual")
    else:
        FROZEN.write_bytes(blob)
    return props


def build_tasks(props: list[dict]) -> list[BoxTask]:
    frames = load_frames()
    gt = load_gt(frames)
    boxes = [Box(f"{p['acq_id']}:{p['sec']}", p["proposal_id"], p["class"], p["conf"],
                 BoxN(*p["box_n"])) for p in props]
    p2g, _ = link(boxes, gt, 0.1)
    return [BoxTask(acq_id=p["acq_id"], sec=p["sec"], det_index=p["rank_in_frame"],
                    det_id=p["proposal_id"], proposed_class=p["class"], conf=p["conf"],
                    box=b.b, truth=geometric_verdict(b, p2g.get(b.uid)))
            for p, b in zip(props, boxes)]


def load_ckpt() -> dict[str, dict]:
    done: dict[str, dict] = {}
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["key"]] = r  # a última linha de uma chave vence
    return done


def main() -> int:
    props = freeze()
    tasks = build_tasks(props)
    OUT.mkdir(exist_ok=True)
    done = {k: r for k, r in load_ckpt().items() if r.get("vlm") and not r.get("error")}
    todo = [t for t in tasks if t.key not in done]
    print(f"{len(tasks)} propostas congeladas; {len(done)} já com veredito; {len(todo)} a fazer")
    # `--limit N`: no máximo N chamadas nesta sessão. O llama-server do Ollama
    # acumula memória privada ao longo das chamadas (16 GB comprometidos em
    # ~470, medido em 10/09/2026) e a máquina tem 15,8 GB; em lotes, com o
    # Ollama reiniciado entre eles, a rodada termina. Não muda prompt, schema,
    # renderização nem ordem das caixas.
    if "--limit" in sys.argv:
        todo = todo[: int(sys.argv[sys.argv.index("--limit") + 1])]
        print(f"--limit: {len(todo)} nesta sessão")

    def image_of(acq: str, sec: int) -> Path:
        return ROOT / "images" / acq / f"{sec:06d}.jpg"

    by_frame: dict[tuple[str, int], list[BoxTask]] = {}
    for t in todo:
        by_frame.setdefault((t.acq_id, t.sec), []).append(t)

    t0 = time.time()
    n = 0
    with VlmClient() as client:
        version = client.health()
        tags = client._client.get("/api/tags").json().get("models", [])
        digest = next((m.get("digest") for m in tags if m.get("name") == client.model), None)
        with CKPT.open("a", encoding="utf-8") as fh:
            for (acq, sec), group in by_frame.items():
                rows = run_phase1(client, group, image_of, v2=True)
                for r in rows:
                    fh.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
                fh.flush()
                n += len(rows)
                if n % 50 < len(rows):
                    el = time.time() - t0
                    print(f"  {n}/{len(todo)}  {el/60:.1f} min  "
                          f"{el/max(n,1):.2f} s/caixa  falhas {client.usage.failures}", flush=True)
        usage = client.usage.as_dict()

    final = load_ckpt()
    rows = [final[t.key] for t in tasks if t.key in final]
    (OUT / "phase1.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False),
                                     encoding="utf-8")
    meta_path = OUT / "meta.json"
    prev = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    sessions = prev.get("sessions", []) + [{
        "finished_at": datetime.now(timezone.utc).isoformat(), "calls_this_session": len(todo),
        "usage": usage}]
    try:
        commit = subprocess.check_output(["git", "-C", str(FORGE), "rev-parse", "HEAD"],
                                         text=True).strip()
    except Exception:
        commit = None
    meta = {
        "arm": "YOLO11x raw (conf>=0.25, 6 GT classes) + VLM decomposed v2",
        "frozen_set": str(FROZEN), "frozen_set_sha256": sha256_bytes(FROZEN.read_bytes()),
        "proposals": len(tasks), "rows_with_verdict": sum(1 for r in rows if r.get("vlm")),
        "rows_with_error": sum(1 for r in rows if r.get("error")),
        "model": client.model, "model_digest": digest, "ollama_version": version,
        "temperature": client.temperature, "schema": "PHASE1_SCHEMA_V2",
        "schema_sha256": sha256_bytes(json.dumps(PHASE1_SCHEMA_V2, sort_keys=True).encode()),
        "prompt_fn": "forge.vlm.phases.phase1_prompt_v2 (unchanged)",
        "first_prompt_sha256": sha256_bytes(phase1_prompt_v2(tasks[0]).encode()),
        "render": "forge.vlm.prepare.draw_box + crop_around (unchanged)",
        "python": platform.python_version(), "forge_git_commit": commit,
        "sessions": sessions,
    }
    meta_path.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("proposals", "rows_with_verdict", "rows_with_error",
                                           "model_digest", "ollama_version")}, indent=1))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VlmError as exc:
        print(f"ERRO VLM: {exc}", file=sys.stderr)
        raise SystemExit(2)
