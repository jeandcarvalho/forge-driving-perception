"""Curadoria de pseudo-rótulo de TREINO pelo VLM v2, retomável.

Não é avaliação e não tem gabarito: aqui o VLM decide o que fica no rótulo, e
não existe verdade humana contra a qual medir acerto. A saída é um diretório de
rótulos novo — o pseudo-rótulo original do YOLO nunca é sobrescrito.

**Por que retomável.** São ~20 mil caixas a ~4,9 s, ou seja mais de um dia de
GPU. Uma execução monolítica é garantia de perder trabalho: o `llama-server` já
morreu duas vezes com `std::bad_alloc` no encoder de visão durante rodadas
longas. O registro é append-only por FRAME COMPLETO — nunca no meio de um frame —
então uma interrupção a qualquer momento custa no máximo o frame corrente.

**O ledger é a fonte da verdade.** `curate_progress.jsonl` é ao mesmo tempo o
checkpoint (de onde a retomada lê), o log (erros, tentativas, timestamps) e a
matéria-prima dos metadados. `curate_meta.json` é derivado dele e pode ser
reconstruído a qualquer momento — não guarda nada que o ledger não tenha.
"""

from __future__ import annotations

import json
import os
import platform
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge.config import (
    VLM_IMAGE_MAX_SIDE,
    VLM_MODEL,
    VLM_TEMPERATURE,
    VLM_URL,
)
from forge.core.classes import COCO_CLASSES, class_id
from forge.core.geometry import BoxN
from forge.vlm.client import VlmClient, VlmError, load_frame_image
from forge.vlm.phases import phase1_prompt_v2
from forge.vlm.prepare import (
    PHASE1_SCHEMA_V2,
    BoxTask,
    crop_around,
    draw_box,
    verdict_from_v2,
)

#: As 6 classes do gabarito. Só elas são curadas — é o vocabulário travado do
#: projeto. Caixa de classe fora daqui passa INTACTA para a saída: descartá-la
#: mudaria a distribuição de classes contra o rótulo original em silêncio, e a
#: decisão do projeto é indexar as 80 do COCO de ponta a ponta.
CURATION_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")

#: Falhas seguidas que derrubam a rodada. Uma falha isolada é transitória e vira
#: log; vinte seguidas significam servidor morto, e continuar só queimaria o
#: resto da fila registrando erro. Abortar deixa a retomada limpa.
MAX_CONSECUTIVE_FAILURES = 20

#: Tentativas por caixa antes de desistir dela.
MAX_ATTEMPTS = 3


@dataclass(slots=True)
class BoxResult:
    idx: int
    src_class: str
    box: list[float]
    verdict: str | None
    out_class: str | None
    survives: bool
    attempts: int = 1
    reason: str = ""
    error: str | None = None


@dataclass(slots=True)
class FrameResult:
    key: str
    acq_id: str
    sec: int
    started_at: str
    seconds: float
    curated: list[BoxResult] = field(default_factory=list)
    #: Caixas de classe fora das 6, copiadas sem perguntar nada ao modelo.
    passthrough: list[list[float]] = field(default_factory=list)
    passthrough_classes: list[str] = field(default_factory=list)

    def as_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


# ── leitura e escrita de rótulo ──────────────────────────────────────────────


def parse_label(path: Path) -> list[tuple[str, BoxN]]:
    """`.txt` YOLO → [(classe, caixa normalizada)]. Formato: cls cx cy w h."""
    out: list[tuple[str, BoxN]] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cid = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:5])
        out.append((
            COCO_CLASSES[cid],
            BoxN(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2),
        ))
    return out


def format_label(entries: list[tuple[str, BoxN]]) -> str:
    """[(classe, caixa)] → texto do `.txt`, no mesmo formato que entrou."""
    lines = []
    for cls, b in entries:
        cx, cy = (b.x1 + b.x2) / 2, (b.y1 + b.y2) / 2
        w, h = b.x2 - b.x1, b.y2 - b.y1
        lines.append(f"{class_id(cls)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return "\n".join(lines) + ("\n" if lines else "")


def frame_key_of(label_path: Path) -> tuple[str, int]:
    """`20240103203234__100.txt` → ("20240103203234", 100)."""
    acq, _, sec = label_path.stem.partition("__")
    return acq, int(sec)


# ── retomada ─────────────────────────────────────────────────────────────────


def completed_keys(ledger: Path) -> set[str]:
    """Frames já concluídos, lidos do ledger.

    Linha corrompida (escrita cortada por um kill no meio do flush) é ignorada
    em silêncio: o frame dela simplesmente será refeito, que é o comportamento
    seguro. Nunca aborta a retomada por causa de uma linha ruim.
    """
    if not ledger.is_file():
        return set()
    done: set[str] = set()
    for line in ledger.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            done.add(json.loads(line)["key"])
        except (json.JSONDecodeError, KeyError):
            continue
    return done


def append_ledger(ledger: Path, result: FrameResult) -> None:
    """Grava um frame COMPLETO e força para o disco.

    `flush` + `fsync` porque o ponto do ledger é sobreviver a queda de processo;
    buffer do SO guardando as últimas N linhas anularia a garantia.
    """
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(result.as_json() + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ── execução ─────────────────────────────────────────────────────────────────


def curate_frame(
    client: VlmClient,
    label_path: Path,
    image_path: Path,
    *,
    classes: tuple[str, ...] = CURATION_CLASSES,
) -> tuple[FrameResult, list[tuple[str, BoxN]]]:
    """Julga as caixas curáveis de um frame; devolve o resultado e o rótulo novo."""
    acq, sec = frame_key_of(label_path)
    started = datetime.now(UTC).isoformat()
    t0 = time.time()
    boxes = parse_label(label_path)
    img = load_frame_image(image_path)

    res = FrameResult(key=label_path.stem, acq_id=acq, sec=sec,
                      started_at=started, seconds=0.0)
    out: list[tuple[str, BoxN]] = []

    for i, (cls, box) in enumerate(boxes):
        if cls not in classes:
            res.passthrough.append([box.x1, box.y1, box.x2, box.y2])
            res.passthrough_classes.append(cls)
            out.append((cls, box))
            continue

        task = BoxTask(acq_id=acq, sec=sec, det_index=i, det_id=None,
                       proposed_class=cls, conf=None, box=box, truth="")
        br = BoxResult(idx=i, src_class=cls,
                       box=[box.x1, box.y1, box.x2, box.y2],
                       verdict=None, out_class=None, survives=True)

        for attempt in range(1, MAX_ATTEMPTS + 1):
            br.attempts = attempt
            try:
                ans = client.ask(
                    phase1_prompt_v2(task),
                    [draw_box(img, box, cls), crop_around(img, box)],
                    PHASE1_SCHEMA_V2,
                )
                verdict, new_cls = verdict_from_v2(ans, cls)
                br.verdict = verdict
                br.reason = str(ans.get("reason", ""))[:300]
                br.error = None
                break
            except VlmError as exc:
                br.error = str(exc)[:200]
                if attempt < MAX_ATTEMPTS:
                    time.sleep(2 * attempt)

        if br.verdict is None:
            # Falhou nas três tentativas: a caixa FICA como o YOLO a entregou.
            # Conservador de propósito — descartar por falha de infraestrutura
            # perderia dado por um motivo que nada tem a ver com a imagem.
            br.survives = True
            br.out_class = cls
            out.append((cls, box))
        elif br.verdict == "rejected":
            br.survives = False
        else:
            br.out_class = new_cls if br.verdict == "class-changed" and new_cls else cls
            out.append((br.out_class, box))
        res.curated.append(br)

    res.seconds = round(time.time() - t0, 2)
    return res, out


def iter_curation(
    client: VlmClient,
    label_dir: Path,
    out_dir: Path,
    image_of: Callable[[str, int], Path],
    *,
    ledger: Path,
    classes: tuple[str, ...] = CURATION_CLASSES,
    limit: int = 0,
) -> Iterator[FrameResult]:
    """Percorre os frames pendentes, gravando rótulo e ledger a cada um."""
    done = completed_keys(ledger)
    todo = sorted(p for p in label_dir.glob("*.txt") if p.stem not in done)
    if limit:
        todo = todo[:limit]
    out_dir.mkdir(parents=True, exist_ok=True)

    consecutive = 0
    for path in todo:
        acq, sec = frame_key_of(path)
        img_path = image_of(acq, sec)
        if not img_path.is_file():
            continue
        result, entries = curate_frame(client, path, img_path, classes=classes)
        # Rótulo primeiro, ledger depois: se cair entre os dois, o frame é
        # refeito e o rótulo reescrito. A ordem inversa deixaria um frame
        # marcado como pronto sem arquivo de rótulo.
        (out_dir / path.name).write_text(format_label(entries), encoding="utf-8")
        append_ledger(ledger, result)

        falhou = all(b.error for b in result.curated) and bool(result.curated)
        consecutive = consecutive + 1 if falhou else 0
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            raise VlmError(
                f"{consecutive} frames seguidos falharam por completo. "
                "O servidor provavelmente caiu (veja `std::bad_alloc` no log do "
                "Ollama). Reinicie e rode de novo — a retomada continua daqui."
            )
        yield result


# ── metadados ────────────────────────────────────────────────────────────────


def _ollama_runtime(url: str = VLM_URL) -> dict[str, Any]:
    """Config REAL do modelo carregado, perguntada ao servidor.

    `/api/ps` devolve o contexto e onde o modelo está (GPU/CPU) para cada modelo
    residente. Se o servidor não responder, registra isso explicitamente em vez
    de inventar um valor.
    """
    try:
        import httpx

        with httpx.Client(base_url=url.rstrip("/"), timeout=10.0) as c:
            version = c.get("/api/version").json().get("version")
            models = c.get("/api/ps").json().get("models", [])
    except Exception as exc:  # noqa: BLE001 — registro nunca derruba a rodada
        return {"erro": f"servidor não respondeu: {exc}"[:120]}
    loaded = [
        {
            "name": m.get("name"),
            "context_length": m.get("context_length"),
            "size_vram": m.get("size_vram"),
        }
        for m in models
    ]
    return {"version": version, "loaded": loaded,
            "num_parallel_env": os.environ.get("OLLAMA_NUM_PARALLEL", "(não visível deste processo)")}


def environment() -> dict[str, Any]:
    """Specs para a seção de método. Coletado por leitura, não digitado."""
    gpu: dict[str, Any] = {}
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15, check=False,
        ).stdout.strip()
        if out:
            name, mem, drv = (x.strip() for x in out.splitlines()[0].split(","))
            gpu = {"name": name, "memory_total": mem, "driver": drv}
    except Exception:  # noqa: BLE001 — specs são registro, não podem derrubar a rodada
        gpu = {"name": "desconhecida"}
    return {
        "gpu": gpu,
        "platform": platform.platform(),
        "model": VLM_MODEL,
        "schema": "v2 (object_present → object_class → box_quality → reason)",
        "temperature": VLM_TEMPERATURE,
        "image_max_side": VLM_IMAGE_MAX_SIDE,
        # Lido do SERVIDOR, não do ambiente deste processo. As variáveis
        # OLLAMA_* pertencem ao processo do `ollama serve`; um cliente que as
        # lesse do próprio ambiente registraria "(padrão)" mesmo com o servidor
        # rodando em 4096 — e iria errado para a seção de método.
        "ollama": _ollama_runtime(),
        "curation_classes": list(CURATION_CLASSES),
    }


def rollup(ledger: Path, total_frames: int) -> dict[str, Any]:
    """Metadados derivados do ledger. Reconstruível a qualquer momento."""
    by_verdict: dict[str, int] = {}
    by_class: dict[str, dict[str, int]] = {}
    errors: list[dict[str, Any]] = []
    frames = 0
    boxes = calls = attempts_total = passthrough = 0
    seconds = 0.0
    first = last = None

    for line in (ledger.read_text(encoding="utf-8").splitlines()
                 if ledger.is_file() else []):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        frames += 1
        seconds += r.get("seconds", 0.0)
        passthrough += len(r.get("passthrough", []))
        first = first or r.get("started_at")
        last = r.get("started_at")
        for b in r.get("curated", []):
            boxes += 1
            attempts_total += b.get("attempts", 1)
            calls += b.get("attempts", 1)
            v = b.get("verdict") or "(falhou)"
            by_verdict[v] = by_verdict.get(v, 0) + 1
            slot = by_class.setdefault(b["src_class"], {})
            slot[v] = slot.get(v, 0) + 1
            if b.get("error"):
                errors.append({"frame": r["key"], "box": b["idx"],
                               "attempts": b.get("attempts"), "error": b["error"]})

    return {
        "frames_processed": frames,
        "frames_total": total_frames,
        "frames_remaining": max(total_frames - frames, 0),
        "boxes_curated": boxes,
        "boxes_passthrough": passthrough,
        "model_calls": calls,
        "retries": attempts_total - boxes,
        "seconds": round(seconds, 1),
        "seconds_per_box": round(seconds / boxes, 2) if boxes else None,
        "seconds_per_frame": round(seconds / frames, 2) if frames else None,
        "first_frame_at": first,
        "last_frame_at": last,
        "by_verdict": dict(sorted(by_verdict.items())),
        "by_class": {k: dict(sorted(v.items())) for k, v in sorted(by_class.items())},
        "errors": errors[:200],
        "error_count": len(errors),
    }


def write_meta(meta_path: Path, ledger: Path, total_frames: int,
               extra: dict[str, Any] | None = None) -> dict[str, Any]:
    doc = {
        "generated_at": datetime.now(UTC).isoformat(),
        "environment": environment(),
        **rollup(ledger, total_frames),
        **(extra or {}),
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc


def eta(done: int, total: int, seconds_spent: float) -> str:
    """Estimativa a partir do ritmo REAL desta rodada, não de um número fixo."""
    if done == 0 or total <= done:
        return "—"
    rate = seconds_spent / done
    left = (total - done) * rate
    h, m = divmod(int(left // 60), 60)
    return f"{h}h{m:02d}m" if h else f"{m}m"
