"""Registro de execução.

Todo número que pode ir para o paper nasce dentro de um run, e todo run grava o
que precisaria para ser refeito: hash dos pesos, parâmetros, versões, commit.
Sem isso "mAP 0.48" é folclore.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forge import __version__


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _versions() -> dict[str, str]:
    vers = {"python": sys.version.split()[0], "forge": __version__}
    for mod in ("torch", "torchvision", "ultralytics", "pycocotools", "numpy"):
        try:
            vers[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            vers[mod] = "ausente"
    return vers


@dataclass(slots=True)
class Run:
    run_id: str
    command: str
    root: Path
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    params: dict[str, Any] = field(default_factory=dict)
    results: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        runs_root: Path,
        command: str,
        *,
        params: dict[str, Any] | None = None,
        weights: Path | None = None,
    ) -> Run:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        run_id = f"{stamp}-{command}"
        root = Path(runs_root) / run_id
        root.mkdir(parents=True, exist_ok=True)

        p = dict(params or {})
        if weights is not None and Path(weights).is_file():
            p["weights_path"] = str(Path(weights).resolve())
            p["weights_sha256"] = sha256_file(Path(weights))
            p["weights_bytes"] = Path(weights).stat().st_size

        run = cls(
            run_id=run_id,
            command=command,
            root=root,
            params=p,
            environment={
                "platform": platform.platform(),
                "machine": platform.machine(),
                "git_commit": _git_commit(),
                "versions": _versions(),
                **_gpu_info(),
            },
        )
        # Grava o meta.json já na criação. Um treino que morre no meio deixaria
        # um diretório vazio sem explicação; assim ele deixa registro do que foi
        # tentado — `results` vazio é justamente o que diz que não terminou.
        run.save()
        return run

    def path(self, name: str) -> Path:
        return self.root / name

    def save(self) -> Path:
        doc = asdict(self)
        doc["root"] = str(self.root)
        meta = self.root / "meta.json"
        meta.write_text(json.dumps(doc, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return meta


def _gpu_info() -> dict[str, Any]:
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {"cuda": None}
    if not torch.cuda.is_available():
        return {"cuda": {"available": False}}
    return {
        "cuda": {
            "available": True,
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch_arch_list": torch.cuda.get_arch_list(),
        }
    }
