"""Infraestrutura comum do bundle (NÃO é código científico).

O código científico — VlmClient, run_phase1(v2=True), prompt, schema, desenho da
caixa, recorte, JPEG — está em source/forge/, copiado byte a byte do carcara-forge
(ver provenance/source_provenance.json). Este arquivo só:
  - garante que `forge` é importado de source/ e de nenhum outro lugar;
  - converte cases/benchmark_cases.json de volta em BoxTask;
  - resolve o caminho dos frames DENTRO do bundle;
  - captura o corpo exato de cada requisição HTTP (impressão digital do payload);
  - calcula as estatísticas com o mesmo método do benchmark anterior.
"""
from __future__ import annotations

import hashlib
import json
import statistics as st
import sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent
SOURCE = BUNDLE / "source"
CASES = BUNDLE / "cases" / "benchmark_cases.json"
FINGERPRINTS = BUNDLE / "cases" / "request_fingerprints.json"
EXPECTED = BUNDLE / "expected" / "original_outputs_120.json"
RESULTS = BUNDLE / "results"

EXPECTED_OLLAMA_VERSION = "0.32.15"
MODEL_TAG = "qwen3-vl:8b-instruct"
MODEL_DIGEST = "0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319"
WEIGHTS_BLOB = "sha256-1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466"
WEIGHTS_BYTES = 6140392672
SEED = 20261001


def import_forge():
    """Importa o `forge` de source/ e confere que não veio de outro lugar."""
    if str(SOURCE) not in sys.path[:1]:
        sys.path.insert(0, str(SOURCE))
    import forge  # noqa: PLC0415

    origin = Path(forge.__file__).resolve()
    if SOURCE not in origin.parents:
        raise SystemExit(f"ERRO: `forge` importado de fora do bundle: {origin}")
    from forge.core.geometry import BoxN  # noqa: PLC0415
    from forge.vlm.client import Usage, VlmClient, VlmError  # noqa: PLC0415
    from forge.vlm.phases import run_phase1  # noqa: PLC0415
    from forge.vlm.prepare import BoxTask  # noqa: PLC0415

    for mod in ("forge.vlm.client", "forge.vlm.phases", "forge.vlm.prepare", "forge.config"):
        f = Path(sys.modules[mod].__file__).resolve()
        if SOURCE not in f.parents:
            raise SystemExit(f"ERRO: {mod} importado de fora do bundle: {f}")
    return BoxN, BoxTask, Usage, VlmClient, VlmError, run_phase1


def load_cases():
    BoxN, BoxTask, *_ = import_forge()
    doc = json.loads(CASES.read_text(encoding="utf-8"))
    cases = sorted(doc["cases"], key=lambda c: c["order"])
    out = []
    for c in cases:
        b = c["box_task"]
        t = BoxTask(acq_id=b["acq_id"], sec=b["sec"], det_index=b["det_index"],
                    det_id=b["det_id"], proposed_class=b["proposed_class"], conf=b["conf"],
                    box=BoxN(*b["box"]), truth=b["truth"])
        assert t.key == c["key"], c["key"]
        out.append((c, t))
    return doc["_meta"], out


def image_of(acq: str, sec: int) -> Path:
    """Mesma convenção de run_vlm_v2_969.py (ROOT/images/acq/sec.jpg), só que no bundle."""
    p = BUNDLE / "frames" / acq / f"{sec:06d}.jpg"
    if not p.is_file():
        raise FileNotFoundError(p)
    return p


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint_body(body: bytes) -> dict:
    """Impressão digital do corpo de POST /api/chat: inteiro e por parte."""
    j = json.loads(body)
    msg = j["messages"][0]
    return {
        "body_sha256": sha256_bytes(body),
        "body_bytes": len(body),
        "prompt_sha256": sha256_bytes(msg["content"].encode("utf-8")),
        "images_sha256": [sha256_bytes(im.encode("ascii")) for im in msg["images"]],
        "format_sha256": sha256_bytes(json.dumps(j["format"], sort_keys=True).encode()),
        "options": j["options"],
        "model": j["model"],
        "stream": j["stream"],
    }


class RequestRecorder:
    """Hook de request do httpx: guarda o corpo exato de cada POST /api/chat."""

    def __init__(self) -> None:
        self.bodies: list[bytes] = []

    def __call__(self, request) -> None:
        if request.url.path.endswith("/api/chat"):
            self.bodies.append(bytes(request.read()))


def offline_fingerprints() -> dict[str, dict]:
    """Monta o corpo HTTP de cada caso SEM Ollama, pelo caminho real do código.

    Usa o VlmClient e o run_phase1(v2=True) de source/ sem alteração; só troca o
    transporte do httpx por um que devolve uma resposta fixa válida no schema. O
    corpo capturado é exatamente o que seria enviado ao Ollama. Uma chamada
    run_phase1 por caso, para alinhar cada corpo à sua chave.
    """
    import httpx  # noqa: PLC0415

    _, _, _, VlmClient, _, run_phase1 = import_forge()
    _, cases = load_cases()
    fake = json.dumps({"object_present": True, "object_class": "car",
                       "box_quality": "tight", "reason": "offline"})

    def handler(request):
        return httpx.Response(200, json={"message": {"role": "assistant", "content": fake},
                                         "prompt_eval_count": 0, "eval_count": 0})

    out: dict[str, dict] = {}
    with VlmClient() as client:
        rec = RequestRecorder()
        real = client._client
        client._client = httpx.Client(base_url=real.base_url, timeout=real.timeout,
                                      transport=httpx.MockTransport(handler),
                                      event_hooks={"request": [rec], "response": []})
        real.close()
        for c, t in cases:
            n0 = len(rec.bodies)
            rows = run_phase1(client, [t], image_of, v2=True)
            assert len(rec.bodies) == n0 + 1 and rows[0].error is None, c["key"]
            out[c["key"]] = fingerprint_body(rec.bodies[-1])
    return out


def summ(v: list[float]) -> dict:
    """Mesmo cálculo do benchmark anterior: statistics.quantiles(n=20) (método
    'exclusive', padrão do Python) → p5 = q[0], p95 = q[-1]; DP amostral (n-1)."""
    q = st.quantiles(v, n=20)
    return {"n": len(v), "mean": st.mean(v), "median": st.median(v), "sd": st.stdev(v),
            "min": min(v), "p5": q[0], "p95": q[-1], "max": max(v), "sum": sum(v)}
