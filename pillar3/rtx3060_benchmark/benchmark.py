"""Benchmark P3 (VLM v2) — 120 propostas + 3 warm-up — sobre o conjunto congelado de 969.

Reproduz o benchmark de 2026-10-01 (bench_vlm_969.py, em provenance/original_scripts/)
com os MESMOS casos, a MESMA ordem e o MESMO método de cronometragem:
  - warm-up: uma chamada run_phase1(v2=True) com os 3 casos de warm-up;
  - medidos: um run_phase1(v2=True) por frame, na ordem registrada;
  - ponta a ponta = intervalo entre callbacks on_step (desenho + recorte + JPEG +
    base64 + HTTP; a 1a caixa de cada frame inclui abrir o JPEG);
  - HTTP = o cronômetro do próprio VlmClient.ask (time.time() em volta do POST).

Pré-requisito: Ollama 0.32.15 iniciado por start_ollama.ps1 (log em results/).

Uso:
    .venv\\Scripts\\python.exe benchmark.py
"""
from __future__ import annotations

import csv
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import p3common as P

OUT_JSON = P.RESULTS / "benchmark_rtx3060.json"
OUT_CSV = P.RESULTS / "benchmark_rtx3060.csv"
ENV_TXT = P.RESULTS / "environment.txt"
CONSOLE = P.RESULTS / "console.log"
PROV = P.RESULTS / "provenance.json"
SERVE_LOG = P.RESULTS / "ollama-serve.log"


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def parse_serve_log() -> dict:
    txt = SERVE_LOG.read_text(encoding="utf-8", errors="replace") if SERVE_LOG.is_file() else ""
    g = lambda pat: (m.group(0) if (m := re.search(pat, txt)) else None)  # noqa: E731
    starts = re.findall(r'msg="starting llama-server" cmd="([^"]*)"', txt)
    return {
        "log_present": bool(txt),
        "server_config": g(r'msg="server config" env="[^"]*"'),
        "inference_compute": re.findall(r'msg="inference compute"[^\n]*', txt),
        "llama_server_cmd": starts,
        "offload": g(r"offloaded \d+/\d+ layers to GPU"),
        "n_ctx": g(r"llama_context: n_ctx\s+=\s+\d+"),
        "n_slots": g(r"n_slots = \d+"),
        "prompt_cache_disabled": "prompt cache is disabled" in txt,
        "flash_attn_setting": g(r"llama_context: flash_attn\s+=\s+\S+"),
        "flash_attention_enabled": ("Flash Attention enabled" in txt
                                    or "flash attention is enabled" in txt),
        "flash_attention_lines": re.findall(r"[^\n]*[Ff]lash[ _][Aa]ttention[^\n]*", txt)[:10],
        "kv_cache": g(r"llama_kv_cache: size = [^\n]*"),
        "mmap": g(r'msg="disabling mmap[^\n]*'),
    }


def check_serve_log(s: dict) -> list[str]:
    errs = []
    if not s["log_present"]:
        errs.append(f"log do servidor ausente em {SERVE_LOG} — inicie o Ollama por start_ollama.ps1")
        return errs
    if not s["llama_server_cmd"]:
        errs.append("llama-server não aparece no log (modelo não carregado por este servidor?)")
    if not any("library=CUDA" in ln for ln in s["inference_compute"]):
        errs.append(f"backend de inferência não é CUDA: {s['inference_compute']}")
    if s["offload"] != "offloaded 37/37 layers to GPU":
        errs.append(f"offload incompleto: {s['offload']}")
    if not (s["n_ctx"] and s["n_ctx"].endswith(" 4096")):
        errs.append(f"contexto efetivo != 4096: {s['n_ctx']}")
    if s["n_slots"] != "n_slots = 1":
        errs.append(f"slots != 1: {s['n_slots']}")
    if not s["prompt_cache_disabled"]:
        errs.append("cache de prompt NÃO está desligado (LLAMA_ARG_CACHE_RAM=0 ausente)")
    return errs


def nvsmi() -> str:
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    except OSError as exc:
        return f"nvidia-smi falhou: {exc}"


def main() -> int:
    P.RESULTS.mkdir(exist_ok=True)
    if OUT_JSON.exists():
        print(f"ERRO: {OUT_JSON} já existe. Mova os resultados anteriores para outra pasta; "
              "este script não sobrescreve resultados.")
        return 3
    log_fh = CONSOLE.open("a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_fh)
    sys.stderr = Tee(sys.__stderr__, log_fh)
    started = datetime.now(timezone.utc).isoformat()
    print(f"=== benchmark P3 RTX 3060 — início {started}")

    # ── 1. verificação completa (bloqueante) ───────────────────────────────────
    import verify_environment
    code, venv = verify_environment.run([])
    if code != 0:
        print("ABORTADO: verify_environment falhou. Nada foi medido.")
        return 1

    import httpx
    _, _, Usage, VlmClient, _, run_phase1 = P.import_forge()

    class RecUsage(Usage):  # igual ao bench_vlm_969.py
        def __init__(self) -> None:
            super().__init__()
            self.raw: list[dict] = []

        def add(self, resp, elapsed):  # type: ignore[override]
            super().add(resp, elapsed)
            self.raw.append({"http_s": elapsed, **{k: resp.get(k) for k in (
                "total_duration", "load_duration", "prompt_eval_count", "prompt_eval_duration",
                "eval_count", "eval_duration", "done_reason")}})

    meta, cases = P.load_cases()
    warm = [(c, t) for c, t in cases if c["role"] == "warmup"]
    meas = [(c, t) for c, t in cases if c["role"] == "measured"]
    fps = json.loads(P.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"]
    exp = json.loads(P.EXPECTED.read_text(encoding="utf-8"))
    exp_by = {e["key"]: e for e in exp["warmup"] + exp["measured"]}

    by_frame: dict = {}
    for c, t in meas:
        by_frame.setdefault((t.acq_id, t.sec), []).append((c, t))

    smi_before = nvsmi()
    req = P.RequestRecorder()
    responses: list[str] = []

    def on_response(r):
        if r.request.url.path.endswith("/api/chat"):
            r.read()
            try:
                responses.append(r.json().get("message", {}).get("content"))
            except ValueError:
                responses.append(None)

    with VlmClient() as client:
        client.usage = RecUsage()
        client._client.event_hooks = {"request": [req], "response": [on_response]}
        version = client.health()
        ps0 = client._client.get("/api/ps").json()

        # ── 2. warm-up (3 casos reais, fora das estatísticas) ──────────────────
        print("warm-up: 3 chamadas...")
        tw = time.perf_counter()
        warm_rows = run_phase1(client, [t for _, t in warm], P.image_of, v2=True)
        warm_s = time.perf_counter() - tw
        warm_raw = list(client.usage.raw)
        warm_bodies, warm_resp = list(req.bodies), list(responses)
        client.usage.raw.clear()
        req.bodies.clear()
        responses.clear()
        ps = client._client.get("/api/ps").json()
        print(f"warm-up: {warm_s:.1f} s; http {[round(x['http_s'], 2) for x in warm_raw]}")

        # ── 3. configuração efetiva do servidor (bloqueante) ───────────────────
        serve = parse_serve_log()
        errs = check_serve_log(serve)
        loaded = [m for m in ps.get("models", []) if m.get("name") == P.MODEL_TAG]
        if not loaded or loaded[0].get("size_vram") != loaded[0].get("size"):
            errs.append(f"modelo não está 100% na VRAM segundo /api/ps: {ps}")
        elif loaded[0].get("context_length") != 4096:
            errs.append(f"context_length != 4096 segundo /api/ps: {loaded[0].get('context_length')}")
        for e in errs:
            print(f"ERRO de runtime: {e}")
        if errs:
            print("ABORTADO antes das 120 medições. Ver results/ollama-serve.log.")
            (P.RESULTS / "aborted_runtime_check.json").write_text(
                json.dumps({"errors": errs, "serve": serve, "ps": ps}, indent=1), encoding="utf-8")
            return 2
        print(f"runtime: {serve['offload']}; {serve['n_ctx']}; {serve['n_slots']}; "
              f"prompt cache off; flash attention "
              f"{'ATIVO' if serve['flash_attention_enabled'] else 'inativo'}")

        # ── 4. 120 medidas ─────────────────────────────────────────────────────
        per_prop: list[float] = []
        rows = []
        t_all = time.perf_counter()
        for (acq, sec), group in by_frame.items():
            last = [time.perf_counter()]

            def step(_d, _t, last=last):
                now = time.perf_counter()
                per_prop.append(now - last[0])
                last[0] = now

            rows += run_phase1(client, [t for _, t in group], P.image_of, v2=True, on_step=step)
            n = len(rows)
            if n % 10 < len(group):
                el = time.perf_counter() - t_all
                print(f"  {n}/120  {el:.0f} s  {el / n:.2f} s/proposta", flush=True)
        wall = time.perf_counter() - t_all
        raw = client.usage.raw
        bodies, resps = list(req.bodies), list(responses)
        ps_end = client._client.get("/api/ps").json()

    smi_after = nvsmi()
    assert len(rows) == len(meas) == len(raw) == len(bodies) == len(per_prop)

    # ── 5. comparação com o run original e com o payload original ─────────────
    recs = []
    for (c, t), r, p, x, body, content in zip(meas, rows, per_prop, raw, bodies, resps):
        assert r.key == c["key"]
        o = exp_by[c["key"]]["row"]
        fp = P.fingerprint_body(body)
        recs.append({
            "order": c["order"], "key": c["key"], "uid": c["uid"], "frame": c["frame"],
            "proposed_class": t.proposed_class, "conf": t.conf,
            "e2e_s": p, **x,
            "raw_model_json": content,
            "vlm": r.vlm, "vlm_class": r.vlm_class, "reason": r.reason, "error": r.error,
            "orig_vlm": o["vlm"], "orig_vlm_class": o["vlm_class"],
            "orig_session": exp_by[c["key"]]["session"],
            "match_verdict": (r.vlm, r.vlm_class) == (o["vlm"], o["vlm_class"]),
            "match_strict": (r.vlm, r.vlm_class, r.reason) == (o["vlm"], o["vlm_class"], o["reason"]),
            "payload_body_sha256": fp["body_sha256"],
            "payload_match": fp == fps[c["key"]],
        })
    warm_recs = []
    for (c, t), r, x, body, content in zip(warm, warm_rows, warm_raw, warm_bodies, warm_resp):
        o = exp_by[c["key"]]["row"]
        warm_recs.append({"key": c["key"], "uid": c["uid"], **x, "raw_model_json": content,
                          "vlm": r.vlm, "vlm_class": r.vlm_class, "error": r.error,
                          "match_verdict": (r.vlm, r.vlm_class) == (o["vlm"], o["vlm_class"]),
                          "payload_match": P.fingerprint_body(body) == fps[c["key"]]})

    eq = sum(r["match_verdict"] for r in recs)
    eq_strict = sum(r["match_strict"] for r in recs)
    pay = sum(r["payload_match"] for r in recs)
    gpu = venv["gpu"]["gpus"][0] if venv["gpu"].get("gpus") else {}
    s = lambda k, f=1.0: P.summ([r[k] / f for r in recs])  # noqa: E731
    result = {
        "experiment": "Pillar 3 atual — VLM decomposed v2 sobre 969 propostas YOLO11x cru",
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "gpu": gpu.get("name"), "vram_mib": gpu.get("memory_total_mib"),
        "driver": gpu.get("driver"), "cuda_driver_api": venv["gpu"].get("cuda_driver_api"),
        "ollama_version": version, "model": P.MODEL_TAG, "model_digest":
            venv.get("ollama_server", {}).get("tag", {}).get("digest"),
        "quantization": venv.get("ollama_server", {}).get("show_details", {}).get("quantization_level"),
        "n": len(recs), "warmups": len(warm_recs), "seed": meta["seed"],
        "frames_measured": len(by_frame),
        "outputs_equal_verdict": f"{eq}/{len(recs)}",
        "outputs_equal_strict_incl_reason": f"{eq_strict}/{len(recs)}",
        "payload_identical_to_original_pc": f"{pay}/{len(recs)}",
        "errors": sum(1 for r in recs if r["error"]),
        "end_to_end_per_proposal_s": s("e2e_s"),
        "http_per_call_s": s("http_s"),
        "ollama_total_duration_s": s("total_duration", 1e9),
        "prompt_eval_s": s("prompt_eval_duration", 1e9),
        "eval_s": s("eval_duration", 1e9),
        "prompt_tokens": s("prompt_eval_count"),
        "eval_tokens": s("eval_count"),
        "total_wall_clock_s": wall,
        "wall_per_proposal_s": wall / len(recs),
        "warmup": {"n": len(warm_recs), "seconds": warm_s, "calls": warm_recs},
        "runtime_effective": serve,
        "ollama_ps_before": ps0, "ollama_ps_after_warmup": ps, "ollama_ps_end": ps_end,
        "reference_previous_benchmark": "reference/bench_vlm_969_result_RTX5060Ti_2026-10-01.json",
        "per_proposal": recs,
    }
    OUT_JSON.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    cols = ["order", "key", "uid", "frame", "proposed_class", "conf", "e2e_s", "http_s",
            "total_duration", "load_duration", "prompt_eval_duration", "eval_duration",
            "prompt_eval_count", "eval_count", "done_reason", "vlm", "vlm_class", "orig_vlm",
            "orig_vlm_class", "match_verdict", "match_strict", "payload_match", "orig_session",
            "error", "reason", "raw_model_json"]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)

    ENV_TXT.write_text(
        f"started_at: {started}\nplatform: {platform.platform()}\npython: {sys.version}\n"
        f"executable: {sys.executable}\nollama server: {version}\n"
        f"ollama binary: {json.dumps(venv.get('ollama_binary'), ensure_ascii=False)}\n"
        f"model: {json.dumps(venv.get('ollama_server', {}).get('tag'))}\n"
        f"model params: {venv.get('ollama_server', {}).get('show_parameters')}\n"
        f"packages: {json.dumps(venv.get('packages'))}\n"
        f"pillow: {json.dumps(venv.get('pillow_features'))}\n"
        f"\n===== runtime efetivo (log do servidor) =====\n{json.dumps(serve, indent=1)}\n"
        f"\n===== nvidia-smi antes =====\n{smi_before}\n===== nvidia-smi depois =====\n{smi_after}\n",
        encoding="utf-8")
    PROV.write_text(json.dumps({
        "bundle_sha256sums_sha256": P.sha256_file(P.BUNDLE / "SHA256SUMS.txt"),
        "cases_sha256": P.sha256_file(P.CASES),
        "expected_sha256": P.sha256_file(P.EXPECTED),
        "fingerprints_sha256": P.sha256_file(P.FINGERPRINTS),
        "verify_environment": venv,
        "runtime_effective": serve,
        "nvidia_smi_before": smi_before, "nvidia_smi_after": smi_after,
        "argv": sys.argv, "started_at": started,
    }, indent=1, ensure_ascii=False), encoding="utf-8")

    e2e, http = result["end_to_end_per_proposal_s"], result["http_per_call_s"]
    print(f"\n=== RESULTADO ({result['gpu']}, Ollama {version})")
    print(f"vereditos iguais ao run original: {result['outputs_equal_verdict']}"
          f"  (com reason idêntico: {result['outputs_equal_strict_incl_reason']})")
    print(f"payload idêntico ao do PC original: {result['payload_identical_to_original_pc']}")
    for name, d in (("ponta a ponta", e2e), ("HTTP", http)):
        print(f"{name:14}: média {d['mean']:.3f}  mediana {d['median']:.3f}  DP {d['sd']:.3f}  "
              f"min {d['min']:.3f}  p5 {d['p5']:.3f}  p95 {d['p95']:.3f}  max {d['max']:.3f}  (n={d['n']})")
    print(f"tempo total ({len(recs)} propostas): {wall:.1f} s")
    print(f"resultados em {P.RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
