"""Verifica o ambiente ANTES do benchmark. Sai com código != 0 se algo essencial falhar.

Uso:
    .venv\\Scripts\\python.exe verify_environment.py              # completo (Ollama no ar)
    .venv\\Scripts\\python.exe verify_environment.py --offline    # sem servidor (passos 1-6)

Classes de resultado:
    FAIL  — científico/essencial. O benchmark NÃO pode começar.
    WARN  — registrado, não bloqueia (ex.: patch do Python diferente).
    OK
"""
from __future__ import annotations

import importlib.metadata as md
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

import p3common as P

REQ = {
    "pillow": "12.3.0", "httpx": "0.28.1", "httpcore": "1.0.9", "h11": "0.16.0",
    "anyio": "4.14.2", "idna": "3.18", "certifi": "2026.7.22", "typing-extensions": "4.16.0",
    "pydantic": "2.13.4", "pydantic-core": "2.46.4", "annotated-types": "0.8.0",
    "typing-inspection": "0.4.4",
}


class Report:
    def __init__(self) -> None:
        self.items: list[dict] = []

    def add(self, status: str, check: str, detail: object = "") -> None:
        self.items.append({"status": status, "check": check, "detail": detail})
        print(f"[{status:4}] {check}: {detail}", flush=True)

    @property
    def failed(self) -> list[dict]:
        return [i for i in self.items if i["status"] == "FAIL"]


def nvidia(rep: Report) -> dict:
    info: dict = {}
    exe = shutil.which("nvidia-smi")
    if not exe:
        rep.add("FAIL", "nvidia-smi", "não encontrado no PATH")
        return info
    q = subprocess.run([exe, "--query-gpu=name,memory.total,driver_version,pci.bus_id",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    full = subprocess.run([exe], capture_output=True, text=True).stdout
    info["nvidia_smi"] = full
    gpus = [dict(zip(("name", "memory_total_mib", "driver", "pci"), [x.strip() for x in ln.split(",")]))
            for ln in q.stdout.strip().splitlines() if ln.strip()]
    info["gpus"] = gpus
    # drivers antigos: "CUDA Version: 12.x"; drivers 6xx: "CUDA UMD Version: 13.x"
    m = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", full)
    info["cuda_driver_api"] = m.group(1) if m else None
    rep.add("OK" if info["cuda_driver_api"] else "FAIL", "CUDA (driver API, nvidia-smi)",
            info["cuda_driver_api"])
    if len(gpus) != 1:
        rep.add("WARN", "número de GPUs", f"{len(gpus)} — o Ollama pode escolher outra; "
                "fixe com CUDA_VISIBLE_DEVICES se preciso")
    ok = [g for g in gpus if re.search(r"RTX 3060(?!\s*Ti)", g["name"])
          and 11000 <= float(g["memory_total_mib"]) <= 13000]
    rep.add("OK" if ok else "FAIL", "GPU RTX 3060 12 GB",
            [f"{g['name']} {g['memory_total_mib']} MiB driver {g['driver']}" for g in gpus])
    return info


def ollama_binary(rep: Report) -> dict:
    exe = os.environ.get("OLLAMA_EXE") or shutil.which("ollama")
    if not exe:
        rep.add("FAIL", "binário ollama", "não encontrado (PATH ou OLLAMA_EXE)")
        return {}
    r = subprocess.run([exe, "--version"], capture_output=True, text=True)
    txt = (r.stdout + r.stderr).strip()
    vers = re.findall(r"version is (\S+)", txt)
    ok = vers and all(v == P.EXPECTED_OLLAMA_VERSION for v in vers)
    rep.add("OK" if ok else "FAIL", f"ollama --version ({exe})", txt.replace("\n", " | "))
    return {"exe": exe, "sha256": P.sha256_file(Path(exe)), "version_output": txt}


def model_on_disk(rep: Report) -> dict:
    root = Path(os.environ.get("OLLAMA_MODELS") or Path.home() / ".ollama" / "models")
    man = root / "manifests/registry.ollama.ai/library/qwen3-vl/8b-instruct"
    out = {"models_dir": str(root)}
    if not man.is_file():
        rep.add("FAIL", "manifesto do modelo em disco", f"ausente: {man}")
        return out
    d = P.sha256_file(man)
    out["manifest_sha256"] = d
    rep.add("OK" if d == P.MODEL_DIGEST else "FAIL", "digest da tag (sha256 do manifesto)", d)
    blob = root / "blobs" / P.WEIGHTS_BLOB
    if not blob.is_file():
        rep.add("FAIL", "blob de pesos", f"ausente: {blob}")
        return out
    size = blob.stat().st_size
    print("       calculando sha256 do blob de pesos (6,1 GB)...", flush=True)
    h = P.sha256_file(blob)
    out["weights_sha256"] = h
    rep.add("OK" if (size == P.WEIGHTS_BYTES and "sha256-" + h == P.WEIGHTS_BLOB) else "FAIL",
            "blob de pesos Q4_K_M (tamanho + sha256)", f"{size} bytes, sha256 {h}")
    return out


def server(rep: Report) -> dict:
    import httpx

    out: dict = {}
    try:
        with httpx.Client(base_url="http://localhost:11434", timeout=30) as c:
            v = c.get("/api/version").json().get("version")
            tags = c.get("/api/tags").json().get("models", [])
            show = c.post("/api/show", json={"model": P.MODEL_TAG}).json()
    except httpx.HTTPError as exc:
        rep.add("FAIL", "servidor Ollama em localhost:11434", f"não responde: {exc}")
        return out
    out["server_version"] = v
    rep.add("OK" if v == P.EXPECTED_OLLAMA_VERSION else "FAIL", "versão do servidor (/api/version)", v)
    m = next((t for t in tags if t.get("name") == P.MODEL_TAG), None)
    out["tag"] = m
    rep.add("OK" if m and m.get("digest") == P.MODEL_DIGEST else "FAIL",
            "modelo/digest servido (/api/tags)", m and m.get("digest"))
    det = show.get("details", {})
    out["show_details"] = det
    out["show_parameters"] = show.get("parameters")
    rep.add("OK" if det.get("quantization_level") == "Q4_K_M" else "FAIL",
            "quantização (/api/show)", det.get("quantization_level"))
    params = show.get("parameters") or ""
    ok = (re.search(r"top_k\s+20\b", params) and re.search(r"top_p\s+0\.95\b", params))
    rep.add("OK" if ok else "FAIL", "Modelfile top_k 20 / top_p 0.95 (/api/show)",
            params.replace("\n", " | "))
    return out


def main(argv: list[str]) -> int:
    offline = "--offline" in argv
    rep = Report()
    env: dict = {"platform": platform.platform(), "python": sys.version,
                 "executable": sys.executable}

    # 1. Python
    vi = sys.version_info
    rep.add("OK" if (vi.major, vi.minor) == (3, 11) else "FAIL", "Python 3.11", sys.version.split()[0])
    if (vi.major, vi.minor, vi.micro) != (3, 11, 9):
        rep.add("WARN", "Python 3.11.9 (o do run)", sys.version.split()[0])
    rep.add("OK" if platform.architecture()[0] == "64bit" else "FAIL", "Python 64-bit",
            platform.architecture()[0])
    if P.BUNDLE not in Path(sys.executable).resolve().parents:
        rep.add("WARN", "interpretador fora do bundle", sys.executable)

    # 2. dependências exatas
    vers = {}
    for name, want in REQ.items():
        try:
            got = md.version(name)
        except md.PackageNotFoundError:
            got = None
        vers[name] = got
        rep.add("OK" if got == want else "FAIL", f"pacote {name}=={want}", got)
    env["packages"] = vers
    from PIL import features
    env["pillow_features"] = {"libjpeg_turbo": features.version("libjpeg_turbo"),
                              "jpg": features.version("jpg"),
                              "freetype2": features.version("freetype2")}
    rep.add("OK", "Pillow libs", env["pillow_features"])

    # 3. código científico vem do bundle
    P.import_forge()
    import forge
    rep.add("OK", "forge importado do bundle", forge.__file__)

    # 4. SHA256SUMS
    sums = P.BUNDLE / "SHA256SUMS.txt"
    bad, n = [], 0
    for ln in sums.read_text(encoding="utf-8").splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        h, rel = ln.split(None, 1)
        rel = rel.lstrip("*").strip()
        n += 1
        p = P.BUNDLE / rel
        if not p.is_file() or P.sha256_file(p) != h:
            bad.append(rel)
    rep.add("OK" if not bad else "FAIL", f"SHA256SUMS.txt ({n} arquivos)", bad or "todos conferem")

    # 5. casos e frames
    meta, cases = P.load_cases()
    nm = sum(1 for c, _ in cases if c["role"] == "measured")
    nw = sum(1 for c, _ in cases if c["role"] == "warmup")
    rep.add("OK" if (nm, nw, meta["seed"]) == (120, 3, P.SEED) else "FAIL",
            "casos (120 medidos + 3 warm-up, seed 20261001)", f"{nm} + {nw}, seed {meta['seed']}")
    fbad = [c["frame"] for c, _ in cases
            if P.sha256_file(P.BUNDLE / c["frame"]) != c["frame_sha256"]]
    rep.add("OK" if not fbad else "FAIL", "sha256 dos frames dos casos", fbad or "todos conferem")
    exp = json.loads(P.EXPECTED.read_text(encoding="utf-8"))
    ek = [e["key"] for e in exp["measured"]]
    rep.add("OK" if ek == [c["key"] for c, _ in cases if c["role"] == "measured"] else "FAIL",
            "expected/ alinhado aos 120 casos", len(ek))

    # 6. payload exato (sem Ollama): mesmas imagens, prompt, schema e opções do PC original
    want = json.loads(P.FINGERPRINTS.read_text(encoding="utf-8"))["fingerprints"]
    got = P.offline_fingerprints()
    diff = [k for k in want if want[k] != got.get(k)]
    rep.add("OK" if not diff else "FAIL", "payload HTTP idêntico ao do PC original (123 casos)",
            "123/123 idênticos" if not diff else f"{len(diff)} divergentes: {diff[:5]}")
    if diff:
        k = diff[0]
        env["first_payload_diff"] = {"key": k, "want": want[k], "got": got.get(k)}

    # 7-9. GPU, Ollama, modelo
    env["gpu"] = nvidia(rep)
    env["ollama_binary"] = ollama_binary(rep)
    env["model_on_disk"] = model_on_disk(rep)
    if not offline:
        env["ollama_server"] = server(rep)
    else:
        rep.add("WARN", "servidor Ollama", "não verificado (--offline)")

    env["checks"] = rep.items
    P.RESULTS.mkdir(exist_ok=True)
    (P.RESULTS / "verify_environment.json").write_text(
        json.dumps(env, indent=1, ensure_ascii=False), encoding="utf-8")
    if rep.failed:
        print(f"\nFALHOU: {len(rep.failed)} verificação(ões) essencial(is). NÃO rode o benchmark.")
        for i in rep.failed:
            print(f"  - {i['check']}: {i['detail']}")
        return 1
    print("\nAmbiente OK" + (" (offline; servidor não verificado)" if offline else ""))
    return 0


def run(argv: list[str] | None = None) -> tuple[int, dict]:
    code = main(argv or [])
    env = json.loads((P.RESULTS / "verify_environment.json").read_text(encoding="utf-8"))
    return code, env


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
