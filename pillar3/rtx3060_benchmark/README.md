# Pillar 3 runtime benchmark (RTX 3060 12 GB replay)

This folder replays **120 of the 969 frozen Pillar 3 proposals** (plus 3 warm-up calls) through the same
VLM pipeline — `qwen3-vl:8b-instruct` (Q4_K_M) served by Ollama 0.32.15 — to measure the **time per
proposal** and to compare each answer with the answer recorded in the original run.

It is a **runtime characterization, not a new experiment**. The scientific results of Pillar 3 come
from the original 969-proposal run; this replay only documents timing and output agreement on
documented hardware.

## What this benchmark does *not* claim

- The metadata of the original 969-proposal run (`runtime/experiment_meta.json`) **does not record the
  GPU**, and no artifact proves which GPU produced it. Earlier runs of the project recorded an
  RTX 5060 Ti. **The RTX 3060 is the GPU of this later benchmark only**; do not read it as the hardware
  of the original run.
- The `meta.json` of the original run covers only the last 470 calls. Do not extrapolate timing from
  this replay to the full run, and do not assume bit-identical output across GPUs
  (temperature 0 does not guarantee it).

## Result (RTX 3060 run of 2026-10-02)

Mean 7.43 s per proposal end-to-end (7.41 s HTTP only), n = 120, three excluded warm-ups, seed
`20261001`. Verdict + class equal to the original run for **118/120** proposals; equal including the
free-text `reason` for 38/120. The full result files are archived on Zenodo (record pending) in
`06_experiment_results/pillar3_runtime_rtx3060/` and are re-checked by the Zenodo package scripts. The
GPU, driver (610.88) and CUDA driver API (13.3) of that run are in its `environment.txt`.

## What is in this folder

| Path | Content |
|---|---|
| `benchmark.py` | verifies the environment, runs 3 warm-ups, times 120 proposals, writes `results/` |
| `verify_environment.py` | blocking checks: GPU, Ollama version, model digest, Python, pinned packages, file hashes, request payloads |
| `start_ollama.ps1` | starts `ollama serve` with the configuration of the final session (`LLAMA_ARG_CACHE_RAM=0`) |
| `p3common.py` | bundle plumbing (imports, cases, paths, payload fingerprint, statistics) — not scientific code |
| `prepare_replay.py` | creates a writable working copy and `SHA256SUMS.txt`, pointing at your own frames directory |
| `requirements.txt` | pinned Python packages (see below) |
| `cases/` | the 123 cases (3 warm-up + 120 timed): exact box task, frozen original proposal, frame hash, and the SHA-256 of the exact HTTP body of each call |
| `expected/` | the original-run rows for the same 120 proposals |
| `runtime/` | run metadata, JSON schema, rendered prompt, model info, Ollama server configuration with the evidence level of each item, upstream checksum list |
| `source/forge/` | byte-for-byte copy of the code that ran the original run (hashes in `provenance/source_provenance.json`) |
| `model_transfer/` | Ollama manifest and `import_model.ps1` for the model blobs |

## What is NOT distributed

- **The original JPEG frames.** Redistribution rights, consent and face/license-plate anonymization
  are not established, so no image is published (not on GitHub, not on Zenodo). The 103 frames used by
  this benchmark are a subset of the 313 evaluation frames; their SHA-256 hashes are in `cases/`. You
  can only run the benchmark if you have **separately authorized** copies of those exact files.
- The **model weights** (6.1 GB blob). Obtain `qwen3-vl:8b-instruct` through Ollama or provide the blob
  to `model_transfer/import_model.ps1`. The tag digest and the weights hash are checked, and any other
  digest is refused.
- The **Ollama binaries**. Use the official `ollama-windows-amd64.zip` of release v0.32.15 (SHA-256 in
  `runtime/ollama_server_config.json`), not the installer that auto-updates.
- Results files from the benchmark run (they live in the Zenodo package).

## Pinned environment (required)

| Component | Required |
|---|---|
| Python | 3.11 (64-bit); the run used 3.11.9 |
| Ollama | **0.32.15** exactly (another version is refused) |
| Model | `qwen3-vl:8b-instruct`, Q4_K_M, tag digest `0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319`, weights `sha256:1329cd5ab37e5ef6faef37cd8ff8660ffc96ed5b47b34fddf8eb0912e6c82466` (6,140,392,672 bytes) |
| Server | `n_ctx = 4096`, 1 slot, all 37 layers on the GPU, prompt cache off (`LLAMA_ARG_CACHE_RAM=0`); flash-attention state is recorded, not forced |
| Python packages | exactly those of `requirements.txt` (pillow 12.3.0, httpx 0.28.1, pydantic 2.13.4, …) installed with `--only-binary=:all:`. `verify_environment.py` rejects other versions (newer `anyio`, `idna` or `pydantic` fail) because the request body is serialized by `httpx` and must match the recorded fingerprints byte for byte |
| GPU | an NVIDIA GPU with ≥ 12 GB; the reference run used an RTX 3060 12 GB |

## How to run

```powershell
cd pillar3\rtx3060_benchmark
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt

# 1. working copy + checksum list; --frames-dir holds your authorized frames as <acq_id>\<sec:06d>.jpg
.\.venv\Scripts\python.exe prepare_replay.py --frames-dir D:\authorized\frames --output D:\forge-replay
cd D:\forge-replay
# verify_environment.py reads the frames under <replay>\frames\<acq_id>\<sec:06d>.jpg:
# copy or link your frames there as well.

# 2. model (Ollama stopped): either `ollama pull qwen3-vl:8b-instruct` and compare the digest,
#    or import the exact blob:
powershell -ExecutionPolicy Bypass -File .\model_transfer\import_model.ps1 -WeightsBlob <path-to-blob>

# 3. offline verification (no server needed): hashes, packages, GPU, binary, and offline
#    reconstruction of the 123 HTTP bodies compared with the recorded fingerprints
.\.venv\Scripts\python.exe verify_environment.py --offline

# 4. start the server (this STOPS any running Ollama process) and run
powershell -ExecutionPolicy Bypass -File .\start_ollama.ps1 -OllamaExe C:\ollama-0.32.15\ollama.exe
.\.venv\Scripts\python.exe benchmark.py
```

`benchmark.py` re-runs the verification with the server up, aborts before timing if the effective
context, slot count, offload or cache setting differ, then times the 120 proposals (≈ 10–15 minutes)
and writes `results/` (JSON, CSV, console log, environment, server log).

## Comparison with the original run

- `outputs_equal_verdict`: derived verdict (`vlm`) and class (`vlm_class`) equal to the original run.
- `outputs_equal_strict_incl_reason`: additionally the same `reason` text.
- The original run stored only the derived verdict, the class and `reason` truncated to 400
  characters, not the raw model JSON; the raw JSON of the replay is stored as `raw_model_json`.
- A different verdict is a **result**, not an error: do not re-run until it matches.

## Documented uncertainties

1. **GPU of the original run: not recorded** (see above).
2. **Context length, slot count and flash attention** were not recorded in the original run. They were
   observed in a later timing run with the same binary and model, and are labeled with their evidence
   level in `runtime/ollama_server_config.json`. The benchmark enforces `n_ctx = 4096` and one slot.
3. **Prompt cache.** The first 499 calls of the original run used the default prompt cache and the
   last 470 used `LLAMA_ARG_CACHE_RAM=0`; 78 re-run cases gave 78/78 identical verdicts. This
   benchmark uses `0`; the column `orig_session` says which session produced each expected output.
4. **Original HTTP bodies were not saved.** The fingerprints in `cases/request_fingerprints.json` were
   generated on the original machine with the same code and package versions; equivalence with the
   original run is inferred from file dates, schema/prompt hashes in the run metadata, and the
   agreement of an earlier timing replay.
5. **Ollama binary.** The original machine had 0.32.15 installed through the installer. The official
   zip is recommended because it does not auto-update; it was not verified that the zip contains
   byte-identical binaries (local hashes are listed in `runtime/ollama_server_config.json`).
6. An earlier 120-proposal timing replay on a different GPU (RTX 5060 Ti) exists; its files are not
   distributed and it is not used as a headline result.
