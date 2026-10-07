# Pillar 3 — local VLM as proposal corrector

For each YOLO proposal, a local VLM sees two images (the full frame with the box drawn, and a crop)
and returns a JSON verdict (keep / rejected / class-changed). Geometry is never changed by the VLM.

## Where the code is

| What | Path |
|---|---|
| Image + box preparation, schemas, human-verdict helpers | `../src/forge/vlm/prepare.py` |
| Ollama client (`/api/chat`, temperature 0, JSON schema) | `../src/forge/vlm/client.py` |
| Prompts and phases | `../src/forge/vlm/phases.py` (`phase1_prompt_v2`), rendered copies in `../prompts/` |
| Flow, curation helpers, scoring, report | `../src/forge/vlm/{flow,curate,score,report}.py` |
| The 969-proposal run (v2 "decomposed") and its batch driver | `../pillar2/repro/raw_yolo11x_313/run_vlm_v2_969.py`, `run_vlm_v2_969_batches.sh`; outputs in `…/vlm_v2_969/` |
| Experiments on semantic geometry and single-call class fix | `analysis/semantic_geometry.py`, `analysis/single_classfix.py`, `analysis/dump_prompts.py` (+ outputs and `pillar3-prompts.txt`) |
| Car/truck breakdown, VLM runtime-config builder, LaTeX appendix tables | `repro/` |
| Runtime benchmark (RTX 3060, 120 proposals + 3 warm-ups) | `rtx3060_benchmark/` |

`src/forge/vlm/` is the **working copy** used for the paper; `rtx3060_benchmark/source/forge/` is the
byte-for-byte copy that ran the original 969-proposal run (hashes in
`rtx3060_benchmark/provenance/source_provenance.json`). The files they share are identical except
`config.py`, where only the default data-root constant differs (the benchmark copy keeps the original
`D:/carcara-data` default so that the recorded hashes stay valid; the benchmark never uses it).

## Frozen configuration

Ollama **0.32.15**; `qwen3-vl:8b-instruct` Q4_K_M; manifest digest
`0533d74300e4f9bc367d675d4e64ffd073d50ff16a2b4096cc2e8a1cf8c96319`; temperature 0.0
(`top_p` 0.95 / `top_k` 20 from the Modelfile); no seed; JSON-schema constrained output;
`LLAMA_ARG_CACHE_RAM=0` for the final 470 of 969 calls. The server log of the original session was
not saved; later-observed settings are labeled as such in `../configs/ollama_server_config.json`.

## Runtime benchmark (RTX 3060 replay)

See [`rtx3060_benchmark/README.md`](rtx3060_benchmark/README.md) (English). In short: it replays 120
of the 969 frozen proposals (+ 3 warm-ups) with the pinned runtime and compares each answer with the
original run. It is a later runtime characterization; **it does not identify the GPU of the original
969-proposal run, which is not recorded**. It needs authorized copies of the original frames (not
distributed), the model weights (not distributed) and the pinned packages of its `requirements.txt`.

Tested in this preparation: `prepare_replay.py` and the start of `verify_environment.py` run; the steps
that need the frames, a GPU and Ollama were **not** executed here. Results of the RTX 3060 run are in
the Zenodo package (`06_experiment_results/pillar3_runtime_rtx3060/`).
