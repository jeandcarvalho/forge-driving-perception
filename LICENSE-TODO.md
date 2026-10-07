# LICENSE — TO DO before publication

**No license has been chosen.** Until one is added, the code is "all rights reserved" by default.
This file is a placeholder and a review checklist, not a license grant and not legal advice. The
license terms below are recorded from general knowledge and from the artifacts in this repository;
**confirm each at release time**.

## 1. Own code we intend to license

| Part | Notes |
|---|---|
| `src/forge/`, `tests/` | the `forge` Python package (written for the research) |
| `pillar1/`, `pillar2/`, `pillar3/` scripts and reports | analysis scripts, the 969-proposal run, VLM prompts and flow |
| `pillar3/rtx3060_benchmark/` code (`benchmark.py`, `verify_environment.py`, `p3common.py`, `prepare_replay.py`, `start_ollama.ps1`) | benchmark harness; `source/forge/` is a byte copy of our own package |
| `platform_reference/` | TypeScript extracted from the **private CarCará platform repository** (also unlicensed). Needs the explicit agreement of whoever owns that code before it can carry a public license. |
| `prompts/`, `configs/`, `docs/` | prompts, JSON schema, run configurations, documentation |

Derived **data** (annotations, proposals, VLM outputs, metrics) lives in the Zenodo package and may
need a separate data license (e.g. CC-BY-4.0). The original **images** have no redistribution right and
are not part of any release.

## 2. External dependencies: license, and whether we copy or only use them

"Used" = imported or called at run time, installed by the user, not part of this repository.
"Copied" = bytes of the third party are in this repository.

| Component | Role | License (confirm) | Copied or used? | Review point |
|---|---|---|---|---|
| **Ultralytics** (`ultralytics`) | YOLO11x inference / labeling / training; imported by `forge.detect.*`, `pillar1/pillar1_content_arms.py`, `pillar1/measure_density.py`, `pillar2/repro/raw_yolo11x_313/run_raw_yolo11x.py` | **AGPL-3.0** (or paid Enterprise license) | **Used** (optional `ml` extra); not bundled | **Main point.** Code that imports an AGPL library and is distributed is generally expected to be AGPL-compatible. Choosing a permissive license for those modules needs a legal opinion; alternatives: license them AGPL-3.0, or isolate them. |
| **YOLO11x weights** (`yolo11x.pt`, sha256 `7bc158aa…3a24`) | detector | AGPL-3.0 (Ultralytics) | **Used**; not distributed (git-ignored) | The released detections are model outputs; check the Ultralytics terms for outputs. |
| **Ollama** | serves the VLM; we call its HTTP API (`forge.vlm.client`), `start_ollama.ps1` launches it | MIT | **Used** (external service). **Copied:** only its release checksum list (`runtime/ollama_v0.32.15_sha256sum.txt`) and the model-registry manifest JSON. No binaries. | none expected |
| **Qwen3-VL-8B-Instruct** (GGUF Q4_K_M via the Ollama registry) | Pillar 3 judge | **Apache-2.0** (the license blob embedded in the Ollama manifest, sha256 `7339fa41…85cb2`, is the Apache License 2.0) | **Used**; weights not distributed. **Copied:** the manifest/metadata JSON (`model_info.json`, which embeds the license text). | Redistributing weights would require keeping the Apache notices; we do not. |
| **pycocotools** | COCO mAP (`forge.evaluate.coco_map`) | BSD-2-Clause | **Used** (optional `ml` extra) | none expected |
| torch, torchvision | inference backend | BSD-3-Clause-style | Used | none expected |
| numpy | numerics | BSD-3-Clause | Used | none |
| pillow | image I/O | HPND (MIT-CMU style) | Used | none |
| httpx | Ollama / API client | BSD-3-Clause | Used | none |
| pydantic | data models | MIT | Used | none |
| typer | CLI | MIT | Used | none |
| tqdm | progress bars | MPL-2.0 AND MIT | Used | none |
| pyarrow | telemetry Parquet writer | Apache-2.0 | Used | none |
| Prisma, Fastify types, MongoDB driver | imported by the **non-runnable** TypeScript in `platform_reference/` | Apache-2.0 / MIT / Apache-2.0 | Used by the reference code only; not bundled | MongoDB server (SSPL) is never distributed |
| Google Drive / CDN | URL *templates* in `platform_reference/` and `src/forge/data/drive.py` | Google terms of service | Used by code that needs private identifiers; no identifier is in this repository | `drive.py` is imported by `forge pull` / `verify` / tests, so it stays in the package; it is outside every public reproduction path. Decide whether to keep it. |

We found no third-party source code vendored in this repository (no copyright or SPDX headers of
other parties in `src/`, `pillar*/`, `platform_reference/`, `tests/`).

## 3. Decisions required from the authors

1. **Ownership:** check the institutional IP policy (employer / university / funding agency) for the
   code, the annotations and the data.
2. **License of the code** (permissive vs. copyleft) — driven by the Ultralytics question above.
3. **License of the derived data** in the Zenodo package.
4. **Images:** no redistribution right, consent/ethics evidence or anonymization of faces/plates was
   found; the COCO `licenses` array is empty. They stay out of every release.

## 4. After deciding

Add `LICENSE` (and a data license), delete this file, update `CITATION.cff` (`license:`) and the README
header. Do **not** edit the frozen Zenodo package to add a license (its hashes are verified; its README
states that no license was assigned when it was assembled). `verify_package.py` prints the limitation
"No code/data reuse license assigned" as a fixed string; that message will then be outdated and can be
left alone or updated in a new package version.
