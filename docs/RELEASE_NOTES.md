# FORGE v1.0.0

First archival research-code release corresponding to the experiments of the paper *FORGE: On-Demand
Dataset Curation and Model-Assisted Annotation for Driving Perception*.

Reproducibility artifacts (Zenodo): <https://doi.org/10.5281/zenodo.23104407>

## Contents

- **Pillar 1 research code** — frame selection, GPS spacing, near-duplicate, density and content analyses
  (`pillar1/`); the platform-side rules behind them as a non-runnable reference (`platform_reference/`).
- **Pillar 2 detection and evaluation code** — YOLO11x run on the 313 evaluation frames, matching, COCO
  metrics and scoring of the 969 frozen proposals against 731 human objects, re-analysis and comparison
  of configurations (`pillar2/`, `src/forge/detect`, `src/forge/evaluate`).
- **Pillar 3 VLM evaluation code** — image and box preparation, client, flow and scoring
  (`src/forge/vlm`), the 969-proposal run, semantic and geometric analyses, and the RTX 3060 runtime
  benchmark harness (`pillar3/`).
- **Prompts and schemas** (`prompts/`) and **run configurations** (`configs/`).
- **Reproduction utilities** (`reproduction/`), file provenance with hashes (`docs/SOURCE_MAP.csv`) and the
  list of every difference from the original research code (`PUBLIC_RELEASE_CHANGES.md`).
- **Tests** — 76 unit tests (2 more require `ultralytics`) with sanitized fixtures.

## Zenodo

The frozen, sanitized artifacts, the larger result files and the verification scripts are archived on
Zenodo and are not duplicated in the Git history. Pillar 1, 2 and 3 reported metrics are recomputed from
that package.

## Not distributed

Source images, GPS and telemetry, the private CarCará infrastructure (database and web application),
model weights and Ollama binaries.

## Limitations

- The original Pillar 1 query was not persisted; its reconstruction recovers 4361 of 4364 frames.
- The hardware of the original 969-proposal VLM run was not reliably recorded; the RTX 3060 benchmark is
  a later benchmark and not evidence of that hardware. Output equality across GPUs is not guaranteed
  (118/120 agreement in the replay).
- Pillar 1 selection and the detector and VLM re-runs require non-public data or authorized images.
- The license has not been chosen yet (`LICENSE-TODO.md`).
