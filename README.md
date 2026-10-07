# FORGE

**On-Demand Dataset Curation and Model-Assisted Annotation for Driving Perception**

This repository contains the research code associated with the paper *FORGE: On-Demand Dataset Curation
and Model-Assisted Annotation for Driving Perception*: the experimental and reproduction code for the three
pillars of the study, together with the documentation needed to inspect and reproduce the reported
analyses.

- This is **not** the repository of the CarCará platform.
- CarCará was the infrastructure used during the research (acquisition indexing, search, curation).
  It is not published here; only the platform code that implements the paper's selection and curation
  rules is included, as a non-runnable reference (`platform_reference/`).
- This repository contains only the experimental and reproduction code associated with the article.
  The original images and private infrastructure are not distributed.

Reproducibility artifacts (frozen, sanitized data and results): <https://doi.org/10.5281/zenodo.23104407>

## Overview

| Pillar | Topic | Content |
|---|---|---|
| 1 | On-demand dataset curation | frame selection, perceptual filters, GPS spacing, redundancy and detection-density analyses |
| 2 | Detector-assisted annotation | YOLO11x proposals, human reference, matching and metrics |
| 3 | VLM-assisted proposal assessment | assessment of each proposal with Qwen3-VL, structured prompts, semantic and geometric analyses |

## Repository structure

```
src/forge/            Core research package
pillar1/              Dataset-selection experiments
pillar2/              Detection and annotation experiments
pillar3/              VLM experiments and benchmark
reproduction/         Reproduction utilities
prompts/              VLM prompts and schemas
configs/              Experiment configurations
tests/                Automated tests
platform_reference/   Selected CarCará reference implementation (not runnable)
docs/                 Release and provenance documentation
```

The 969-proposal run is a joint Pillar 2 / Pillar 3 experiment; its scripts are in
`pillar2/repro/raw_yolo11x_313/` and are indexed from `pillar3/README.md`.

## Reproducibility

| Component | Public reproduction status |
|---|---|
| Pillar 1 reported metrics | Reproducible from frozen artifacts (Zenodo) |
| Pillar 1 full acquisition/selection | Requires the non-public source dataset and infrastructure (private database with GPS and telemetry, original images) |
| Pillar 2 reported metrics | Reproducible from frozen artifacts (Zenodo) |
| Pillar 2 detector rerun | Requires the original authorized images and the model environment (`yolo11x.pt`, Ultralytics 8.4.120, Torch 2.11.0+cu128) |
| Pillar 3 frozen evaluation | Reproducible from frozen artifacts (Zenodo) |
| Pillar 3 complete VLM replay | Requires the original authorized frames and the pinned runtime (Ollama 0.32.15, `qwen3-vl:8b-instruct` Q4_K_M with the recorded digest, pinned Python packages); identical output across GPUs is not guaranteed |
| Pillar 3 runtime benchmark (RTX 3060) | Same requirements as the full replay; recorded results are in the Zenodo package |
| Unit tests of the `forge` package | Run from this repository alone (76 pass, 2 skipped without `ultralytics`) |

"Reproducible from frozen artifacts" means: the package verifier passes and the reported metrics are
recomputed from the frozen proposals, human reference and recorded VLM answers. It does not mean that
proposals or VLM outputs are regenerated.

## Reproduction artifacts

The frozen artifacts, larger result files and derived data used to reproduce the metrics are archived on
Zenodo (<https://doi.org/10.5281/zenodo.23104407>) and are deliberately **not** duplicated in the Git
history. They include the sanitized collection tables, the human reference, the 969 YOLO11x proposals,
the 969 recorded VLM answers, prompts and schemas, experiment results and logs, and the scripts
`verify_package.py` and `reproduce_results.py`.

## Quick start

Tests (Python 3.11 or 3.12):

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e . pytest
pytest
```

Verification and metric recomputation (standard library only, Python ≥ 3.10). Unpack the Zenodo package
outside this checkout, then:

```bash
python -B /path/to/ZENODO_FORGE_v1.0.0/verify_package.py
python -B /path/to/ZENODO_FORGE_v1.0.0/reproduce_results.py
```

Use `python -B`, not `-I`: the verifier imports its sibling script.

Scripts in `pillar2/repro/raw_yolo11x_313/` read large frozen files that live in the Zenodo package.
Restore them (checksum-verified, git-ignored) with:

```bash
python reproduction/place_frozen_artifacts.py --zenodo-dir /path/to/ZENODO_FORGE_v1.0.0
```

Scripts that additionally read the private dataset (`$FORGE_DATA_ROOT/all-perception-in-one/`) stop with a
`FileNotFoundError` when it is absent; see the table above. The VLM runtime benchmark has its own
instructions in `pillar3/rtx3060_benchmark/README.md`; those steps were not run on the machine used to
prepare this repository.

## Data availability

- Original images are not redistributed (redistribution rights, consent and anonymization of faces and
  license plates are not established).
- GPS and telemetry are not redistributed.
- Model weights (YOLO11x, Qwen3-VL) and Ollama binaries are not redistributed.
- The derived artifacts needed to reproduce the reported metrics are on Zenodo. Acquisition identifiers
  and timestamps are retained in them as residual metadata; no anonymization claim is made.
- Internal database identifiers are replaced by placeholders such as `<CARCARA_COLLECTION_ID>`.

## Known reproduction limitations

1. The original Pillar 1 query was not persisted. Its reconstruction through the platform interface
   recovers 4361 of the 4364 selected frames; three frames are unexplained
   (`pillar1/results/pillar1-query.txt`). The reported numbers refer to the actual 4364-frame collection.
2. The hardware used for the original 969-proposal VLM run has no reliable record in the available
   artifacts. Its metadata covers only the last 470 calls (4.51 s per call), which is not a full-run average.
3. The RTX 3060 benchmark is a later runtime benchmark (120 proposals plus 3 warm-ups, mean 7.43 s per
   proposal, verdict and class equal to the original run for 118/120). It must not be read as the hardware
   of the original run. Temperature 0 does not guarantee identical output across GPUs.
4. Pillar 1 spatial near-duplicates are defined by GPS distance, not visual similarity; GPS is not
   distributed, so spatial metrics cannot be re-derived from public material.
5. Several scripts and reports are historical research artifacts (some in Portuguese). Hash lists inside
   them describe the original files; every edit made for release is listed in
   `PUBLIC_RELEASE_CHANGES.md` and `docs/SOURCE_MAP.csv`.

## Citation

Carvalho, J. D., de Lima, J. J., dos Santos, G. M., Santos, M. M. D. *FORGE: On-Demand Dataset Curation and
Model-Assisted Annotation for Driving Perception.*

Reproducibility artifacts: <https://doi.org/10.5281/zenodo.23104407>. The paper has no DOI of its own yet.
Machine-readable metadata: `CITATION.cff`.

## License

The license of this repository is under review and has not been chosen yet; see
[`LICENSE-TODO.md`](LICENSE-TODO.md) for the code, the third-party dependencies and the open points.
Until a license is added, no reuse rights are granted.
