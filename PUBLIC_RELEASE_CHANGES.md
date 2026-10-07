# PUBLIC_RELEASE_CHANGES

_In the tables below the original local user name is shown as `REPRO_USER`._

Every difference between the original research code and this repository. Scope: **portability and
privacy only**. No scientific logic, constant, threshold, prompt, schema, seed or result was changed.
Frozen result files (`p2p3_frozen_969.json`, `vlm_v2_969/*`, `*_results.json`, the whole
the Zenodo package, now outside this repository) are byte-identical to their sources, except for the user-name redaction described below: see `docs/SOURCE_MAP.csv`
(source and public SHA-256 side by side; `status` = identical / modified / moved-to-zenodo / zenodo-package-outside-git / removed-from-public-copy).

Portability convention: a hard-coded `D:/carcara-data` becomes `$FORGE_DATA_ROOT` (default `./data`),
the variable `forge.config` already used; other local paths became documented environment variables
(`FORGE_P1_FRAMELISTS`, `FORGE_OUT`, `FORGE_YOLO_WEIGHTS`, `FORGE_ACQS_JSON`) or relative paths.

## 1. Code edits

| file | change | reason | scientific logic |
|---|---|---|---|
| `pillar1/analyze_spacing_and_dup.py` | `ROOT = Path(r"D:\carcara-data\all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar1/decompose_spacing.py` | `ROOT = Path(r"D:\carcara-data\all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar1/decompose_spacing.py` | `ACQS = Path( ⏎     r"C:\Users\REPRO_USER\AppData\Local\Temp\claude\C--Users-REPRO_USER-Documents-carcara" ⏎     r"\0adef562-b9b5-426d-88eb-1d4e2a13be30\scratchp` → `ACQS = Path(os.environ.get("FORGE_ACQS_JSON", "acqs.json"))  # per-acquisition GPS list exported from the private database; not distributed` | Temporary scratch path of a local session replaced by `$FORGE_ACQS_JSON`. The input was an ad-hoc export from the private database and is not in this repository. | unchanged — only where a path/location is read |
| `pillar1/measure_density.py` | `ROOT = Path(r"D:\carcara-data\all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar1/measure_density.py` | `OUT = Path(r"C:\Users\REPRO_USER\AppData\Local\Temp\claude\C--Users-REPRO_USER-Documents-carcara\685b6d4c-9866-439d-8c6e-a41fbe3a1e8a\scratchpad")` → `OUT = Path(os.environ.get("FORGE_OUT", "out"))` | Temporary scratch path replaced by `$FORGE_OUT`. | unchanged — only where a path/location is read |
| `pillar1/measure_density.py` | `model = YOLO("C:/Users/REPRO_USER/Documents/carcara-forge/yolo11x.pt")` → `model = YOLO(os.environ.get("FORGE_YOLO_WEIGHTS", "yolo11x.pt"))` | Local weights path replaced by `$FORGE_YOLO_WEIGHTS` (weights are not redistributed). | unchanged — only where a path/location is read |
| `pillar1/pillar1_naive.py` | `ROOT = Path(r"D:\carcara-data\all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar1/pillars.py` | `ROOT = Path(r"D:\carcara-data\all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar1/spacing_full.py` | `ROOT = Path(r"D:\carcara-data\all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar1/tables3_naive.py` | `ROOT = Path(r"D:\carcara-data\all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar1/tables_34.py` | `ROOT = Path(r"D:\carcara-data\all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar2/ruler.py` | `ROOT = Path(r"D:\carcara-data\all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar2/split_gt_by_acquisition.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar2/analysis/compare_arms.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar2/analysis/reanalysis.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar2/analysis/sweep_repro.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar2/repro/raw_yolo11x_313/analyze_raw_p2p3.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar2/repro/raw_yolo11x_313/run_raw_yolo11x.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar2/repro/raw_yolo11x_313/run_vlm_v2_969.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar3/analysis/dump_prompts.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar3/analysis/semantic_geometry.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar3/analysis/single_classfix.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar3/repro/build_vlm_runtime_config.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar3/repro/car_truck_breakdown.py` | `ROOT = Path("D:/carcara-data/all-perception-in-one")` → `import os ⏎ ROOT = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one"` | Hard-coded `D:/carcara-data` replaced by `$FORGE_DATA_ROOT` (default `./data`), the variable already used by `forge.config`. | unchanged — only where a path/location is read |
| `pillar1/pillar1_content_arms.py` | `LISTS = Path(r"C:\Users\REPRO_USER\Documents\carcara\apps\api\scripts\out\pillar1-content-framelists-s42-51.json") ⏎ IMAGES = Path(r"D:\carcara-data\all-pe` → `import os ⏎ DATA = Path(os.environ.get("FORGE_DATA_ROOT", "data")) / "all-perception-in-one" ⏎ LISTS = Path(os.environ.get("FORGE_P1_FRAMELISTS", "pil` | Absolute local paths replaced by `$FORGE_DATA_ROOT`, `$FORGE_P1_FRAMELISTS` (private frame-list JSON, not distributed) and `$FORGE_OUT`. | unchanged — only where a path/location is read |
| `pillar1/pillar1_content_arms.py` | `OLD_LABELS = Path(r"D:\carcara-data\all-perception-in-one\labels\auto")` → `OLD_LABELS = DATA / "labels" / "auto"` | Same as above (root derived from `$FORGE_DATA_ROOT`). | unchanged — only where a path/location is read |
| `pillar1/pillar1_content_fetch.py` | `LISTS = Path(r"C:\Users\REPRO_USER\Documents\carcara\apps\api\scripts\out\pillar1-content-framelists-s42-51.json") ⏎ IMAGES = Path(r"D:\carcara-data\all-pe` → `import os ⏎ LISTS = Path(os.environ.get("FORGE_P1_FRAMELISTS", "pillar1-content-framelists-s42-51.json"))  # private: contains Drive ids ⏎ IMAGES = Pa` | Absolute local paths replaced by `$FORGE_P1_FRAMELISTS` and `$FORGE_DATA_ROOT`. | unchanged — only where a path/location is read |
| `src/forge/config.py` | `DEFAULT_DATA_ROOT = Path("D:/carcara-data")` → `DEFAULT_DATA_ROOT = Path("data")` | Default data root made relative; `$FORGE_DATA_ROOT` still overrides it. | unchanged — only where a path/location is read |
| `pillar2/repro/raw_yolo11x_313/run_vlm_v2_969_batches.sh` | `FORGE=/c/Users/REPRO_USER/Documents/carcara-forge ⏎ PY="$FORGE/.venv/Scripts/python.exe" ⏎ SCRIPT="$FORGE/paper/repro/raw_yolo11x_313/run_vlm_v2_969.py"` → `FORGE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)" ⏎ PY="${FORGE_PYTHON:-$FORGE/.venv/Scripts/python.exe}" ⏎ SCRIPT="$FORGE/pillar2/repro/` | Repository root derived from the script location; interpreter overridable with `$FORGE_PYTHON`; script path updated to the new layout. | unchanged — only where a path/location is read |
| `platform_reference/apps/api/scripts/pillar1-content-export.ts` | `const OLD_ARMS = 'D:/carcara-data/all-perception-in-one/arms'` → `const OLD_ARMS = join(process.env.FORGE_DATA_ROOT ?? 'data', 'all-perception-in-one', 'arms')` | Absolute local path replaced by `$FORGE_DATA_ROOT`. | unchanged — only where a path/location is read |
| `platform_reference/apps/api/scripts/repro-frame-classes.ts` | `const out = 'C:/Users/REPRO_USER/Documents/carcara-forge/paper/repro/corpus-frame-classes.json'` → `const out = 'corpus-frame-classes.json'` | Absolute output path replaced by a relative one. | unchanged — only where a path/location is read |
| `platform_reference/apps/api/scripts/repro-query-signature.ts` | `const out = 'C:/Users/REPRO_USER/Documents/carcara-forge/paper/repro/corpus-gps-yolo.json'` → `const out = 'corpus-gps-yolo.json'` | Absolute output path replaced by a relative one. | unchanged — only where a path/location is read |

32 edits in 28 files. Where the original did not import `os`, `import os` was added next to the edit.

## 2. Prose files with the local user name redacted

Same convention as the Zenodo package (`C:\Users\REPRO_USER` → `C:\Users\REPRO_USER`). Text only; no number touched.

- `configs/vlm_v2_969_experiment_meta.json`
- `pillar1/results/pillar1-query.txt`
- `pillar2/analysis/RELATORIO-PILLAR23.md`
- `pillar2/analysis/pillar23-sweep-repro.json`
- `pillar2/repro/raw_yolo11x_313/REPORT-RAW-YOLO11X-P2P3.md`
- `pillar2/repro/raw_yolo11x_313/config.json`
- `pillar2/repro/raw_yolo11x_313/manifest.json`
- `pillar2/repro/raw_yolo11x_313/manifest_conf020_sensitivity.json`
- `pillar2/repro/raw_yolo11x_313/p2p3_969_results.json`
- `pillar2/repro/raw_yolo11x_313/vlm_v2_969/meta.json`
- `pillar3/repro/vlm-runtime-config.json`
- `pillar3/rtx3060_benchmark/model_transfer/import_model.ps1`
- `pillar3/rtx3060_benchmark/runtime/experiment_meta.json`

(Includes path strings inside frozen *metadata/run-config* JSON. The JSON files that carry scientific results
were redacted the same way as in the Zenodo package and are byte-identical to the Zenodo copies; `p2p3_frozen_969.json`,
`phase1.json/jsonl` and `raw_predictions_tail.json` contain no user name and are untouched. Historical image paths such as
`D:\\carcara-data\\…\\<acq>\\<sec>.jpg` inside frozen records are kept: they are not personal and the hashes of the frozen files
are verified.)

## 2b. Test fixtures sanitized (`tests/fixtures/*.json`)

The three fixtures were raw pages of the private dataset manifest. For public release:
every `drive_id` was replaced by a synthetic 33-character id derived from its hash (`FAKE…`), the derived `source_link`,
`url_original` and `url_cdn` were rebuilt from it, every `gps` object (lat/lon/alt) was replaced by synthetic values,
and the curation field `generated_by` (an account label) became `redacted-account`. Acquisition IDs, seconds, classes, boxes,
verdicts and detection IDs are untouched. The unit tests do not depend on the removed values: `pytest` gives 76 passed / 2 skipped,
the same as before sanitizing. This concerns test data only, not experimental results.

## 3. Files removed from the public copy (present in the sources)

- `apps/api/scripts/enrich-yolo-bbox.mjs` — platform data backfill from a local CSV path; not part of the paper.
- `pillar3/rtx3060_benchmark/provenance/build_bundle.py` and `…/original_scripts/` — internal bundle-building script and copies of earlier run scripts with local paths (the run scripts themselves are in `pillar2/repro/raw_yolo11x_313/`).

## 4. Left as-is on purpose

- `pillar3/rtx3060_benchmark/source/forge/` is a byte-identical copy of the code that ran the 969-proposal run, hashed in `provenance/source_provenance.json`; its `config.py` keeps the default `D:/carcara-data` (unused by the benchmark).
- `src/forge/data/view.py` mentions `D:\carcara-data\` inside two user-facing message strings.
- `pillar2/repro/raw_yolo11x_313/SHA256SUMS.txt` describes the original files; edited files no longer match it (see its README note).
- Mongo ID of the target collection (`6a149cdc…`) remains as a constant in scripts and `.env.example`-style defaults (`src/forge/config.py`): an identifier, not a credential — pending author decision.
- Portuguese comments, reports and identifiers are untouched.
- Package name `carcara-forge` in `pyproject.toml` / `uv.lock` is unchanged (renaming would require editing the lock).

## 5. Layout changes (no content change)

Scripts keep the directory depth they had in `carcara-forge/paper/…` so their `sys.path`-based imports of `forge` still resolve: `pillar1/*.py` (was `paper/`), `pillar2|3/analysis/*.py` (was `paper/pillar23/`), `pillar2/repro/raw_yolo11x_313/` (was `paper/repro/raw_yolo11x_313/`). `pip install -e .` also makes `forge` importable.

## 6. Not copied (out of scope)

Figure exporters that need images and Windows fonts (`export_qualitative_*`, `compose_qualitative_2x2.py`, `export_fig3_revised.py`), contact sheets, `paper.rar`, `dossie.*`, `PATCH-main-tex.md`, `numeros.csv`, `.bak` files, `diagnostic_imgsz1280/`, `corpus-frame-classes.json` (9.8 MB per-frame corpus export; its generating script projects GPS and speed fields), the first-round curation backups, the web app, `NEXT.md`/`PROMPT.md`/`CLAUDE.md`/`AGENTS.md`, YOLO weights, model blobs, JPEGs.

## 7. Final preparation pass (GitHub vs. Zenodo split)

### 7.1 Zenodo package removed from the versioned tree
`reproduction/zenodo_package/` (the 148-file package `ZENODO_FORGE_v1.0.0`) is no longer part of the repository; it is the Zenodo deposit and lives outside the Git tree. Its content is unchanged and still passes `verify_package.py`. `.gitignore` blocks it from being re-added. `docs/SOURCE_MAP.csv` lists its files with status `zenodo-package-outside-git`.

### 7.2 Heavy frozen files moved to Zenodo
Each file below was removed from the GitHub tree after checking that a byte-identical copy exists in the package. `reproduction/place_frozen_artifacts.py` restores them (checksum-verified, git-ignored).

| removed from the repository | identical file in the Zenodo package |
|---|---|
| `pillar2/repro/raw_yolo11x_313/raw_predictions_tail.json` | `03_detector_proposals/raw_predictions_tail.json` |
| `pillar2/repro/raw_yolo11x_313/predictions_coco_tail.json` | `03_detector_proposals/predictions_coco_tail.json` |
| `pillar2/repro/raw_yolo11x_313/manifest.json` | `03_detector_proposals/manifest.json` |
| `pillar2/repro/raw_yolo11x_313/manifest.csv` | `03_detector_proposals/manifest.csv` |
| `pillar2/repro/raw_yolo11x_313/p2p3_frozen_969.json` | `03_detector_proposals/p2p3_frozen_969.json` |
| `pillar2/repro/raw_yolo11x_313/vlm_v2_969/phase1.json` | `04_vlm_outputs/vlm_v2_969/phase1.json` |
| `pillar2/repro/raw_yolo11x_313/vlm_v2_969/phase1.jsonl` | `04_vlm_outputs/vlm_v2_969/phase1.jsonl` |

Not part of the frozen package, so shipped as a **supplementary** Zenodo file (`SUPPLEMENTARY/pillar2_raw_yolo11x_313/`) instead of GitHub:
- `pillar2/repro/raw_yolo11x_313/manifest_conf020_sensitivity.json`
- `pillar2/repro/raw_yolo11x_313/manifest_conf020_sensitivity.csv`

### 7.3 Google-Drive downloader
`pillar1/pillar1_content_fetch.py` moved to `platform_reference/historical/pillar1_content_fetch.py` with a 'historical, not executable' banner (comment lines added at the top; code unchanged). `src/forge/data/drive.py` was **kept**: it is imported at module level by `forge.data.download`, `forge.cli` (`pull`), `forge.data.verify` and the tests, so removing it would mean editing package code. It holds only public URL templates, no identifiers or credentials, and no public reproduction path uses it.

### 7.4 Mongo collection IDs replaced by placeholders
`6a149cdc…` -> `<CARCARA_COLLECTION_ID>`, `6aa00b68…` -> `<CARCARA_ROUND2_COLLECTION_ID>` in the files below. The scripts that used the ID as a constant now need it to be supplied by editing the constant or, for `forge.config`, via `FORGE_COLLECTION_ID`. Not touched: `pillar3/rtx3060_benchmark/source/forge/config.py` (byte copy of the original run code; its hash is recorded in `provenance/source_provenance.json`), and everything inside the Zenodo package (frozen, hashed).

- `pillar1/results/pillar1-query.txt`
- `pillar2/analysis/RELATORIO-PILLAR23.md`
- `pillar3/analysis/single_classfix.py`
- `platform_reference/apps/api/scripts/out/pillar2/agreement-report.json`
- `platform_reference/apps/api/scripts/pillar1-content-export.ts`
- `platform_reference/apps/api/scripts/pillar1-spacing-arms.ts`
- `platform_reference/apps/api/scripts/repro-annotators-313.ts`
- `platform_reference/apps/api/scripts/repro-pillar1-uiquery.ts`
- `platform_reference/apps/api/scripts/repro-provenance.ts`
- `src/forge/config.py`
- `tests/fixtures/manifest_discarded_head.json`
- `tests/fixtures/manifest_test_full.json`
- `tests/fixtures/manifest_train_head.json`
- `tests/test_label_and_train.py`

### 7.5 Documentation rewritten / added
`README.md` (status table, GitHub vs. Zenodo), `reproduction/README.md`, `reproduction/place_frozen_artifacts.py` (new helper, not scientific code), `pillar3/rtx3060_benchmark/README.md` (rewritten in English; no `CLAUDE.md`, no references to undistributed folders; states that no JPEGs are distributed and that the RTX 3060 is not claimed for the original run), `LICENSE-TODO.md`, `CITATION.cff`, `docs/RELEASE_NOTES.md`, `pillar*/README.md`. The agent-instruction file `CLAUDE.md` of the benchmark bundle was never copied.
