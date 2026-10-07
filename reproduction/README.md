# reproduction

The frozen artifact package `ZENODO_FORGE_v1.0.0` is **not stored in this repository**. It is deposited
on Zenodo (DOI pending) as the long-term archive; it is kept out of the Git history on purpose
(≈ 20 MB, including a 7 MB raw-prediction file). Unpack it **outside** this checkout.

What the package contains (all file hashes in its `SHA256SUMS.txt` / `MANIFEST.csv`):

| Folder | Content |
|---|---|
| `01_dataset` | 4364-frame collection (no GPS), safe arm membership lists |
| `02_human_annotations` | 731-object COCO file and normalized reference of the 313 frames |
| `03_detector_proposals` | raw YOLO11x run: config, manifest, frozen 969 proposals, raw predictions |
| `04_vlm_outputs` | all 969 VLM responses (`phase1.json`, `.jsonl`, `meta.json`) |
| `05_evaluation_frames` | per-frame SHA-256 manifest of the 313 (restricted, not distributed) JPEGs |
| `06_experiment_results` | Pillar 1 / Pillars 2–3 / RTX 3060 benchmark results and logs |
| `07_prompts_and_schemas` | prompt template, rendered example, JSON schema |
| `08_reproduction_scripts` | **archival snapshot** of the scripts as deposited (user name replaced by `REPRO_USER`); the maintained copies are in this repository |
| `09_audit` | privacy/license audit and pre-archive validation |
| `verify_package.py`, `reproduce_results.py` | standard-library verification / metric recomputation |

```bash
python -B /path/to/ZENODO_FORGE_v1.0.0/verify_package.py
python -B /path/to/ZENODO_FORGE_v1.0.0/reproduce_results.py
python -B /path/to/ZENODO_FORGE_v1.0.0/verify_package.py --frames-dir /authorized/private/images   # optional
python reproduction/place_frozen_artifacts.py --zenodo-dir /path/to/ZENODO_FORGE_v1.0.0
```

`place_frozen_artifacts.py` copies the heavy frozen files back to the paths the scripts in
`pillar2/repro/raw_yolo11x_313/` expect (checksum-verified; the copies are git-ignored). Files not in the
frozen package (`manifest_conf020_sensitivity.*`) are shipped as a separate Zenodo supplement and are
placed with `--supplementary-dir`.

What the package **cannot** do: regenerate proposals or VLM outputs (needs authorized images and the
external models) or re-derive GPS-based Pillar 1 metrics (GPS is withheld).
