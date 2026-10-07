# Pillar 1 — curated selection vs. random sampling

Question: does platform-side selection (perception filters + GPS spacing + curation) give a better
dataset than random sampling from the same corpus? Measured as content (detections per frame, empty
frames) and redundancy (near-duplicates by GPS distance, nearest-neighbour distance).

## Files

| File | Role | Needs |
|---|---|---|
| `pillar1_content_arms.py` | YOLO11x (conf .25, NMS IoU .7, imgsz 640, batch 8) detection counts for the 21 arms (Forge once; naive-random and random+30 m spacing, seeds 42–51) | private images + private frame lists (`$FORGE_P1_FRAMELISTS`) + weights |
| `pillar1_naive.py`, `tables3_naive.py`, `tables_34.py`, `pillars.py` | density / redundancy tables, naive-random control over several seeds | private `manifest.json` (GPS) and labels |
| `spacing_full.py`, `decompose_spacing.py`, `analyze_spacing_and_dup.py` | near-duplicate rate and nearest-neighbour distance, decomposition of the spacing effect, audited denominator | private `manifest.json` (GPS); `decompose_spacing.py` also an ad-hoc `acqs.json` export (not distributed) |
| `measure_density.py` | density of the collection with the same detector | private images + weights |
| `results/` | derived outputs of the above (JSON/TeX/text) | public |

`platform_reference/historical/pillar1_content_fetch.py` is the archived downloader of the images of the extra random arm. It needs private storage identifiers, is not executable from this repository, and is kept only as a record of how those images were obtained.

Platform-side code (selection rule, spacing, query matching, export of the arm lists) is in
[`../platform_reference/`](../platform_reference/README.md).

## What is reproducible

- **From public artifacts:** the content totals and arm membership
  (Zenodo package: `06_experiment_results/pillar1/`, `01_dataset/`), re-checked by its
  `reproduce_results.py`.
- **Not reproducible without private data:** sampling the arms (needs MongoDB), the spatial metrics
  (need GPS), and re-running detection (needs the images).

## Known limitation

The exact search expression that produced the 4364-frame collection was not saved. Reconstructing it
through the UI query path reproduces 4361 of the 4364 frames (`results/pillar1-query.txt`,
`platform_reference/apps/api/scripts/repro-pillar1-uiquery.ts`). The three remaining frames are
unexplained additions; the paper's numbers refer to the real collection.

Environment variables used by these scripts: `FORGE_DATA_ROOT` (default `./data`),
`FORGE_P1_FRAMELISTS`, `FORGE_OUT` (default `./out`), `FORGE_YOLO_WEIGHTS`, `FORGE_ACQS_JSON`.
