# Pillar 2 — YOLO11x proposals vs. human reference

Raw (un-tracked) YOLO11x proposals on the 313 evaluation frames, filtered to the six ground-truth
classes at conf ≥ 0.25 (969 frozen proposals), scored against 731 human objects.

## Files

| Path | Role |
|---|---|
| `repro/raw_yolo11x_313/run_raw_yolo11x.py` | runs YOLO11x on the 313 frames, writes raw predictions + manifest + config |
| `repro/raw_yolo11x_313/analyze_raw_p2p3.py`, `score_p2p3_969.py` | matching and scoring (C1, C2) of the 969 proposals, with and without the VLM |
| `repro/raw_yolo11x_313/{config.json,p2p3_969_results.json,analysis.json,…}` | small frozen outputs of the run (heavy ones: Zenodo) |
| `repro/raw_yolo11x_313/{P2P3-969-RESULTS.md,REPORT-RAW-YOLO11X-P2P3.md}` | run reports (Portuguese) |
| `analysis/reanalysis.py`, `sweep_repro.py`, `compare_arms.py` | audit re-analysis, confidence sweep, comparison of arms/configurations (+ their JSON/CSV outputs) |
| `ruler.py`, `split_gt_by_acquisition.py` | ground-truth audit (735 vs. 731 objects) and acquisition-exclusive split |
| `../src/forge/detect/infer.py`, `../src/forge/evaluate/` | inference, IoU matcher, COCO mAP, error decomposition |

Human curation (collections, the 63-frame second-annotator round, κ = 0.742) was done inside the
CarCará platform; its code is in [`../platform_reference/`](../platform_reference/README.md) and its
summary in `platform_reference/apps/api/scripts/out/pillar2/agreement-report.json`.

## Reproducibility

- **Public:** recomputation of the C1/C2 metrics from the frozen JSON (`reproduce_results.py`).
- **External model + images:** `run_raw_yolo11x.py` (weights `yolo11x.pt`, not distributed).
- **Private data:** every script above that reads `$FORGE_DATA_ROOT/all-perception-in-one/manifest.json`.

Large frozen files of this folder (`raw_predictions_tail.json`, `predictions_coco_tail.json`, `manifest.*`, `p2p3_frozen_969.json`, `vlm_v2_969/phase1.json[l]`, `manifest_conf020_sensitivity.*`) are **not in the GitHub repository**: they are in the Zenodo package (and its supplement). Restore them with `python reproduction/place_frozen_artifacts.py --zenodo-dir <package>`; the scripts here expect them next to themselves. The small result files (`p2p3_969_results.json`, `config.json`, `analysis.json`, the reports) are kept.

Note: `repro/raw_yolo11x_313/SHA256SUMS.txt` lists the hashes of the **original** files; the scripts
whose hardcoded paths were changed (and the two reports with the user name redacted) no longer match
it. The frozen data files (`p2p3_frozen_969.json`, `vlm_v2_969/phase1.json`, …) are byte-identical;
see `../docs/SOURCE_MAP.csv`.

Image-based figure exporters (`export_qualitative_*`, contact sheets) were deliberately left out.
