"""Copy the heavy frozen artifacts from an unpacked Zenodo package into the places where the
scripts of this repository expect them.

    python reproduction/place_frozen_artifacts.py --zenodo-dir /path/to/ZENODO_FORGE_v1.0.0 \
        [--supplementary-dir /path/to/SUPPLEMENTARY]

Nothing is downloaded and no existing file is overwritten with different content. Every copied
file is checked against the package's own SHA256SUMS.txt (the optional supplementary files have
no checksum list and are copied as they are). The placed files are git-ignored on purpose: they
live in the Zenodo record, not in this repository.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = "pillar2/repro/raw_yolo11x_313"

# (path inside the Zenodo package, destination inside this repository)
FROM_PACKAGE = [
    ("03_detector_proposals/raw_predictions_tail.json", f"{RAW}/raw_predictions_tail.json"),
    ("03_detector_proposals/predictions_coco_tail.json", f"{RAW}/predictions_coco_tail.json"),
    ("03_detector_proposals/manifest.json", f"{RAW}/manifest.json"),
    ("03_detector_proposals/manifest.csv", f"{RAW}/manifest.csv"),
    ("03_detector_proposals/p2p3_frozen_969.json", f"{RAW}/p2p3_frozen_969.json"),
    ("04_vlm_outputs/vlm_v2_969/phase1.json", f"{RAW}/vlm_v2_969/phase1.json"),
    ("04_vlm_outputs/vlm_v2_969/phase1.jsonl", f"{RAW}/vlm_v2_969/phase1.jsonl"),
]
# (file name in SUPPLEMENTARY/pillar2_raw_yolo11x_313, destination)
FROM_SUPPLEMENTARY = [
    ("manifest_conf020_sensitivity.json", f"{RAW}/manifest_conf020_sensitivity.json"),
    ("manifest_conf020_sensitivity.csv", f"{RAW}/manifest_conf020_sensitivity.csv"),
]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def put(src: Path, dst_rel: str, expected: str | None) -> bool:
    dst = REPO / dst_rel
    got = sha256(src)
    if expected is not None and got != expected:
        print(f"FAIL checksum  {src}", file=sys.stderr)
        return False
    if dst.exists():
        if sha256(dst) != got:
            print(f"FAIL exists with different content  {dst_rel}", file=sys.stderr)
            return False
        print(f"ok (already there)  {dst_rel}")
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"placed  {dst_rel}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zenodo-dir", type=Path, required=True)
    ap.add_argument("--supplementary-dir", type=Path)
    a = ap.parse_args()
    sums = {}
    for line in (a.zenodo_dir / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(None, 1)
            sums[name.strip().lstrip("*")] = digest
    ok = True
    for rel, dst in FROM_PACKAGE:
        if rel not in sums:
            print(f"FAIL not listed in SHA256SUMS.txt  {rel}", file=sys.stderr)
            ok = False
            continue
        ok &= put(a.zenodo_dir / rel, dst, sums[rel])
    if a.supplementary_dir:
        for name, dst in FROM_SUPPLEMENTARY:
            src = a.supplementary_dir / "pillar2_raw_yolo11x_313" / name
            ok &= put(src, dst, None) if src.exists() else (print(f"missing  {src}", file=sys.stderr) or False)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
