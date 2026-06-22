"""
Batch-remap protocol segmentation labels to CoNeMOS channels for an entire BIDS dataset.

For every file matching *_{seg_suffix}.nii.gz under --bids-dir, applies the
label map defined in --label-map for --protocol and saves the result next to
the source file with '2con' inserted before the trailing '_dseg'.

Examples
--------
# DHCP (drawem9_albert protocol)
python scripts/remap_labels_to_conemos.py \
    --bids-dir /data/fetalDHCP/resampled05 \
    --protocol drawem9_albert \
    --seg-suffix desc-drawem9_dseg \
    --label-map label_map_multiprotocol.csv

# FeTA
python scripts/remap_labels_to_conemos.py \
    --bids-dir /data/feta_2.4 \
    --protocol feta \
    --seg-suffix rec-mial_dseg \
    --label-map label_map_multiprotocol.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def load_label_map(csv_path: Path, protocol: str) -> dict[int, int]:
    label_map = {}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["protocol"] == protocol:
                label_map[int(row["raw_label"])] = int(row["channel"])
    if not label_map:
        raise ValueError(f"No entries found for protocol '{protocol}' in {csv_path}")
    return label_map


def remap_array(data: np.ndarray, label_map: dict[int, int]) -> np.ndarray:
    out = np.zeros_like(data)
    for raw, channel in label_map.items():
        out[data == raw] = channel
    return out


def output_path(src: Path, seg_suffix: str) -> Path:
    # seg_suffix ends in '_dseg'; insert '2con' before that trailing '_dseg'
    assert seg_suffix.endswith("_dseg"), f"seg_suffix must end with '_dseg', got: {seg_suffix}"
    base_suffix = seg_suffix[: -len("_dseg")]          # e.g. 'desc-drawem9' or 'rec-irtk'
    new_suffix = f"{base_suffix}2con_dseg"             # e.g. 'desc-drawem92con_dseg'
    new_name = src.name.replace(f"_{seg_suffix}.nii.gz", f"_{new_suffix}.nii.gz")
    return src.parent / new_name


def main():
    parser = argparse.ArgumentParser(description="Remap segmentation labels to CoNeMOS channels.")
    parser.add_argument("--bids-dir", type=Path, required=True, help="Root of the BIDS dataset.")
    parser.add_argument("--protocol", required=True, help="Protocol name as in label_map CSV (e.g. feta, drawem9_albert).")
    parser.add_argument("--seg-suffix", required=True, help="Segmentation suffix without leading '_' and without '.nii.gz' (e.g. desc-drawem9_dseg or rec-mial_dseg).")
    parser.add_argument("--label-map", type=Path, required=True, help="Path to label_map_multiprotocol.csv.")
    args = parser.parse_args()

    if not args.bids_dir.is_dir():
        sys.exit(f"bids-dir not found: {args.bids_dir}")
    if not args.label_map.is_file():
        sys.exit(f"label-map not found: {args.label_map}")
    if not args.seg_suffix.endswith("_dseg"):
        sys.exit(f"--seg-suffix must end with '_dseg', got: {args.seg_suffix}")

    label_map = load_label_map(args.label_map, args.protocol)
    print(f"Label map for '{args.protocol}': {label_map}")

    pattern = f"*_{args.seg_suffix}.nii.gz"
    sources = sorted(args.bids_dir.rglob(pattern))

    if not sources:
        sys.exit(f"No files matching '{pattern}' found under {args.bids_dir}")

    print(f"Found {len(sources)} file(s) to remap.\n")

    for src in sources:
        dst = output_path(src, args.seg_suffix)
        img = nib.load(src)
        data = np.asarray(img.dataobj, dtype=np.int32)
        remapped = remap_array(data, label_map)
        nib.save(nib.Nifti1Image(remapped, img.affine, img.header), dst)
        print(f"  {src.name}  →  {dst.name}")

    print(f"\nDone. {len(sources)} file(s) saved.")


if __name__ == "__main__":
    main()
