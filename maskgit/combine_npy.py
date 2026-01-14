import numpy as np
import argparse
from pathlib import Path
import re


CHR_RE = re.compile(r"chr(\d+|[XYM]+)_(hr|lr)\.npy$", re.IGNORECASE)


def list_chr_files(data_dir: Path, kind: str):
    """
    kind: "hr" or "lr"
    returns: sorted list of Paths (sorted by chromosome order)
    """
    data_dir = Path(data_dir)
    files = []

    for p in data_dir.glob("*.npy"):
        m = CHR_RE.match(p.name)
        if not m:
            continue
        chr_id, suffix = m.group(1), m.group(2).lower()
        if suffix != kind:
            continue
        files.append(p)

    if not files:
        raise ValueError(f"No *_{kind}.npy files found in {data_dir}")

    def chr_sort_key(path: Path):
        m = CHR_RE.match(path.name)
        chr_id = m.group(1).upper()

        # numeric chromosomes first
        if chr_id.isdigit():
            return (0, int(chr_id))
        # then X, Y, M
        order = {"X": 1000, "Y": 1001, "M": 1002, "MT": 1002}
        return (1, order.get(chr_id, 2000))

    return sorted(files, key=chr_sort_key)


def combine_npy_files(npy_files, output_file):
    """
    Combine list of .npy files into one .npy (concat on axis 0).
    """
    print(f"[INFO] Files to combine ({len(npy_files)}):")
    for f in npy_files:
        print(f"  - {f.name}")

    tiles = []
    for npy_file in npy_files:
        print(f"[INFO] Loading {npy_file.name}...")
        data = np.load(npy_file)
        print(f"  Shape: {data.shape}")
        tiles.append(data)

    print(f"\n[INFO] Concatenating {len(tiles)} arrays...")
    combined = np.concatenate(tiles, axis=0)
    print(f"[INFO] Combined shape: {combined.shape}")
    print(f"[INFO] Combined min/max: {combined.min():.4f} / {combined.max():.4f}")

    np.save(output_file, combined)
    print(f"\n[DONE] Saved → {output_file}")
    return output_file


def get_chr_set(files):
    """Return set of chromosome IDs present in file list."""
    s = set()
    for p in files:
        m = CHR_RE.match(p.name)
        if m:
            s.add(m.group(1).upper())
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Combine chrXX_hr.npy and/or chrXX_lr.npy files into single .npy files."
    )

    parser.add_argument(
        "--hr-dir",
        type=str,
        default="chr_data_highres",
        help="Directory containing chr*_hr.npy files (default: chr_data_highres)",
    )
    parser.add_argument(
        "--lr-dir",
        type=str,
        default="chr_data_lowres",
        help="Directory containing chr*_lr.npy files (default: chr_data_lowres)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["hr", "lr", "both"],
        default="both",
        help="What to combine: hr only, lr only, or both (default: both)",
    )
    parser.add_argument(
        "--out-hr",
        type=str,
        default="hic_vqgan_train_hr.npy",
        help="Output filename for combined HR (default: hic_vqgan_train_hr.npy)",
    )
    parser.add_argument(
        "--out-lr",
        type=str,
        default="hic_vqgan_train_lr.npy",
        help="Output filename for combined LR (default: hic_vqgan_train_lr.npy)",
    )
    parser.add_argument(
        "--require-match",
        action="store_true",
        help="If set, require HR and LR to have the same chromosome set (useful when mode=both).",
    )

    args = parser.parse_args()

    if args.mode in ("hr", "both"):
        hr_files = list_chr_files(Path(args.hr_dir), "hr")
    else:
        hr_files = []

    if args.mode in ("lr", "both"):
        lr_files = list_chr_files(Path(args.lr_dir), "lr")
    else:
        lr_files = []

    if args.mode == "both" and args.require_match:
        hr_set = get_chr_set(hr_files)
        lr_set = get_chr_set(lr_files)
        if hr_set != lr_set:
            missing_in_lr = sorted(hr_set - lr_set)
            missing_in_hr = sorted(lr_set - hr_set)
            raise ValueError(
                "HR/LR chromosome sets do not match.\n"
                f"  Present in HR but missing in LR: {missing_in_lr}\n"
                f"  Present in LR but missing in HR: {missing_in_hr}\n"
                "Fix filenames or regenerate missing chromosomes, or run without --require-match."
            )

    if args.mode in ("hr", "both"):
        combine_npy_files(hr_files, args.out_hr)

    if args.mode in ("lr", "both"):
        combine_npy_files(lr_files, args.out_lr)
