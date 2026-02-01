#!/usr/bin/env python3
"""
Convert .hic → .npy tiles for VQGAN / MaskGIT training (DIAGONAL ONLY),
and ALSO generate a *low-depth* (LR) version by binomial thinning in RAW-count space,
then applying the SAME preprocessing (log1p(counts) + filtering).

Note: Uses log1p(counts) normalization (matching HiCPlus+FiLM trainConvNet.py):
    - HR: clip to 100, then log1p
    - LR: multiply by 16, clip to 100, then log1p
    NOT O/E normalization.

Python 3.9-compatible (no PEP604 | unions, no built-in generics like tuple[...]).

Usage:
  python hic_to_npy_lr.py --hic file.hic --chrom 2 --resolution 10000 \
      --tile-size 128 --stride 64 --norm KR --frac 0.10 \
      --output chr02_lr.npy --output-hr chr02_hr.npy
"""

import argparse
from typing import Optional, Tuple

import numpy as np
import hicstraw
from tqdm import tqdm

# Normalization constants (matching hicplus/trainConvNet.py)
DOWN_SAMPLE_RATIO = 16
HIC_MAX_VALUE = 100


def tile_observed_over_expected(tile: np.ndarray) -> np.ndarray:
    """Per-tile O/E by distance stratum (in-place on a copy)."""
    H = tile.shape[0]
    out = tile.astype(np.float32, copy=True)

    # start at d=1 (skip main diagonal)
    for d in range(1, H):
        diag = np.diagonal(out, offset=d)
        if diag.size == 0:
            continue
        mean = diag.mean()
        if mean > 1e-8:
            i = np.arange(H - d)
            out[i, i + d] /= mean
            out[i + d, i] /= mean

    return out


def preprocess_tile(raw_tile_counts: np.ndarray, is_lr: bool = False) -> np.ndarray:
    """
    RAW counts -> log1p(counts) (matching HiCPlus+FiLM normalization).
    
    Normalization steps (matching hicplus/trainConvNet.py):
    - For LR: multiply by DOWN_SAMPLE_RATIO, then clip to HIC_MAX_VALUE, then log1p
    - For HR: clip to HIC_MAX_VALUE, then log1p
    
    Args:
        raw_tile_counts: [H,W] raw count matrix
        is_lr: If True, apply down_sample_ratio scaling (for LR tiles)
    
    Returns:
        [H,W] float32 log1p-normalized tile
    """
    x = raw_tile_counts.astype(np.float32, copy=False)
    
    # For LR: multiply by down_sample_ratio (matching trainConvNet.py line 246)
    if is_lr:
        x = x * DOWN_SAMPLE_RATIO
    
    # Clip to stabilize (matching trainConvNet.py lines 250-251)
    x = np.minimum(HIC_MAX_VALUE, x)
    
    # Apply log1p transformation (matching trainConvNet.py lines 259-260)
    x = np.log1p(x)
    
    return x.astype(np.float32, copy=False)


def thin_tile_binomial_symmetric(
    raw_tile_counts: np.ndarray,
    frac: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    Symmetry-preserving binomial thinning on a single tile.

    raw_tile_counts: [H,W], nonnegative (float ok, will be rounded to int)
    returns: [H,W] float32 (still in raw-count space, not log/OE)

    NOTE: Uses RandomState for broad compatibility. (default_rng is fine on 3.9,
          but RandomState makes it easy to keep everything "old-style".)
    """
    if not (0.0 < frac <= 1.0):
        raise ValueError("frac must be in (0, 1].")

    H, W = raw_tile_counts.shape
    if H != W:
        raise ValueError("Tile must be square.")

    # round to integer counts
    n = np.rint(np.clip(raw_tile_counts, 0, None)).astype(np.int64, copy=False)

    out = np.zeros((H, W), dtype=np.int64)

    # sample upper triangle (including diagonal), mirror to lower
    iu = np.triu_indices(H, k=0)
    upper_n = n[iu]

    # RandomState.binomial expects integer n and float p; returns int array
    upper_sample = rng.binomial(upper_n, frac).astype(np.int64, copy=False)

    out[iu] = upper_sample
    # mirror off-diagonals
    il = (iu[1], iu[0])
    out[il] = out[iu]

    return out.astype(np.float32)


def extract_tiles_diagonal_only_hr_lr(
    dense: np.ndarray,
    tile_size: int,
    stride: int,
    frac: float,
    min_nonzero_frac_hr: float,
    min_nonzero_frac_lr: float,
    seed: Optional[int],
    filter_on: str,
) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """
    Extract diagonal tiles from dense RAW counts matrix, create LR via thinning,
    preprocess both, and return (HR_pre, LR_pre) stacks aligned 1:1.

    filter_on:
      - "hr": apply min_nonzero_frac_hr on HR_pre first (recommended)
      - "lr": apply min_nonzero_frac_lr on LR_pre first
      - "both": require both pass their respective thresholds
      - "none": no filtering (not recommended)
    """
    # RandomState: if seed is None, uses unpredictable seed
    rng = np.random.RandomState(seed) if seed is not None else np.random.RandomState()

    hr_tiles = []
    lr_tiles = []

    n = dense.shape[0]
    kept = 0

    for i in range(0, n - tile_size + 1, stride):
        hr_raw = dense[i : i + tile_size, i : i + tile_size]
        if hr_raw.max() == 0:
            continue

        lr_raw = thin_tile_binomial_symmetric(hr_raw, frac=frac, rng=rng)

        # preprocess both into log1p(counts) representation (matching HiCPlus+FiLM)
        # HR: clip to HIC_MAX_VALUE, then log1p
        # LR: multiply by DOWN_SAMPLE_RATIO, clip to HIC_MAX_VALUE, then log1p
        hr_pre = preprocess_tile(hr_raw, is_lr=False)
        lr_pre = preprocess_tile(lr_raw, is_lr=True)

        # compute information content (same criterion style as your original script)
        hr_nzf = (np.abs(hr_pre) > 1e-6).mean()
        lr_nzf = (np.abs(lr_pre) > 1e-6).mean()

        if filter_on == "hr":
            keep = hr_nzf >= min_nonzero_frac_hr
        elif filter_on == "lr":
            keep = lr_nzf >= min_nonzero_frac_lr
        elif filter_on == "both":
            keep = (hr_nzf >= min_nonzero_frac_hr) and (lr_nzf >= min_nonzero_frac_lr)
        elif filter_on == "none":
            keep = True
        else:
            raise ValueError("Unknown filter_on: {}".format(filter_on))

        if not keep:
            continue

        hr_tiles.append(hr_pre)
        lr_tiles.append(lr_pre)
        kept += 1

    if kept == 0:
        raise RuntimeError("No valid diagonal tiles extracted (after filtering).")

    hr_stack = np.stack(hr_tiles).astype(np.float32) if len(hr_tiles) else None
    lr_stack = np.stack(lr_tiles).astype(np.float32)

    return hr_stack, lr_stack


def main(args):
    print("[INFO] Loading {}".format(args.hic))
    hic = hicstraw.HiCFile(args.hic)

    try:
        chroms = hic.chromosomes
    except AttributeError:
        chroms = hic.getChromosomes()

    chrom_sizes = {c.name: c.length for c in chroms}

    # helpful hint if naming mismatch (e.g., "2" vs "chr2")
    if args.chrom not in chrom_sizes:
        close = [k for k in chrom_sizes.keys() if k.replace("chr", "") == args.chrom.replace("chr", "")]
        msg = "Chromosome {} not found. Available example(s): {}".format(args.chrom, list(chrom_sizes.keys())[:10])
        if close:
            msg += " | Did you mean one of: {} ?".format(close)
        raise ValueError(msg)

    chrom_size = chrom_sizes[args.chrom]
    n_bins = chrom_size // args.resolution + 1

    print("[INFO] Chromosome {}".format(args.chrom))
    print("[INFO] Resolution {} bp → {} bins".format(args.resolution, n_bins))
    print("[INFO] LR thinning frac = {} (binomial thinning in RAW counts)".format(args.frac))

    records = hicstraw.straw(
        "observed",
        args.norm,
        args.hic,
        args.chrom,
        args.chrom,
        "BP",
        args.resolution,
    )

    dense = np.zeros((n_bins, n_bins), dtype=np.float32)

    nrec = 0
    for rec in tqdm(records, desc="Filling dense matrix"):
        nrec += 1
        i = rec.binX // args.resolution
        j = rec.binY // args.resolution
        dense[i, j] += rec.counts
        if i != j:
            dense[j, i] += rec.counts

    print("[INFO] Records read: {}".format(nrec))
    print("[INFO] Dense nonzero frac: {:.4f}".format((dense > 0).mean()))
    print("[INFO] Dense min/max: {} / {}".format(dense.min(), dense.max()))

    hr_tiles, lr_tiles = extract_tiles_diagonal_only_hr_lr(
        dense=dense,
        tile_size=args.tile_size,
        stride=args.stride,
        frac=args.frac,
        min_nonzero_frac_hr=args.min_nonzero_frac_hr,
        min_nonzero_frac_lr=args.min_nonzero_frac_lr,
        seed=args.seed,
        filter_on=args.filter_on,
    )

    # save LR (always)
    lr_tiles = lr_tiles[:, None, :, :]  # [N,1,H,W]
    print("[INFO] LR tiles shape: {}".format(lr_tiles.shape))
    print("[INFO] LR tiles min/max: {:.4f} / {:.4f}".format(lr_tiles.min(), lr_tiles.max()))
    np.save(args.output, lr_tiles)
    print("[DONE] Saved LR → {}".format(args.output))

    # optionally save HR aligned
    if args.output_hr is not None:
        if hr_tiles is None:
            raise RuntimeError("HR tiles are None but --output-hr was provided.")
        hr_tiles = hr_tiles[:, None, :, :]  # [N,1,H,W]
        print("[INFO] HR tiles shape: {}".format(hr_tiles.shape))
        print("[INFO] HR tiles min/max: {:.4f} / {:.4f}".format(hr_tiles.min(), hr_tiles.max()))
        np.save(args.output_hr, hr_tiles)
        print("[DONE] Saved HR → {}".format(args.output_hr))


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    p.add_argument("--hic", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--resolution", type=int, required=True)

    p.add_argument("--tile-size", type=int, default=40)
    p.add_argument("--stride", type=int, default=40)

    p.add_argument("--norm", default="NONE", choices=["NONE", "KR"])

    # thinning + reproducibility
    p.add_argument("--frac", type=float, required=True, help="Binomial thinning fraction in (0,1].")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for thinning (optional).")

    # Filtering controls
    p.add_argument(
        "--filter-on",
        default="hr",
        choices=["hr", "lr", "both", "none"],
        help="Which tiles must pass min_nonzero_frac threshold(s). Default: hr",
    )
    p.add_argument(
        "--min-nonzero-frac-hr",
        type=float,
        default=0.05,
        help="Min nonzero fraction threshold for HR_pre tiles (abs(log1p(counts))>1e-6).",
    )
    p.add_argument(
        "--min-nonzero-frac-lr",
        type=float,
        default=0.02,
        help="Min nonzero fraction threshold for LR_pre tiles (often lower because of thinning).",
    )

    # Outputs
    p.add_argument("--output", required=True, help="Output .npy path for LR tiles.")
    p.add_argument("--output-hr", default=None, help="Optional output .npy path for aligned HR tiles.")

    main(p.parse_args())
