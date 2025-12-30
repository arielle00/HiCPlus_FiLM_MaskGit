#!/usr/bin/env python3
"""
Convert .hic → .npy tiles for VQGAN / MaskGIT training.

STRICTLY DIAGONAL-ONLY VERSION:

- Extracts ONLY tiles:
      dense[i:i+H, i:i+H]
- No off-diagonal tiles

Pipeline (per tile):
  observed (NONE or KR)
  → per-tile O/E (by distance)
  → log1p
  → low-information filtering

Output:
  .npy of shape [N, 1, H, W]
"""

import argparse
import numpy as np
import hicstraw
from tqdm import tqdm


def tile_observed_over_expected(tile: np.ndarray) -> np.ndarray:
    """Per-tile O/E by distance stratum."""
    H = tile.shape[0]
    out = tile.astype(np.float32, copy=True)

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


def extract_tiles_diagonal_only(
    dense: np.ndarray,
    tile_size: int,
    stride: int,
    min_nonzero_frac: float,
) -> np.ndarray:
    tiles = []
    n = dense.shape[0]

    for i in range(0, n - tile_size + 1, stride):
        tile = dense[i:i + tile_size, i:i + tile_size]

        if tile.max() == 0:
            continue

        tile = tile_observed_over_expected(tile)
        tile = np.log1p(tile)

        if (np.abs(tile) > 1e-6).mean() < min_nonzero_frac:
            continue

        tiles.append(tile)

    if len(tiles) == 0:
        raise RuntimeError("No valid diagonal tiles extracted")

    return np.stack(tiles).astype(np.float32)


def main(args):
    print(f"[INFO] Loading {args.hic}")
    hic = hicstraw.HiCFile(args.hic)

    try:
        chroms = hic.chromosomes
    except AttributeError:
        chroms = hic.getChromosomes()

    chrom_sizes = {c.name: c.length for c in chroms}
    if args.chrom not in chrom_sizes:
        raise ValueError(f"Chromosome {args.chrom} not found")

    chrom_size = chrom_sizes[args.chrom]
    n_bins = chrom_size // args.resolution + 1

    print(f"[INFO] Chromosome {args.chrom}")
    print(f"[INFO] Resolution {args.resolution} bp → {n_bins} bins")

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

    for rec in tqdm(records, desc="Filling dense matrix"):
        i = rec.binX // args.resolution
        j = rec.binY // args.resolution
        dense[i, j] += rec.counts
        if i != j:
            dense[j, i] += rec.counts

    print(f"[INFO] Dense nonzero frac: {(dense > 0).mean():.4f}")
    print(f"[INFO] Dense min/max: {dense.min()} / {dense.max()}")

    tiles = extract_tiles_diagonal_only(
        dense=dense,
        tile_size=args.tile_size,
        stride=args.stride,
        min_nonzero_frac=args.min_nonzero_frac,
    )

    tiles = tiles[:, None, :, :]  # [N,1,H,W]
    print(f"[INFO] Tiles shape: {tiles.shape}")
    print(f"[INFO] Tiles min/max: {tiles.min():.4f} / {tiles.max():.4f}")

    np.save(args.output, tiles)
    print(f"[DONE] Saved → {args.output}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--hic", required=True)
    p.add_argument("--chrom", required=True)
    p.add_argument("--resolution", type=int, required=True)
    p.add_argument("--tile-size", type=int, default=40)
    p.add_argument("--stride", type=int, default=40)
    p.add_argument("--min-nonzero-frac", type=float, default=0.05)
    p.add_argument("--norm", default="NONE", choices=["NONE", "KR"])
    p.add_argument("--output", required=True)
    main(p.parse_args())
