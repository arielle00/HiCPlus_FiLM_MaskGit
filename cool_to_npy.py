#!/usr/bin/env python3
"""
Convert .cool file to .npy format for MaskGit/VQGAN pipelines.

Adds optional "writeBed-style" scaling/quantization to mimic HiCPlus pred_chromosome -> writeBed -> cooler load:

writeBed behavior:
- optionally zero-out values < min_thresh
- vals = ceil(M * scale)
- vals[vals < min_count] = min_count     (applied only to nonzero positions)
- zeros stay zero

New flags:
  --scale FLOAT
  --min-thresh FLOAT
  --min-count INT
  --writebed-like     (apply ceil+min_count to nonzeros after scaling)

By default, this script DOES NOT quantize; it just loads the cool matrix and (optionally) log1p.

Notes:
- If your .cool was already created via writeBed, you probably should NOT pass --writebed-like again.
"""

import argparse
from pathlib import Path
import numpy as np
import cooler


def _match_chrom(chroms, chrom: str) -> str:
    if chrom is None:
        return chroms[0]
    chrom_norm = chrom.replace("chr", "")
    for cname in chroms:
        if cname == chrom:
            return cname
        if cname.replace("chr", "") == chrom_norm:
            return cname
    raise ValueError(f"Chromosome {chrom} not found. Example available: {chroms[:10]}")


def _fetch_matrix(c: cooler.Cooler, chrom: str, balance: bool,
                 start_bp: int = None, end_bp: int = None):
    if start_bp is not None and end_bp is not None:
        region = f"{chrom}:{start_bp}-{end_bp}"
        print(f"[INFO] Extracting region: {region}")
        mat = c.matrix(balance=balance).fetch(region)
    else:
        print(f"[INFO] Extracting full chromosome: {chrom}")
        mat = c.matrix(balance=balance).fetch(chrom)

    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    return np.asarray(mat)


def _apply_symmetry(mat: np.ndarray) -> np.ndarray:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return mat
    return 0.5 * (mat + mat.T)


def _tile_matrix(mat2d: np.ndarray, tile: int, stride: int):
    """
    mat2d: [H,W]
    returns:
      patches: [N,1,tile,tile] float32
      coords:  [N,2] int32 with (i,j) top-left offsets
    """
    H, W = mat2d.shape
    if H < tile or W < tile:
        raise ValueError(f"Matrix {mat2d.shape} smaller than tile={tile}")

    patches = []
    coords = []
    for i in range(0, H - tile + 1, stride):
        for j in range(0, W - tile + 1, stride):
            patches.append(mat2d[i:i + tile, j:j + tile])
            coords.append((i, j))

    patches = np.stack(patches, axis=0)[:, None, :, :]  # [N,1,tile,tile]
    coords = np.asarray(coords, dtype=np.int32)
    return patches.astype(np.float32, copy=False), coords


def _print_stats(name: str, x: np.ndarray):
    x = np.asarray(x)
    finite = np.isfinite(x)
    if not finite.all():
        print(f"[STATS] {name}: non-finite present ({(~finite).sum()}), will be nan_to_num")
    x2 = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[STATS] {name}: shape={x2.shape} min={x2.min():.6f} max={x2.max():.6f} "
          f"mean={x2.mean():.6f} nz_frac={(x2 > 0).mean():.6f}")


def _apply_writebed_like(
    mat_counts: np.ndarray,
    scale: float,
    min_thresh: float,
    min_count: int,
    do_writebed_like: bool,
) -> np.ndarray:
    """
    Operates in COUNT space.
    If do_writebed_like:
      - threshold tiny values
      - multiply by scale
      - ceil nonzeros and enforce min_count on nonzeros
    Else:
      - just threshold and scale (continuous)
    """
    m = mat_counts.astype(np.float32, copy=True)

    # clip negatives (cool matrices should be >=0 but balanced can have small negatives)
    m = np.clip(m, a_min=0.0, a_max=None)

    if min_thresh > 0:
        m[m < float(min_thresh)] = 0.0

    if scale != 1.0:
        m *= float(scale)

    if do_writebed_like:
        nz = m > 0
        if np.any(nz):
            vals = np.ceil(m[nz])
            if min_count > 0:
                vals[vals < min_count] = min_count
            m[nz] = vals.astype(np.float32)
        # zeros remain zero

    return m


def cool_to_npy(
    cool_file: str,
    output_file: str,
    chrom: str = None,
    balance: bool = False,
    log_transform: bool = True,
    start_bp: int = None,
    end_bp: int = None,
    sym: bool = True,
    tile: bool = False,
    tile_size: int = 128,
    stride: int = 128,
    save_coords: bool = True,
    # NEW (writeBed-style controls)
    scale: float = 1.0,
    min_thresh: float = 0.0,
    min_count: int = 1,
    writebed_like: bool = False,
):
    print(f"[INFO] Loading .cool file: {cool_file}")
    c = cooler.Cooler(cool_file)

    chroms = c.chromnames
    print(f"[INFO] Available chromosomes (first 10): {chroms[:10]}")

    chrom = _match_chrom(chroms, chrom)
    print(f"[INFO] Using chromosome: {chrom}")

    mat = _fetch_matrix(c, chrom=chrom, balance=balance, start_bp=start_bp, end_bp=end_bp)

    # finite numeric
    mat = np.asarray(mat)
    if not np.isfinite(mat).all():
        print("[WARN] Non-finite values found; replacing with 0.")
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    _print_stats("raw_from_cool", mat)

    # Symmetry
    if sym:
        if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
            mat = _apply_symmetry(mat)
            print("[INFO] Applied symmetry: (M + M.T) / 2")
        else:
            print("[WARN] --sym requested but matrix is not square; skipping symmetry.")

    # Apply writeBed-like scaling/quantization in COUNT space (before log1p)
    if (scale != 1.0) or (min_thresh > 0.0) or writebed_like:
        print(f"[INFO] Applying writeBed-style controls: scale={scale}, min_thresh={min_thresh}, "
              f"min_count={min_count}, writebed_like={writebed_like}")
        mat = _apply_writebed_like(
            mat_counts=mat,
            scale=scale,
            min_thresh=min_thresh,
            min_count=min_count,
            do_writebed_like=writebed_like,
        )
        _print_stats("after_scale_quantize_counts", mat)

    # log1p
    mat = np.asarray(mat, dtype=np.float32)
    if log_transform:
        print("[INFO] Applying log1p transform (after clipping to >= 0)")
        mat = np.log1p(np.clip(mat, a_min=0.0, a_max=None)).astype(np.float32, copy=False)
        _print_stats("after_log1p", mat)

    # Output
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not tile:
        np.save(str(output_path), mat)
        print(f"\n[SUCCESS] Saved full matrix to {output_path}")
        print(f"[INFO] Saved shape: {mat.shape}")
        print(f"[INFO] File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return mat

    patches, coords = _tile_matrix(mat, tile=tile_size, stride=stride)
    np.save(str(output_path), patches)

    print(f"\n[SUCCESS] Saved tiled patches to {output_path}")
    print(f"[INFO] Patches shape: {patches.shape}  (N,1,{tile_size},{tile_size})")
    print(f"[INFO] Num patches: {patches.shape[0]}")
    print(f"[INFO] File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    if save_coords:
        coords_path = output_path.with_suffix(output_path.suffix + ".coords.npy")
        np.save(str(coords_path), coords)
        print(f"[INFO] Saved tile coords to {coords_path} (shape {coords.shape})")
        print(f"[INFO] coords rows are (i,j) top-left offsets in the full matrix")

    return patches


def main():
    parser = argparse.ArgumentParser(description="Convert .cool to .npy (full matrix or tiled patches)")

    parser.add_argument("input_cool", type=str, help="Input .cool file")
    parser.add_argument("output_npy", type=str, help="Output .npy file path")

    parser.add_argument("--chrom", type=str, default=None,
                        help="Chromosome name (e.g., 'chr20' or '20'). If not specified, uses first chromosome.")
    parser.add_argument("--balance", action="store_true",
                        help="Use balanced matrix instead of raw counts")
    parser.add_argument("--no-log", action="store_true",
                        help="Don't apply log1p transform (use raw counts)")
    parser.add_argument("--start-bp", type=int, default=None,
                        help="Start position in bp (optional)")
    parser.add_argument("--end-bp", type=int, default=None,
                        help="End position in bp (optional)")

    parser.add_argument("--no-sym", action="store_true",
                        help="Disable symmetry enforcement (default is sym ON).")
    parser.add_argument("--tile", action="store_true",
                        help="Output tiled patches [N,1,tile,tile] instead of full [H,W].")
    parser.add_argument("--tile-size", type=int, default=128,
                        help="Patch tile size (default 128).")
    parser.add_argument("--stride", type=int, default=128,
                        help="Stride between tiles (default 128 = non-overlap, 64 = overlap).")
    parser.add_argument("--no-coords", action="store_true",
                        help="Do not save tile coordinates file (only applies when --tile).")

    # NEW: writeBed-like scaling/quantization controls
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Multiply counts by this factor BEFORE log1p (like writeBed(scale=...)).")
    parser.add_argument("--min-thresh", type=float, default=0.0,
                        help="Zero-out entries < this threshold BEFORE scaling/quantization (like writeBed(min_thresh)).")
    parser.add_argument("--min-count", type=int, default=1,
                        help="If --writebed-like, enforce at least this many counts on nonzero pixels.")
    parser.add_argument("--writebed-like", action="store_true",
                        help="Apply writeBed-style quantization: ceil(M*scale) and min_count on nonzeros.")

    args = parser.parse_args()

    cool_to_npy(
        cool_file=args.input_cool,
        output_file=args.output_npy,
        chrom=args.chrom,
        balance=args.balance,
        log_transform=not args.no_log,
        start_bp=args.start_bp,
        end_bp=args.end_bp,
        sym=not args.no_sym,
        tile=args.tile,
        tile_size=args.tile_size,
        stride=args.stride,
        save_coords=not args.no_coords,
        scale=args.scale,
        min_thresh=args.min_thresh,
        min_count=args.min_count,
        writebed_like=args.writebed_like,
    )


if __name__ == "__main__":
    main()
