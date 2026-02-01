#!/usr/bin/env python3
"""
cool_to_npy.py

Convert a .cool file to .npy for MaskGit/VQGAN pipelines.

Key improvements vs your version:
1) Optional near-diagonal tiling:
   - --diag-max-sep-bins K   keeps only tiles whose top-left offsets satisfy |i-j| <= K (in BIN units)
   - --diag-max-sep-bp   X   same idea but specified in basepairs (converted using cooler.binsize)
   This prevents generating mostly off-diagonal patches (which look "not like Hi-C diagonals").

   Practical default for "diagonal-looking 128x128 tiles":
     set K <= tile_size-1 (e.g., 127). If K is larger than tile_size, many tiles won't contain the diagonal.

2) Explicit handling of balanced matrices:
   - Balanced matrices can contain small negatives. By default we clip negatives to 0.
   - You can disable clipping via --no-clip-negatives (not recommended unless you know what you're doing).

3) Optional writeBed-like quantization (COUNT space) BEFORE log1p.

Outputs:
- If --tile: saves patches [N,1,tile,tile] float32
  and coords [N,2] int32 with (i_bin, j_bin) top-left bin offsets.
  Optionally also saves coords in bp as (i_bp, j_bp).

Notes:
- If your VQGAN/MaskGit were trained on log1p(raw counts), then for LR/HR/proposal conversion:
  use raw matrix (no --balance), apply log1p consistently, and do NOT re-apply writeBed-like steps
  unless you truly need to mimic a specific upstream bedGraph quantization.
"""

import argparse
from pathlib import Path
from typing import Optional
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


def _print_stats(name: str, x: np.ndarray):
    x = np.asarray(x)
    finite = np.isfinite(x)
    if not finite.all():
        print(f"[STATS] {name}: non-finite present ({(~finite).sum()}), will be nan_to_num")
    x2 = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    nz = (x2 > 0)
    print(
        f"[STATS] {name}: shape={x2.shape} dtype={x2.dtype} "
        f"min={x2.min():.6g} max={x2.max():.6g} mean={x2.mean():.6g} "
        f"nz_frac={nz.mean():.6g}"
    )


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
      - zeros remain zero

    Else:
      - just threshold and scale (continuous)
    """
    m = mat_counts.astype(np.float32, copy=True)

    # raw counts should be >=0; balanced can have negatives but you usually should not run writebed-like on balanced
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
    return m


def _tile_matrix(
    mat2d: np.ndarray,
    tile: int,
    stride: int,
    diag_max_sep_bins: Optional[int],
):
    """
    mat2d: [H,W]
    returns:
      patches: [N,1,tile,tile] float32
      coords:  [N,2] int32 with (i_bin, j_bin) top-left offsets
    """
    H, W = mat2d.shape
    if H < tile or W < tile:
        raise ValueError(f"Matrix {mat2d.shape} smaller than tile={tile}")

    patches = []
    coords = []

    for i in range(0, H - tile + 1, stride):
        for j in range(0, W - tile + 1, stride):
            if diag_max_sep_bins is not None:
                if abs(i - j) > diag_max_sep_bins:
                    continue
            patches.append(mat2d[i:i + tile, j:j + tile])
            coords.append((i, j))

    if len(patches) == 0:
        raise ValueError(
            "No tiles selected. If using --diag-max-sep-bins, it may be too small "
            "for your matrix/stride. Try increasing it or removing it."
        )

    patches = np.stack(patches, axis=0)[:, None, :, :]  # [N,1,tile,tile]
    coords = np.asarray(coords, dtype=np.int32)
    return patches.astype(np.float32, copy=False), coords


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
    save_coords_bp: bool = False,
    clip_negatives: bool = True,
    # writeBed-like controls (COUNT space)
    scale: float = 1.0,
    min_thresh: float = 0.0,
    min_count: int = 1,
    writebed_like: bool = False,
    # near-diag selection
    diag_max_sep_bins: Optional[int] = None,
    diag_max_sep_bp: Optional[int] = None,
):
    print(f"[INFO] Loading .cool file: {cool_file}")
    c = cooler.Cooler(cool_file)

    binsize = c.binsize  # may be None for variable bins; most Hi-C cools have fixed binsize
    print(f"[INFO] binsize: {binsize}")

    chroms = c.chromnames
    print(f"[INFO] Available chromosomes (first 10): {chroms[:10]}")

    chrom = _match_chrom(chroms, chrom)
    print(f"[INFO] Using chromosome: {chrom}")

    mat = _fetch_matrix(c, chrom=chrom, balance=balance, start_bp=start_bp, end_bp=end_bp)

    # numeric safety
    mat = np.asarray(mat)
    if not np.isfinite(mat).all():
        print("[WARN] Non-finite values found; replacing with 0.")
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)

    _print_stats("raw_from_cool", mat)

    # symmetry
    if sym:
        if mat.ndim == 2 and mat.shape[0] == mat.shape[1]:
            mat = _apply_symmetry(mat)
            print("[INFO] Applied symmetry: (M + M.T) / 2")
        else:
            print("[WARN] --sym requested but matrix is not square; skipping symmetry.")

    # balanced negatives
    if clip_negatives:
        if (mat < 0).any():
            neg_ct = int((mat < 0).sum())
            print(f"[INFO] Clipping negatives to 0 (count={neg_ct})")
        mat = np.clip(mat, a_min=0.0, a_max=None)

    # writeBed-like in COUNT space (only sensible on raw counts)
    if (scale != 1.0) or (min_thresh > 0.0) or writebed_like:
        if balance:
            print("[WARN] You enabled writeBed-like scaling while using --balance. "
                  "This is usually not what you want.")
        print(
            f"[INFO] Applying writeBed-style controls: scale={scale}, min_thresh={min_thresh}, "
            f"min_count={min_count}, writebed_like={writebed_like}"
        )
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
        print("[INFO] Applying log1p transform")
        mat = np.log1p(np.clip(mat, a_min=0.0, a_max=None)).astype(np.float32, copy=False)
        _print_stats("after_log1p", mat)

    # near-diag selection conversion bp->bins (uses top-left bin coords)
    if diag_max_sep_bp is not None:
        if binsize is None:
            raise ValueError("--diag-max-sep-bp requires fixed binsize in the .cool file.")
        diag_max_sep_bins = int(np.floor(diag_max_sep_bp / binsize))
        print(f"[INFO] diag_max_sep_bp={diag_max_sep_bp} => diag_max_sep_bins={diag_max_sep_bins}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not tile:
        np.save(str(output_path), mat)
        print(f"\n[SUCCESS] Saved full matrix to {output_path}")
        print(f"[INFO] Saved shape: {mat.shape}")
        print(f"[INFO] File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
        return mat

    patches, coords = _tile_matrix(
        mat,
        tile=tile_size,
        stride=stride,
        diag_max_sep_bins=diag_max_sep_bins,
    )
    np.save(str(output_path), patches)

    print(f"\n[SUCCESS] Saved tiled patches to {output_path}")
    print(f"[INFO] Patches shape: {patches.shape}  (N,1,{tile_size},{tile_size})")
    print(f"[INFO] Num patches: {patches.shape[0]}")
    print(f"[INFO] File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")

    if save_coords:
        coords_path = output_path.with_suffix(output_path.suffix + ".coords.npy")
        np.save(str(coords_path), coords)
        print(f"[INFO] Saved tile coords (bin offsets) to {coords_path} (shape {coords.shape})")
        print("[INFO] coords rows are (i_bin, j_bin) top-left offsets in BIN units")

        if save_coords_bp:
            if binsize is None:
                raise ValueError("--save-coords-bp requires fixed binsize in the .cool file.")
            coords_bp = coords.astype(np.int64) * int(binsize)
            coords_bp_path = output_path.with_suffix(output_path.suffix + ".coords_bp.npy")
            np.save(str(coords_bp_path), coords_bp.astype(np.int64))
            print(f"[INFO] Saved tile coords (bp offsets) to {coords_bp_path} (shape {coords_bp.shape})")
            print("[INFO] coords_bp rows are (i_bp, j_bp) top-left offsets in basepairs")

    return patches


def main():
    p = argparse.ArgumentParser(description="Convert .cool to .npy (full matrix or tiled patches)")

    p.add_argument("input_cool", type=str, help="Input .cool file")
    p.add_argument("output_npy", type=str, help="Output .npy file path")

    p.add_argument("--chrom", type=str, default=None,
                   help="Chromosome name (e.g., 'chr20' or '20'). If not specified, uses first chromosome.")
    p.add_argument("--balance", action="store_true",
                   help="Use balanced matrix instead of raw counts.")
    p.add_argument("--no-log", action="store_true",
                   help="Don't apply log1p transform (use raw counts).")

    p.add_argument("--start-bp", type=int, default=None, help="Start position in bp (optional).")
    p.add_argument("--end-bp", type=int, default=None, help="End position in bp (optional).")

    p.add_argument("--no-sym", action="store_true",
                   help="Disable symmetry enforcement (default is sym ON).")

    p.add_argument("--tile", action="store_true",
                   help="Output tiled patches [N,1,tile,tile] instead of full [H,W].")
    p.add_argument("--tile-size", type=int, default=128,
                   help="Patch tile size (default 128).")
    p.add_argument("--stride", type=int, default=128,
                   help="Stride between tiles (default 128 = non-overlap, 64 = overlap).")

    p.add_argument("--no-coords", action="store_true",
                   help="Do not save tile coordinates file (only applies when --tile).")
    p.add_argument("--save-coords-bp", action="store_true",
                   help="Also save coords in basepairs as .coords_bp.npy (requires fixed binsize).")

    # balanced negative handling
    p.add_argument("--no-clip-negatives", action="store_true",
                   help="Do not clip negative values to 0 (not recommended).")

    # writeBed-like controls
    p.add_argument("--scale", type=float, default=1.0,
                   help="Multiply counts by this factor BEFORE log1p.")
    p.add_argument("--min-thresh", type=float, default=0.0,
                   help="Zero-out entries < this threshold BEFORE scaling/quantization.")
    p.add_argument("--min-count", type=int, default=1,
                   help="If --writebed-like, enforce at least this many counts on nonzero pixels.")
    p.add_argument("--writebed-like", action="store_true",
                   help="Apply writeBed-style quantization: ceil(M*scale) and min_count on nonzeros.")

    # near-diag selection (BIN or BP)
    p.add_argument("--diag-max-sep-bins", type=int, default=None,
                   help="Keep only tiles where |i_bin - j_bin| <= this value. "
                        "For diagonal-looking 128x128 tiles, use <= 127.")
    p.add_argument("--diag-max-sep-bp", type=int, default=None,
                   help="Same as --diag-max-sep-bins but in basepairs (converted using binsize).")

    args = p.parse_args()

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
        save_coords_bp=args.save_coords_bp,
        clip_negatives=not args.no_clip_negatives,
        scale=args.scale,
        min_thresh=args.min_thresh,
        min_count=args.min_count,
        writebed_like=args.writebed_like,
        diag_max_sep_bins=args.diag_max_sep_bins,
        diag_max_sep_bp=args.diag_max_sep_bp,
    )


if __name__ == "__main__":
    main()
