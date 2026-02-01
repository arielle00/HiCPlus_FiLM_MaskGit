#!/usr/bin/env python3
"""
verify_normalization_space.py  (COORD-ALIGNED)

Verifies normalization space consistency across:
1) HR ref patches used for HiCPlus/FiLM training
2) LR patches fed to HiCPlus/FiLM
3) Targets used to train VQGAN/MaskGit (should be log1p(raw counts))

IMPORTANT FIX vs your previous script:
- Visualization is now COORD-ALIGNED across datasets.
  It uses the corresponding *.coords.npy files to show the SAME (i,j) patch
  in HR/LR/VQGAN columns, rather than random unrelated samples.

Assumptions:
- Patch arrays are either [N,128,128] or [N,1,128,128]
- Coord arrays are [N,2] int with (i,j) top-left offsets in bin units

Usage:
python3 verify_normalization_space.py \
  --hr_hicplus /home/012002744/hicplus_thesis/ref_chr20_patches.npy \
  --lr_hicplus /home/012002744/hicplus_thesis/low_res_chr20_patches.npy \
  --vqgan_target /home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_hr.npy \
  --output normalization_verification.png \
  --n_samples 8 \
  --diag_max_sep_bins 512 \
  --min_counts_sum 50 \
  --min_nonzero_frac 0.002 \
  --seed 0
"""

import argparse
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# IO helpers
# -------------------------
def ensure_nchw(x: np.ndarray) -> np.ndarray:
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"Expected [N,1,H,W] or [N,H,W], got {x.shape}")
    return x.astype(np.float32, copy=False)


def autodetect_coords(npy_path: str) -> str | None:
    p = Path(npy_path)
    cand = str(p) + ".coords.npy"
    return cand if Path(cand).exists() else None


def load_coords(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    if not Path(path).exists():
        return None
    c = np.load(path)
    if c.ndim != 2 or c.shape[1] != 2:
        raise ValueError(f"coords must be [N,2], got {c.shape} from {path}")
    return c.astype(np.int32, copy=False)


# -------------------------
# Space analysis
# -------------------------
def analyze_space(data_nchw: np.ndarray, name: str):
    data_flat = data_nchw.reshape(-1)

    non_zero = data_flat[data_flat > 0]
    stats = {
        "name": name,
        "shape": tuple(data_nchw.shape),
        "zero_fraction": float(1.0 - (len(non_zero) / data_flat.size)),
        "min": float(data_flat.min()),
        "max": float(data_flat.max()),
        "mean": float(data_flat.mean()),
        "median_nonzero": float(np.median(non_zero)) if len(non_zero) > 0 else 0.0,
        "std": float(data_flat.std()),
        "p95_nonzero": float(np.percentile(non_zero, 95)) if len(non_zero) > 0 else 0.0,
        "p99_nonzero": float(np.percentile(non_zero, 99)) if len(non_zero) > 0 else 0.0,
        "likely_space": "Unknown",
    }

    # crude heuristic
    if stats["max"] < 20:
        try:
            counts = np.expm1(data_flat)
            counts = counts[counts > 0]
            stats["back_to_counts_max"] = float(counts.max()) if counts.size else 0.0
            stats["back_to_counts_mean"] = float(counts.mean()) if counts.size else 0.0
            if stats["back_to_counts_max"] < 1e5:
                stats["likely_space"] = "log1p(counts) [verified-ish]"
        except Exception:
            pass
    elif stats["max"] > 100:
        stats["likely_space"] = "Raw counts (not log1p)"
    else:
        stats["likely_space"] = "Unknown"

    return stats


def compare_datasets(stats_list):
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    spaces = [s["likely_space"] for s in stats_list]
    if len(set(spaces)) == 1:
        print(f"✓ All datasets appear to be in the same space: {spaces[0]}")
    else:
        print("⚠ WARNING: Datasets appear to be in DIFFERENT spaces:")
        for s in stats_list:
            print(f"  - {s['name']}: {s['likely_space']}")

    print("\nKey Statistics (log-space):")
    header = f"{'Metric':<18} " + "".join(f"{s['name'][:18]:>20}" for s in stats_list)
    print(header)
    print("-" * len(header))

    for metric in ["mean", "median_nonzero", "std", "p95_nonzero", "p99_nonzero", "max"]:
        row = f"{metric:<18} " + "".join(f"{s[metric]:>20.3f}" for s in stats_list)
        print(row)

    print("\nZero fraction:")
    for s in stats_list:
        print(f"  {s['name']:<22}: {s['zero_fraction']*100:6.2f}%")

    print("\nBack-converted counts (only if log-like):")
    for s in stats_list:
        if "back_to_counts_max" in s:
            print(
                f"  {s['name']:<22}: max={s['back_to_counts_max']:.1f}, "
                f"mean(nonzero)={s['back_to_counts_mean']:.2f}"
            )


# -------------------------
# Coord-aligned sampling
# -------------------------
def build_coord_index(coords: np.ndarray) -> dict[tuple[int, int], int]:
    # if duplicates exist, keep first
    m = {}
    for i, (a, b) in enumerate(coords):
        key = (int(a), int(b))
        if key not in m:
            m[key] = i
    return m


def filter_coords(
    coords: np.ndarray,
    diag_max_sep_bins: int | None,
) -> np.ndarray:
    if diag_max_sep_bins is None:
        return coords
    d = np.abs(coords[:, 0] - coords[:, 1])
    return coords[d <= diag_max_sep_bins]


def pick_aligned_coords(
    hr_coords: np.ndarray | None,
    lr_coords: np.ndarray | None,
    vq_coords: np.ndarray | None,
    n_samples: int,
    seed: int,
    diag_max_sep_bins: int | None,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)

    coord_sets = []
    if hr_coords is not None:
        coord_sets.append(set(map(tuple, filter_coords(hr_coords, diag_max_sep_bins))))
    if lr_coords is not None:
        coord_sets.append(set(map(tuple, filter_coords(lr_coords, diag_max_sep_bins))))
    if vq_coords is not None:
        coord_sets.append(set(map(tuple, filter_coords(vq_coords, diag_max_sep_bins))))

    if not coord_sets:
        return []

    common = set.intersection(*coord_sets)
    common = list(common)

    if len(common) == 0:
        return []

    rng.shuffle(common)
    return [tuple(map(int, c)) for c in common[:n_samples]]


def patch_nonzero_frac(x_log1p: np.ndarray) -> float:
    # x_log1p: [1,H,W] or [H,W]
    z = x_log1p
    return float((z > 0).mean())


def patch_counts_sum(x_log1p: np.ndarray) -> float:
    # sum in count space
    return float(np.expm1(np.clip(x_log1p, 0, None)).sum())


def refine_coords_by_content(
    coords_list: list[tuple[int, int]],
    hr: np.ndarray | None,
    lr: np.ndarray | None,
    vq: np.ndarray | None,
    hr_map: dict[tuple[int, int], int] | None,
    lr_map: dict[tuple[int, int], int] | None,
    vq_map: dict[tuple[int, int], int] | None,
    min_counts_sum: float,
    min_nonzero_frac: float,
) -> list[tuple[int, int]]:
    out = []
    for c in coords_list:
        keep = True

        for arr, mp in [(hr, hr_map), (lr, lr_map), (vq, vq_map)]:
            if arr is None or mp is None:
                continue
            idx = mp.get(c, None)
            if idx is None:
                keep = False
                break
            patch = arr[idx, 0]
            if min_nonzero_frac > 0 and patch_nonzero_frac(patch) < min_nonzero_frac:
                keep = False
                break
            if min_counts_sum > 0 and patch_counts_sum(patch) < min_counts_sum:
                keep = False
                break

        if keep:
            out.append(c)

    return out


# -------------------------
# Visualization (aligned)
# -------------------------
def visualize_aligned(
    chosen_coords: list[tuple[int, int]],
    hr: np.ndarray | None,
    lr: np.ndarray | None,
    vq: np.ndarray | None,
    hr_map: dict[tuple[int, int], int] | None,
    lr_map: dict[tuple[int, int], int] | None,
    vq_map: dict[tuple[int, int], int] | None,
    output_path: str,
):
    cols = []
    if hr is not None:
        cols.append(("HR (HiCPlus training)", hr, hr_map))
    if lr is not None:
        cols.append(("LR (HiCPlus input)", lr, lr_map))
    if vq is not None:
        cols.append(("VQGAN/MaskGit target", vq, vq_map))

    n_rows = len(chosen_coords)
    n_cols = len(cols)

    if n_rows == 0 or n_cols == 0:
        print("[WARN] No aligned samples to visualize.")
        return

    # compute a consistent vmax from selected aligned patches across all columns
    all_vals = []
    for (name, arr, mp) in cols:
        for c in chosen_coords:
            idx = mp.get(c, None) if mp is not None else None
            if idx is None:
                continue
            all_vals.append(arr[idx, 0].reshape(-1))
    all_vals = np.concatenate(all_vals, axis=0)
    all_vals = all_vals[all_vals > 0]
    vmax = np.percentile(all_vals, 95) if all_vals.size else 1.0
    vmin = 0.0

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    if n_cols == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, coord in enumerate(chosen_coords):
        for c, (name, arr, mp) in enumerate(cols):
            ax = axes[r, c]
            idx = mp.get(coord, None) if mp is not None else None
            if idx is None:
                ax.axis("off")
                continue
            img = arr[idx, 0]
            ax.imshow(img, cmap="Reds", vmin=vmin, vmax=vmax)
            ax.set_title(f"{name}\ncoord={coord} idx={idx}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved COORD-ALIGNED visualization to: {output_path}")


# -------------------------
# Main
# -------------------------
def main():
    p = argparse.ArgumentParser(description="Verify normalization space consistency (coord-aligned visualization)")
    p.add_argument("--hr_hicplus", type=str, default=None, help="HR ref patches used for HiCPlus/FiLM training (.npy)")
    p.add_argument("--lr_hicplus", type=str, default=None, help="LR patches fed to HiCPlus/FiLM (.npy)")
    p.add_argument(
        "--vqgan_target",
        type=str,
        default="/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_hr.npy",
        help="Target patches used to train VQGAN/MaskGit (.npy)",
    )

    # coords (optional; if omitted, tries auto-detect by appending .coords.npy)
    p.add_argument("--hr_coords", type=str, default=None, help="HR coords .npy (defaults to hr_hicplus+'.coords.npy')")
    p.add_argument("--lr_coords", type=str, default=None, help="LR coords .npy (defaults to lr_hicplus+'.coords.npy')")
    p.add_argument("--vq_coords", type=str, default=None, help="VQ coords .npy (defaults to vqgan_target+'.coords.npy')")

    p.add_argument("--output", type=str, default="normalization_verification.png", help="Output image path")
    p.add_argument("--n_samples", type=int, default=8, help="Number of aligned patches to visualize")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")

    # filters
    p.add_argument("--diag_max_sep_bins", type=int, default=None, help="Keep only coords with |i-j| <= this")
    p.add_argument("--min_counts_sum", type=float, default=0.0, help="Drop patches with sum(expm1(x)) below this")
    p.add_argument("--min_nonzero_frac", type=float, default=0.0, help="Drop patches with nonzero_frac below this")

    args = p.parse_args()

    print("=" * 80)
    print("LOADING DATASETS")
    print("=" * 80)

    datasets = []
    hr = lr = vq = None

    if args.hr_hicplus:
        hr = ensure_nchw(np.load(args.hr_hicplus))
        datasets.append(("HR (HiCPlus training)", hr))
        print(f"[HR] {args.hr_hicplus}  shape={hr.shape}")
    else:
        print("[HR] not provided")

    if args.lr_hicplus:
        lr = ensure_nchw(np.load(args.lr_hicplus))
        datasets.append(("LR (HiCPlus input)", lr))
        print(f"[LR] {args.lr_hicplus}  shape={lr.shape}")
    else:
        print("[LR] not provided")

    vq = ensure_nchw(np.load(args.vqgan_target))
    datasets.append(("VQGAN/MaskGit target", vq))
    print(f"[VQ] {args.vqgan_target}  shape={vq.shape}")

    # coords
    hr_coords_path = args.hr_coords or (autodetect_coords(args.hr_hicplus) if args.hr_hicplus else None)
    lr_coords_path = args.lr_coords or (autodetect_coords(args.lr_hicplus) if args.lr_hicplus else None)
    vq_coords_path = args.vq_coords or autodetect_coords(args.vqgan_target)

    hr_coords = load_coords(hr_coords_path)
    lr_coords = load_coords(lr_coords_path)
    vq_coords = load_coords(vq_coords_path)

    print("\n" + "=" * 80)
    print("COORD FILES")
    print("=" * 80)
    print(f"hr_coords: {hr_coords_path if hr_coords is not None else 'None'}")
    print(f"lr_coords: {lr_coords_path if lr_coords is not None else 'None'}")
    print(f"vq_coords: {vq_coords_path if vq_coords is not None else 'None'}")

    hr_map = build_coord_index(hr_coords) if hr_coords is not None else None
    lr_map = build_coord_index(lr_coords) if lr_coords is not None else None
    vq_map = build_coord_index(vq_coords) if vq_coords is not None else None

    # analyze normalization space
    print("\n" + "=" * 80)
    print("ANALYZING NORMALIZATION SPACE")
    print("=" * 80)

    stats_list = []
    for name, arr in datasets:
        s = analyze_space(arr, name)
        stats_list.append(s)
        print(f"\n{name}:")
        print(f"  shape: {s['shape']}")
        print(f"  min/max: {s['min']:.4f} / {s['max']:.4f}")
        print(f"  mean/std: {s['mean']:.4f} / {s['std']:.4f}")
        print(f"  median(nonzero): {s['median_nonzero']:.4f}")
        print(f"  p95/p99(nonzero): {s['p95_nonzero']:.4f} / {s['p99_nonzero']:.4f}")
        print(f"  zero_fraction: {s['zero_fraction']:.2%}")
        print(f"  likely_space: {s['likely_space']}")
        if "back_to_counts_max" in s:
            print(f"  back_to_counts max/mean(nonzero): {s['back_to_counts_max']:.1f} / {s['back_to_counts_mean']:.2f}")

    compare_datasets(stats_list)

    # coord-aligned visualization
    print("\n" + "=" * 80)
    print("COORD-ALIGNED VISUALIZATION")
    print("=" * 80)

    if (hr_coords is None and hr is not None) or (lr_coords is None and lr is not None) or (vq_coords is None and vq is not None):
        print("[WARN] Missing coords for at least one dataset; cannot do coord-aligned visualization.")
        print("       Provide *.coords.npy (saved by your tiling script) or pass --*_coords explicitly.")
        return

    chosen = pick_aligned_coords(
        hr_coords=hr_coords,
        lr_coords=lr_coords,
        vq_coords=vq_coords,
        n_samples=max(args.n_samples * 5, args.n_samples),  # oversample before filtering by content
        seed=args.seed,
        diag_max_sep_bins=args.diag_max_sep_bins,
    )

    if len(chosen) == 0:
        print("[WARN] No common coords found across datasets (after diag filter).")
        return

    chosen = refine_coords_by_content(
        coords_list=chosen,
        hr=hr, lr=lr, vq=vq,
        hr_map=hr_map, lr_map=lr_map, vq_map=vq_map,
        min_counts_sum=args.min_counts_sum,
        min_nonzero_frac=args.min_nonzero_frac,
    )

    if len(chosen) == 0:
        print("[WARN] No coords left after content filtering (min_counts_sum / min_nonzero_frac).")
        return

    chosen = chosen[: args.n_samples]

    print("[INFO] chosen_coords:")
    for c in chosen:
        print(f"  {c}")

    visualize_aligned(
        chosen_coords=chosen,
        hr=hr, lr=lr, vq=vq,
        hr_map=hr_map, lr_map=lr_map, vq_map=vq_map,
        output_path=args.output,
    )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
