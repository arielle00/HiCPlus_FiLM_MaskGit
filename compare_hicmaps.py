#!/usr/bin/env python
# compare_hicmaps.py  (Py3.6-friendly)

import argparse
import numpy as np
import matplotlib.pyplot as plt
import cooler
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim


# ---------------------------- helpers ----------------------------

def _binsize(c):
    """Get single-resolution binsize from a Cooler object."""
    try:
        return c.binsize
    except Exception:
        return int(c.info.get("bin-size"))

def _norm_chrom_name(c, chrom):
    """Return a chromosome name present in the cooler (handles 19 vs chr19)."""
    chroms = list(c.chromnames)
    s = str(chrom)
    if s in chroms:
        return s
    if "chr" + s in chroms:
        return "chr" + s
    for nm in chroms:
        if nm.replace("chr", "") == s:
            return nm
    raise ValueError("Chromosome {} not found. Available: {}".format(chrom, chroms))

def _region_str(c, chrom, start, end=None):
    nm = _norm_chrom_name(c, chrom)
    if end is None:
        end = int(c.chromsizes[nm])
    return "{}:{}-{}".format(nm, int(start), int(end))

def fetch_square(c, chrom, start, end):
    reg = _region_str(c, chrom, start, end)
    return c.matrix(balance=False).fetch(reg, reg)

def overall_metrics(A, B):
    """Compute Pearson, Spearman, MSE, SSIM on log1p matrices."""
    a = np.log1p(A); b = np.log1p(B)
    af = a.ravel(); bf = b.ravel()
    m = np.isfinite(af) & np.isfinite(bf)
    af = af[m]; bf = bf[m]
    if af.size == 0 or np.std(af) == 0 or np.std(bf) == 0:
        return np.nan, np.nan, np.nan, np.nan
    pr = pearsonr(af, bf)[0]
    sr = spearmanr(af, bf)[0]
    mse = mean_squared_error(af, bf)
    dr = float(max(a.max(), b.max()) - min(a.min(), b.min()) + 1e-9)
    s = ssim(a, b, data_range=dr)
    return pr, sr, mse, s

def corr_vs_distance(A, B, binsize, max_kbins=200, min_pts=25):
    """Pearson & Spearman along diagonals (distance-stratified)."""
    a = np.log1p(A); b = np.log1p(B)
    n = min(a.shape)
    max_k = min(max_kbins, n - 1)
    dists, pears, spears = [], [], []
    for d in range(1, max_k):
        t = np.diag(a, k=d); p = np.diag(b, k=d)
        m = np.isfinite(t) & np.isfinite(p)
        t = t[m]; p = p[m]
        if t.size >= min_pts and np.std(t) > 0 and np.std(p) > 0:
            dists.append(d * binsize / 1000.0)           # kb
            pears.append(pearsonr(t, p)[0])
            spears.append(spearmanr(t, p)[0])
    return np.array(dists), np.array(pears), np.array(spears)


# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Compare low/pred vs high-res reference Hi-C (coolers).")
    ap.add_argument("--ref",  required=True, help="High-res reference .cool (or mcool::path)")
    ap.add_argument("--low",  required=True, help="Low-coverage/downsampled .cool (same binsize)")
    ap.add_argument("--pred", required=True, help="Predicted/enhanced .cool (same binsize)")
    ap.add_argument("--chrom", default="19", help="Chromosome (e.g., 19 or chr19)")
    ap.add_argument("--start", type=int, default=0, help="Start bp (default 0)")
    ap.add_argument("--end",   type=int, default=2000000, help="End bp (default 2 Mb)")
    ap.add_argument("--max_kbins", type=int, default=200, help="Max diagonal offset (bins)")
    ap.add_argument("--min_pts",   type=int, default=25,  help="Min pairs per distance bin")
    ap.add_argument("--out",  default="compare_chr_metrics.png", help="Output figure")
    args = ap.parse_args()

    # Load
    cref = cooler.Cooler(args.ref)
    clow = cooler.Cooler(args.low)
    cpred = cooler.Cooler(args.pred)

    # Same binsize check
    bref, blow, bpred = _binsize(cref), _binsize(clow), _binsize(cpred)
    if not (bref == blow == bpred):
        raise RuntimeError("Bin sizes differ: ref={}, low={}, pred={}".format(bref, blow, bpred))

    # Fetch region and crop to common square
    mat_ref  = fetch_square(cref,  args.chrom, args.start, args.end)
    mat_low  = fetch_square(clow,  args.chrom, args.start, args.end)
    mat_pred = fetch_square(cpred, args.chrom, args.start, args.end)
    n = min(mat_ref.shape[0], mat_low.shape[0], mat_pred.shape[0])
    mat_ref, mat_low, mat_pred = mat_ref[:n, :n], mat_low[:n, :n], mat_pred[:n, :n]

    # Overall metrics
    prL, srL, mseL, ssimL = overall_metrics(mat_ref, mat_low)
    prP, srP, mseP, ssimP = overall_metrics(mat_ref, mat_pred)

    # Distance-stratified correlations
    d_low,  p_low,  s_low  = corr_vs_distance(mat_ref, mat_low,  bref, args.max_kbins, args.min_pts)
    d_pred, p_pred, s_pred = corr_vs_distance(mat_ref, mat_pred, bref, args.max_kbins, args.min_pts)

    # --------------------------- plot ---------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    im0 = axes[0].imshow(np.log1p(mat_ref),  cmap="Reds", origin="lower", aspect="equal")
    axes[0].set_title("High-res reference")
    axes[0].set_xlabel("{} bins".format(_norm_chrom_name(cref, args.chrom)))
    axes[0].set_ylabel("{} bins".format(_norm_chrom_name(cref, args.chrom)))

    im1 = axes[1].imshow(np.log1p(mat_pred), cmap="Reds", origin="lower", aspect="equal")
    axes[1].set_title("Predicted")
    axes[1].set_xlabel("{} bins".format(_norm_chrom_name(cref, args.chrom)))
    axes[1].set_ylabel("{} bins".format(_norm_chrom_name(cref, args.chrom)))

    # single colorbar for both heatmaps
    fig.colorbar(im1, ax=axes[:2], label="log(1 + counts)")

    axes[2].plot(d_low,  p_low,  label="Pearson: low vs ref",  lw=1.8)
    axes[2].plot(d_pred, p_pred, label="Pearson: pred vs ref", lw=1.8)
    axes[2].plot(d_low,  s_low,  label="Spearman: low vs ref",  lw=1.8)
    axes[2].plot(d_pred, s_pred, label="Spearman: pred vs ref", lw=1.8)
    axes[2].set_xlabel("Genomic distance (kb)")
    axes[2].set_ylabel("Correlation"); axes[2].set_ylim(0, 1.0)
    axes[2].set_title("Correlation vs distance"); axes[2].legend(fontsize=8, loc="lower left")

    plt.savefig(args.out, dpi=300)
    plt.close()

    # ------------------------ print & save ----------------------
    print("=== Overall metrics [{}:{}-{}] ===".format(
        _norm_chrom_name(cref, args.chrom), args.start, args.end))
    print("Low  vs Ref: Pearson={:.4f} Spearman={:.4f} MSE={:.4f} SSIM={:.4f}".format(prL, srL, mseL, ssimL))
    print("Pred vs Ref: Pearson={:.4f} Spearman={:.4f} MSE={:.4f} SSIM={:.4f}".format(prP, srP, mseP, ssimP))

    metrics_txt = args.out.replace(".png", "_metrics.txt")
    with open(metrics_txt, "w") as f:
        f.write("# region {}:{}-{}\n".format(_norm_chrom_name(cref, args.chrom), args.start, args.end))
        f.write("low_vs_ref\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(prL, srL, mseL, ssimL))
        f.write("pred_vs_ref\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(prP, srP, mseP, ssimP))


if __name__ == "__main__":
    main()
