#!/usr/bin/env python
# compare_hicmaps.py  (Py3.6-friendly)
#
# Changes vs your version:
# - Adds --pred_scale to scale prediction in-memory (no need to rebuild .cool)
# - Uses shared vmin/vmax (based on ref percentile) for fair visual comparison
# - Plots REF / LOW / PRED heatmaps + correlation-vs-distance in a 1x4 figure
# - Prints means/sums for counts + log1p (helps diagnose global bias)

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

    # NEW: scale prediction in-memory (avoids rebuilding .cool)
    ap.add_argument("--pred_scale", type=float, default=1.0,
                    help="Multiply predicted counts by this factor before metrics/plots.")

    # Shared heatmap scale (based on ref log1p)
    ap.add_argument("--vmax_pct", type=float, default=99.5,
                    help="Percentile of REF log1p used as shared vmax for all heatmaps.")

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

    # Apply scaling to prediction IN-MEMORY
    if args.pred_scale != 1.0:
        mat_pred = mat_pred * float(args.pred_scale)

    n = min(mat_ref.shape[0], mat_low.shape[0], mat_pred.shape[0])
    mat_ref, mat_low, mat_pred = mat_ref[:n, :n], mat_low[:n, :n], mat_pred[:n, :n]

    chrom_nm = _norm_chrom_name(cref, args.chrom)

    # Overall metrics (log1p-space)
    prL, srL, mseL, ssimL = overall_metrics(mat_ref, mat_low)
    prP, srP, mseP, ssimP = overall_metrics(mat_ref, mat_pred)

    # Distance-stratified correlations
    d_low,  p_low,  s_low  = corr_vs_distance(mat_ref, mat_low,  bref, args.max_kbins, args.min_pts)
    d_pred, p_pred, s_pred = corr_vs_distance(mat_ref, mat_pred, bref, args.max_kbins, args.min_pts)

    # Diagnostics: means & sums
    ref_mean, low_mean, pred_mean = mat_ref.mean(), mat_low.mean(), mat_pred.mean()
    ref_sum,  low_sum,  pred_sum  = mat_ref.sum(),  mat_low.sum(),  mat_pred.sum()
    ref_lmean = np.log1p(mat_ref).mean()
    low_lmean = np.log1p(mat_low).mean()
    pred_lmean = np.log1p(mat_pred).mean()

    # --------------------------- plot ---------------------------
    fig, axes = plt.subplots(1, 4, figsize=(22, 5), constrained_layout=True)

    ref_log  = np.log1p(mat_ref)
    low_log  = np.log1p(mat_low)
    pred_log = np.log1p(mat_pred)

    vmin = 0.0
    finite_ref = ref_log[np.isfinite(ref_log)]
    vmax = float(np.percentile(finite_ref, args.vmax_pct)) if finite_ref.size else 1.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(ref_log)) if np.isfinite(np.nanmax(ref_log)) else 1.0

    im0 = axes[0].imshow(ref_log,  cmap="Reds", origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
    axes[0].set_title("High-res reference")

    im1 = axes[1].imshow(low_log,  cmap="Reds", origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
    axes[1].set_title("Low-res")

    im2 = axes[2].imshow(pred_log, cmap="Reds", origin="lower", aspect="equal", vmin=vmin, vmax=vmax)
    axes[2].set_title("Predicted (scaled x{:.3g})".format(args.pred_scale))

    for ax in axes[:3]:
        ax.set_xlabel("{} bins".format(chrom_nm))
        ax.set_ylabel("{} bins".format(chrom_nm))

    fig.colorbar(im2, ax=axes[:3], label="log(1 + counts)  (shared scale)")

    axes[3].plot(d_low,  p_low,  label="Pearson: low vs ref",  lw=1.8)
    axes[3].plot(d_pred, p_pred, label="Pearson: pred vs ref", lw=1.8)
    axes[3].plot(d_low,  s_low,  label="Spearman: low vs ref",  lw=1.8)
    axes[3].plot(d_pred, s_pred, label="Spearman: pred vs ref", lw=1.8)
    axes[3].set_xlabel("Genomic distance (kb)")
    axes[3].set_ylabel("Correlation")
    axes[3].set_ylim(0, 1.0)
    axes[3].set_title("Correlation vs distance")
    axes[3].legend(fontsize=8, loc="lower left")

    txt = (
        "Region: {}:{}-{}\n"
        "Low  vs Ref: P={:.3f} S={:.3f} MSE={:.4f} SSIM={:.3f}\n"
        "Pred vs Ref: P={:.3f} S={:.3f} MSE={:.4f} SSIM={:.3f}\n"
        "Means(counts): ref={:.4f} low={:.4f} pred={:.4f}\n"
        "Means(log1p):  ref={:.4f} low={:.4f} pred={:.4f}\n"
        "Sums(counts):  ref={:.2e} low={:.2e} pred={:.2e}\n"
        "Heatmap vmax: p{}(ref)={:.3f}"
    ).format(chrom_nm, args.start, args.end,
             prL, srL, mseL, ssimL,
             prP, srP, mseP, ssimP,
             ref_mean, low_mean, pred_mean,
             ref_lmean, low_lmean, pred_lmean,
             ref_sum, low_sum, pred_sum,
             args.vmax_pct, vmax)
    axes[3].text(0.01, 0.99, txt, transform=axes[3].transAxes,
                 va="top", ha="left", fontsize=8)

    plt.savefig(args.out, dpi=300)
    plt.close()

    # ------------------------ print & save ----------------------
    print("=== Overall metrics [{}:{}-{}] ===".format(chrom_nm, args.start, args.end))
    print("Low  vs Ref: Pearson={:.4f} Spearman={:.4f} MSE={:.4f} SSIM={:.4f}".format(prL, srL, mseL, ssimL))
    print("Pred vs Ref: Pearson={:.4f} Spearman={:.4f} MSE={:.4f} SSIM={:.4f}".format(prP, srP, mseP, ssimP))

    print("Means (counts): ref={:.4f} low={:.4f} pred={:.4f}".format(ref_mean, low_mean, pred_mean))
    print("Means (log1p):  ref={:.4f} low={:.4f} pred={:.4f}".format(ref_lmean, low_lmean, pred_lmean))
    print("Sums  (counts): ref={:.2e} low={:.2e} pred={:.2e}".format(ref_sum, low_sum, pred_sum))

    metrics_txt = args.out.replace(".png", "_metrics.txt")
    with open(metrics_txt, "w") as f:
        f.write("# region {}:{}-{}\n".format(chrom_nm, args.start, args.end))
        f.write("# pred_scale {}\n".format(args.pred_scale))
        f.write("# heatmap_vmax_pct {}\n".format(args.vmax_pct))
        f.write("# heatmap_vmax {}\n".format(vmax))
        f.write("low_vs_ref\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(prL, srL, mseL, ssimL))
        f.write("pred_vs_ref\t{:.6f}\t{:.6f}\t{:.6f}\t{:.6f}\n".format(prP, srP, mseP, ssimP))
        f.write("means_counts\t{:.6f}\t{:.6f}\t{:.6f}\n".format(ref_mean, low_mean, pred_mean))
        f.write("means_log1p\t{:.6f}\t{:.6f}\t{:.6f}\n".format(ref_lmean, low_lmean, pred_lmean))
        f.write("sums_counts\t{:.6e}\t{:.6e}\t{:.6e}\n".format(ref_sum, low_sum, pred_sum))


if __name__ == "__main__":
    main()