#!/usr/bin/env python3
"""
Quick sanity check: plot a Hi-C heatmap from a .hic file.

- Uses RAW observed counts
- NO region strings (hicstraw does not support them)
- Manual slicing to a small region
"""

import hicstraw
import numpy as np
import matplotlib.pyplot as plt

# =====================
# Parameters
# =====================
hic_file = "4DNFI7H4II2V.hic"
chrom = "19"          # IMPORTANT: use "19", not "chr19"
binsize = 10000       # 10 kb
norm = "NONE"

start_bp = 0
end_bp = 2_000_000    # 0–2 Mb region for visualization

# =====================
# Load Hi-C file
# =====================
hic = hicstraw.HiCFile(hic_file)

try:
    chroms = hic.chromosomes
except AttributeError:
    chroms = hic.getChromosomes()

chrom_sizes = {c.name: c.length for c in chroms}

if chrom not in chrom_sizes:
    raise ValueError(f"Chromosome {chrom} not found in .hic")

# =====================
# Bin coordinates
# =====================
start_bin = start_bp // binsize
end_bin = end_bp // binsize
n_bins = end_bin - start_bin

print(f"[INFO] Plotting chr{chrom}:{start_bp}-{end_bp} ({n_bins} bins)")

# =====================
# Read observed contacts
# =====================
records = hicstraw.straw(
    "observed",
    norm,
    hic_file,
    chrom,
    chrom,
    "BP",
    binsize
)

# =====================
# Build dense matrix
# =====================
mat = np.zeros((n_bins, n_bins), dtype=np.float32)

for rec in records:
    i = rec.binX // binsize
    j = rec.binY // binsize

    if start_bin <= i < end_bin and start_bin <= j < end_bin:
        ii = i - start_bin
        jj = j - start_bin
        mat[ii, jj] += rec.counts
        if ii != jj:
            mat[jj, ii] += rec.counts

print("[INFO] Dense matrix stats:")
print("  min / max:", mat.min(), mat.max())
print("  nonzero fraction:", (mat > 0).mean())

# =====================
# Log transform
# =====================
mat_log = np.log1p(mat)

# =====================
# Plot
# =====================
plt.figure(figsize=(6, 6))
plt.imshow(mat_log, cmap="inferno", origin="lower")
plt.colorbar(label="log(1 + contact counts)")
plt.title(f"Hi-C chr{chrom}: {start_bp//1_000_000}-{end_bp//1_000_000} Mb (10 kb)")
plt.xlabel("Genomic bin")
plt.ylabel("Genomic bin")
plt.tight_layout()
plt.savefig("hic_chr19_0_2Mb.png", dpi=300)
plt.close()

print("[DONE] Saved hic_chr19_0_2Mb.png")
