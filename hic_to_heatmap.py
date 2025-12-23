from straw import straw
import numpy as np
import matplotlib.pyplot as plt

# Parameters
hic_file = "4DNFI1UEG1HD.hic"
chrom = "chr19"
region = "0:2000000"
binsize = 10000   # 10 kb resolution
normalization = "NONE"  # try "KR" for balanced

# Fetch data with straw
pixels = straw(normalization, hic_file,
               chrom, chrom,
               "BP", binsize,
               region, region)

# Convert to dense matrix
n_bins = (2000000 - 0) // binsize
mat = np.zeros((n_bins, n_bins))

for x, y, count in pixels:
    i = (x // binsize)
    j = (y // binsize)
    mat[i, j] = count
    mat[j, i] = count  # symmetrize

# Log transform
mat_log = np.log1p(mat)

# Plot
plt.figure(figsize=(6, 6))
plt.imshow(mat_log, cmap="Reds", origin="lower", aspect="equal")
plt.colorbar(label="log(1 + contact counts)")
plt.title("Original Hi-C Contact Map (chr19:0–2Mb, log-scaled)")
plt.xlabel("chr19 bins")
plt.ylabel("chr19 bins")
plt.tight_layout()
plt.savefig("hic_chr19_chr19_from_hic.png", dpi=300)
plt.close()
