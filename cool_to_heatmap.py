import cooler
import matplotlib.pyplot as plt
import numpy as np

# Load the .cool file
c = cooler.Cooler("test.out.film.loog1p.sc.txt.cool")


# Fetch a block: chr19 (0–2 Mb) vs chr20 (64–64.2 Mb)
mat = c.matrix(balance=False).fetch("chr19:0-2000000", "chr19:0-2000000")

# Log transform (avoid log(0))
mat_log = np.log1p(mat)

# Plot with square aspect and nice colormap
plt.figure(figsize=(6, 6))
plt.imshow(mat_log, cmap="Reds", origin="lower", aspect="equal")
plt.colorbar(label="log(1 + contact counts)")
plt.title("Hi-C Contact Map (chr19 vs chr19, log-scaled)")
plt.xlabel("chr19 bins")
plt.ylabel("chr19 bins")
plt.tight_layout()

plt.savefig("hic_chr19_chr19_film_sc_log1p.png", dpi=300)
plt.close()
