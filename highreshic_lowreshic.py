#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
import h5py
import cooler

"""
Usage:
  python highres_lowres.py input.cool frac output.cool [chunk_size]

Also supports:
  file.mcool::resolutions/10000
"""

# -----------------------------
# Args
# -----------------------------
if len(sys.argv) < 4:
    print("Usage: python highres_lowres.py input.cool frac output.cool [chunk_size]")
    sys.exit(1)

in_uri = sys.argv[1]
frac = float(sys.argv[2])
out_cool = sys.argv[3]
chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else 5_000_000

if not (0.0 < frac <= 1.0):
    raise ValueError("frac must be in (0, 1]")

# -----------------------------
# Load bins
# -----------------------------
c = cooler.Cooler(in_uri)
bins = c.bins()[["chrom", "start", "end"]][:]

# Parse mcool URI
if "::" in in_uri:
    file_path, grp_path = in_uri.split("::", 1)
else:
    file_path, grp_path = in_uri, "/"

# -----------------------------
# Pixel iterators
# -----------------------------
def iter_pixels_modern():
    for df in c.pixels(chunksize=chunk_size):
        yield df[["bin1_id", "bin2_id", "count"]].copy()

def iter_pixels_h5():
    with h5py.File(file_path, "r") as f:
        grp = f[grp_path]
        p = grp["pixels"]
        n = p["bin1_id"].shape[0]

        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            yield pd.DataFrame({
                "bin1_id": p["bin1_id"][start:end],
                "bin2_id": p["bin2_id"][start:end],
                "count":   p["count"][start:end]
            })

# -----------------------------
# Thinning generator
# -----------------------------
def thinned_pixel_gen():
    rng = np.random.default_rng()

    # Try modern API, fallback to HDF5
    try:
        iterator = iter_pixels_modern()
        first = next(iterator)
    except Exception:
        iterator = iter_pixels_h5()
        first = next(iterator, None)

    def thin_and_dedup(df):
        cnt = df["count"].to_numpy(dtype=np.int64, copy=False)
        df = df.copy()
        df["count"] = rng.binomial(cnt, frac).astype(np.int32)
        df = df[df["count"] > 0]

        # REQUIRED: collapse duplicate pixels
        return (
            df.groupby(["bin1_id", "bin2_id"], as_index=False, sort=False)
              .agg({"count": "sum"})
        )

    if first is not None:
        yield thin_and_dedup(first)

    for df in iterator:
        yield thin_and_dedup(df)

# -----------------------------
# Write cooler
# -----------------------------
cooler.create_cooler(
    out_cool,
    bins=bins,
    pixels=thinned_pixel_gen(),
    ordered=True,
    symmetric_upper=True,
    dtypes={"count": "int32"}
)

print(f"Wrote {out_cool}")
