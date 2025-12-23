#!/usr/bin/env python3
import sys, math, numpy as np, pandas as pd, h5py, cooler

"""
Usage:
  python3 thin_cool_compat.py input.cool 0.0625 downsampled.cool [chunk_size]

- Works with old cooler versions that don't support chunksize in c.pixels()
- Also accepts URIs like file.mcool::resolutions/10000
"""

if len(sys.argv) < 4:
    print("Usage: python3 thin_cool_compat.py input.cool frac out.cool [chunk_size]")
    sys.exit(1)

in_uri = sys.argv[1]
frac = float(sys.argv[2])
out_cool = sys.argv[3]
chunk_size = int(sys.argv[4]) if len(sys.argv) > 4 else 5000000


c = cooler.Cooler(in_uri)
bins = c.bins()[["chrom", "start", "end"]][:]


if "::" in in_uri:
    file_path, grp_path = in_uri.split("::", 1)
else:
    file_path, grp_path = in_uri, "/"

def iter_pixels_modern():

    for df in c.pixels(chunksize=chunk_size):
        yield df[["bin1_id","bin2_id","count"]].copy()

def iter_pixels_h5():

    with h5py.File(file_path, "r") as f:
        grp = f[grp_path]
        p = grp["pixels"]
        n = p["bin1_id"].shape[0]
        for start in range(0, n, chunk_size):
            end = min(n, start + chunk_size)
            df = pd.DataFrame({
                "bin1_id": p["bin1_id"][start:end],
                "bin2_id": p["bin2_id"][start:end],
                "count":   p["count"][start:end]
            })
            yield df

def thinned_pixel_gen():

    use_h5 = False
    try:
        iterator = iter_pixels_modern()

        first = next(iterator)
    except Exception:
        use_h5 = True

    if use_h5:
        iterator = iter_pixels_h5()
        first = next(iterator, None)

    rng = np.random.default_rng()

    def thin_counts(df):
        cnt = df["count"].to_numpy(dtype=np.int64, copy=False)
        df = df.copy()
        df["count"] = rng.binomial(cnt, frac).astype(np.int32)
        return df[df["count"] > 0]

    if first is not None:
        yield thin_counts(first)

    for df in iterator:
        yield thin_counts(df)

cooler.create_cooler(
    out_cool,
    bins=bins,
    pixels=thinned_pixel_gen(),
    ordered=True,
    symmetric_upper=True,
    dtypes={"count": "int32"}
)

print(f"Wrote {out_cool}")