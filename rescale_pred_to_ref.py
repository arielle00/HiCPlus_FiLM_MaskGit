#!/usr/bin/env python3
import argparse
import numpy as np

def ensure_nchw(x):
    if x.ndim == 3:
        x = x[:, None, :, :]
    assert x.ndim == 4 and x.shape[1] == 1
    return x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_npy", required=True)
    ap.add_argument("--ref_npy", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--clip_alpha_min", type=float, default=0.01)
    ap.add_argument("--clip_alpha_max", type=float, default=10.0)
    args = ap.parse_args()

    pred = ensure_nchw(np.load(args.pred_npy)).astype(np.float32)
    ref  = ensure_nchw(np.load(args.ref_npy)).astype(np.float32)

    if pred.shape != ref.shape:
        raise ValueError(f"shape mismatch pred={pred.shape} ref={ref.shape}")

    # log1p -> counts
    pred_c = np.expm1(pred).clip(min=0)
    ref_c  = np.expm1(ref).clip(min=0)

    # per-patch mass
    pred_sum = pred_c.reshape(pred_c.shape[0], -1).sum(1)
    ref_sum  = ref_c.reshape(ref_c.shape[0], -1).sum(1)

    alpha = (ref_sum + args.eps) / (pred_sum + args.eps)
    alpha = np.clip(alpha, args.clip_alpha_min, args.clip_alpha_max).astype(np.float32)

    # apply scaling in count space, back to log1p
    pred_c_scaled = pred_c * alpha[:, None, None, None]
    pred_scaled = np.log1p(pred_c_scaled).astype(np.float32)

    np.save(args.out_npy, pred_scaled)

    print("alpha stats:", float(alpha.min()), float(np.median(alpha)), float(alpha.max()))
    # sanity: compare median sums
    new_sum = np.expm1(pred_scaled).reshape(pred_scaled.shape[0], -1).sum(1)
    print("median sums (counts): ref", float(np.median(ref_sum)),
          "pred_before", float(np.median(pred_sum)),
          "pred_after", float(np.median(new_sum)))

if __name__ == "__main__":
    main()
