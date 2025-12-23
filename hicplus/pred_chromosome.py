#!/usr/bin/env python3
# Works with: bulk or single-cell | counts or log1p | HiCPlus or FiLM

import argparse
import numpy as np
import torch
from datetime import datetime

from hicplus import utils
from hicplus import model as model_mod

startTime = datetime.now()
use_gpu = 1  # set 0 to force CPU

# ---------------------------
# Defaults (override via CLI)
# ---------------------------
DEFAULT_IN_SCALE = 1.0     # neutral; set to your training downsample (e.g., 16) when needed
DEFAULT_CLIP_MAX = 100.0   # typical clip prior to log1p during training
DEFAULT_STRIDE   = 1
DEFAULT_PBATCH   = 512


# ---------------------------
# Helpers
# ---------------------------
def _device():
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def _load_ckpt(path, device):
    """
    Supports:
      - new: {'state_dict','arch','space','D_in','D_out','film': {...}}
      - legacy: raw state_dict
    """
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        return ckpt, ckpt["state_dict"]
    return {}, ckpt  # legacy weights only


def _infer_film_from_state_dict(state_dict):
    """Best-effort width/depth inference for FiLM when metadata is missing."""
    keys = list(state_dict.keys())
    film_gamma = [k for k in keys if "film1.gamma.weight" in k]
    if not film_gamma:
        return None, None
    cond = state_dict[film_gamma[0]].shape[1]  # channels_in for FiLM gamma
    width = int(cond * 2)
    block_ids = []
    for k in keys:
        if ".blocks." in k:
            try:
                block_ids.append(int(k.split(".blocks.")[1].split(".")[0]))
            except Exception:
                pass
    depth = (max(block_ids) + 1) if block_ids else None
    return width, depth


def _build_model_for_pred(ckpt_meta, args, device):
    """Prefer checkpoint metadata; allow CLI overrides for arch/space."""
    arch = getattr(args, "arch", None) or ckpt_meta.get("arch", "hicplus")

    # training space: from ckpt if present; else from CLI; else counts
    space = ckpt_meta.get("space", None)
    if getattr(args, "space", None):
        space = args.space
    if space is None:
        space = "counts"
    space = space.lower()

    D_in  = int(ckpt_meta.get("D_in", 40))
    D_out = int(ckpt_meta.get("D_out", 28))
    nonneg = not (space == "log1p")

    # FiLM hyperparams
    width = depth = use_aux = None
    film_meta = ckpt_meta.get("film", {})
    if film_meta:
        width   = film_meta.get("width",   None)
        depth   = film_meta.get("depth",   None)
        use_aux = film_meta.get("use_aux", None)

    if (width is None or depth is None) and hasattr(args, "_state_dict_for_infer"):
        w_inf, d_inf = _infer_film_from_state_dict(args._state_dict_for_infer)
        width = width or w_inf
        depth = depth or d_inf

    # CLI fallbacks
    if width is None:
        width = getattr(args, "width", 64)
    if depth is None:
        depth = getattr(args, "depth", 8)
    if use_aux is None:
        use_aux = not getattr(args, "no_aux", False)

    # Build model
    if hasattr(model_mod, "build_model"):
        net = model_mod.build_model(
            arch=arch, D_in=D_in, D_out=D_out,
            width=width, depth=depth, use_aux=use_aux, nonneg=nonneg
        )
    else:
        if arch == "film" and hasattr(model_mod, "FiLMNet"):
            net = model_mod.FiLMNet(D_in, D_out, width=width, depth=depth,
                                    use_aux=use_aux, nonneg=nonneg)
        else:
            net = model_mod.Net(D_in, D_out)

    return net.to(device), arch, space, D_in, D_out


def _preprocess_lr_batch(x_np, space, in_scale, clip_max):
    """
    Generic test-time preprocessing:
      raw counts -> *in_scale -> optional clip -> optional log1p
    - in_scale: set to your training downsample factor (e.g., 16), or the
      coverage correction you want (e.g., 1.25 for ×1.25).
    """
    x = x_np.astype(np.float32)
    if in_scale and in_scale != 1.0:
        x *= float(in_scale)
    if clip_max is not None:
        np.clip(x, 0.0, float(clip_max), out=x)
    if space == "log1p":
        x = np.log1p(x)
    return x


def _postprocess_pred_tensor(pred_t, space, out_clip_max=None):
    """Convert network output in training space back to counts numpy."""
    with torch.no_grad():
        if space == "log1p":
            pred_t = torch.expm1(pred_t).clamp_min(0)
        if out_clip_max is not None:
            pred_t = pred_t.clamp_max(float(out_clip_max))
    return pred_t.detach().cpu().numpy()


def _print_stats(name, M):
    nz = int((M > 0).sum())
    p = np.percentile(M[M > 0], [50, 90, 99]).tolist() if nz > 0 else [0, 0, 0]
    print(f"[stats] {name}: shape={M.shape} nz={nz} min={float(M.min()):.4g} "
          f"p50/90/99={p} max={float(M.max()):.4g}")


# ---------------------------
# Core prediction
# ---------------------------
def predict(M, N, inmodel, args):
    device = _device()

    # Load model
    ckpt_meta, state = _load_ckpt(inmodel, device)
    setattr(args, "_state_dict_for_infer", state)
    net, arch, space, D_in, D_out = _build_model_for_pred(ckpt_meta, args, device)
    net.load_state_dict(state, strict=False)
    net.eval()

    border = max(0, (D_in - D_out) // 2)

    # dense base matrix (counts)
    base = M.toarray().astype(np.float32) if hasattr(M, "toarray") else np.asarray(M, dtype=np.float32)

    if getattr(args, "debug", False):
        _print_stats("base_input_counts", base)

    # Overlap accumulators
    acc  = np.zeros((N, N), dtype=np.float32)
    wsum = np.zeros((N, N), dtype=np.float32)

    # Feathered weight window + core crop to hide seams
    w1d = np.hanning(D_out).astype(np.float32)
    w1d = 0.1 + 0.9 * w1d
    w2d = np.outer(w1d, w1d).astype(np.float32)
    b = 4
    core = np.zeros((D_out, D_out), np.float32)
    core[b:D_out-b, b:D_out-b] = 1.0
    w2d *= core

    # Stride & batch
    stride  = max(1, int(getattr(args, "stride", DEFAULT_STRIDE)))
    pred_bs = int(getattr(args, "pred_batch", DEFAULT_PBATCH))

    in_scale = float(getattr(args, "in_scale", DEFAULT_IN_SCALE))
    clip_max = getattr(args, "clip_max", DEFAULT_CLIP_MAX)
    out_clip = getattr(args, "out_clip_max", None)

    print(f"[pred] arch={arch} space={space} D_in={D_in} D_out={D_out} stride={stride} pred_bs={pred_bs} "
          f"in_scale={in_scale} clip_max={clip_max}")

    patches, coords = [], []

    def flush():
        nonlocal patches, coords, acc, wsum
        if not patches:
            return
        lr_np = np.concatenate(patches, axis=0)  # (B,1,D_in,D_in)
        lr_np = _preprocess_lr_batch(lr_np, space, in_scale, clip_max)
        lr_t  = torch.from_numpy(lr_np).to(device).float()

        with torch.no_grad():
            pred = net(lr_t)  # (B,1,D_out,D_out) in training space
            pred = _postprocess_pred_tensor(pred, space, out_clip)  # counts np
            pred = pred[:, 0]  # (B, D_out, D_out)

        for k, (x0, y0) in enumerate(coords):
            xs = x0 + border
            ys = y0 + border
            acc[xs:xs+D_out, ys:ys+D_out] += pred[k] * w2d
            wsum[xs:xs+D_out, ys:ys+D_out] += w2d

        patches, coords = [], []

    # Sliding window over the matrix
    for x in range(0, max(1, N - D_in + 1), stride):
        x = min(x, N - D_in)
        for y in range(0, max(1, N - D_in + 1), stride):
            y = min(y, N - D_in)
            patch = base[x:x+D_in, y:y+D_in][None, None, :, :]  # (1,1,D_in,D_in)
            patches.append(patch)
            coords.append((x, y))
            if len(patches) == pred_bs:
                flush()
    flush()

    # Average overlaps; fill untouched with base
    mask = wsum > 0
    prediction = np.where(mask, acc / np.maximum(wsum, 1e-8), base)

    # enforce symmetry unless disabled
    if not getattr(args, "no_sym", False):
        prediction = 0.5 * (prediction + prediction.T)

    if getattr(args, "debug", False):
        _print_stats("prediction_counts", prediction)

    return prediction.astype(np.float32)


def chr_pred(hicfile, chrN1, chrN2, binsize, inmodel, args):
    M = utils.matrix_extract(chrN1, chrN2, binsize, hicfile)
    N = M.shape[0]
    return predict(M, N, inmodel, args)


def writeBed(Mat, outname, binsize, chrN1, chrN2, scale=1.0, min_count=1, min_thresh=0.0):
    """
    Write a 7-column Juicer-style list from a dense counts matrix.
    - scale: multiply counts before integer cast (can rescue sub-1 values)
    - min_thresh: zero-out tiny predicted counts before rounding (e.g., 0.05 for log1p models)
    - min_count: floor to keep at least 1 read per written pixel
    """
    M = Mat.astype(np.float64, copy=True)
    if min_thresh > 0:
        M[M < float(min_thresh)] = 0.0

    nz = M > 0
    r, c = np.where(nz)
    if r.size == 0:
        print("[writeBed] Warning: matrix has no positive entries")
    vals = np.ceil(M[r, c] * float(scale))
    vals[vals < min_count] = min_count

    with open(outname, "w") as fh:
        for rr, cc, v in zip(r, c, vals.astype(np.int64)):
            fh.write(
                f"chr{chrN1}\t{rr*binsize}\t{(rr+1)*binsize}\t"
                f"chr{chrN2}\t{cc*binsize}\t{(cc+1)*binsize}\t{int(v)}\n"
            )
    print(f"[writeBed] wrote {len(r)} pixels to {outname} (scale={scale}, min={min_count}, min_thresh={min_thresh})")


def main(args=None):
    """
    If called from the top-level `hicplus` CLI, we receive an argparse.Namespace in `args`.
    If run directly (python pred_chromosome.py ...), we parse our own args here.
    """
    if args is None:
        ap = argparse.ArgumentParser(description="Predict/enhance Hi-C using HiCPlus/FiLM (bulk or single-cell).")
        ap.add_argument("-i", "--inputfile", required=True, help=".hic file (input)")
        ap.add_argument("-m", "--model",     required=True, help="checkpoint (.pt)")
        ap.add_argument("-o", "--outputfile",required=True, help="output BED-like predictions")
        ap.add_argument("-c", "--chrN",      nargs=2, type=int, required=True, help="chromosome pair, e.g. 19 19")
        ap.add_argument("-r", "--binsize",   type=int, required=True, help="bin size (e.g., 10000)")
        ap.add_argument("--arch",  default=None, choices=["hicplus","film"],
                        help="override architecture if checkpoint lacks it")
        ap.add_argument("--space", default=None, choices=["counts","log1p"],
                        help="training space; override if checkpoint lacks it")
        ap.add_argument("--stride", type=int, default=DEFAULT_STRIDE, help="sliding-window stride")
        ap.add_argument("--pred-batch", type=int, default=DEFAULT_PBATCH, help="micro-batch size")
        ap.add_argument("--no-sym", action="store_true", help="disable symmetry enforcement")
        ap.add_argument("--in-scale", type=float, default=DEFAULT_IN_SCALE,
                        help="multiply input counts to match training coverage (e.g., 16 or 1.25)")
        ap.add_argument("--clip-max", type=float, default=DEFAULT_CLIP_MAX,
                        help="clip input counts before log1p (None to disable)")
        ap.add_argument("--out-clip-max", type=float, default=None, help="clip predictions (counts space)")
        ap.add_argument("--scale", type=float, default=1.0,
                        help="multiply predictions before integer cast to BED")
        ap.add_argument("--min-count", type=int, default=1, help="minimum integer count to write")
        ap.add_argument("--min-thresh", type=float, default=0.0,
                        help="zero-out predicted counts below this before rounding to int")
        ap.add_argument("--debug", action="store_true", help="print quick matrix stats")
        args = ap.parse_args()

    # Provide gentle defaults if invoked from old entrypoints lacking new flags
    for k, v in dict(in_scale=DEFAULT_IN_SCALE, clip_max=DEFAULT_CLIP_MAX, pred_batch=DEFAULT_PBATCH,
                     stride=DEFAULT_STRIDE, scale=1.0, min_count=1, min_thresh=0.0, debug=False).items():
        if not hasattr(args, k):
            setattr(args, k, v)

    chrN1, chrN2  = args.chrN
    binsize       = args.binsize
    inmodel       = args.model
    hicfile       = args.inputfile
    outname       = args.outputfile

    Mat = chr_pred(hicfile, chrN1, chrN2, binsize, inmodel, args)
    writeBed(Mat, outname, binsize, chrN1, chrN2,
             scale=args.scale, min_count=args.min_count, min_thresh=args.min_thresh)
    print("Prediction runtime:", datetime.now() - startTime)


if __name__ == "__main__":
    main()
