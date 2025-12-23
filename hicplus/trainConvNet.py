# hicplus/trainConvNet.py

from __future__ import print_function
import os
from time import gmtime, strftime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data

try:
    from torch.cuda.amp import autocast, GradScaler
    _AMP_AVAILABLE = True
except Exception:
    _AMP_AVAILABLE = False

# Import your models (model.py must define Net; FiLMNet/build_model are optional)
from hicplus import model as model_mod

# -------------------------
# Defaults / constants
# -------------------------
use_gpu = 1
down_sample_ratio = 16
HiC_max_value = 100

# original HiCPlus valid-conv shrink for 40->28 (9,1,5 kernels)
conv2d1_filters_size = 9
conv2d2_filters_size = 1
conv2d3_filters_size = 5

# training defaults (can be overridden via kwargs)
DEFAULT_EPOCHS = 3500         # keep original loop length; early-stop saves time
DEFAULT_BATCH = 64            # safer on 12GB GPUs; increase if you can
DEFAULT_LR = 0.0001             # AdamW works well in log1p space
DEFAULT_WD = 1e-4
DEFAULT_ACCUM = 1             # grad accumulation steps
DEFAULT_LOG_EVERY = 50
DEFAULT_PATIENCE = 15     # early-stop patience (epochs w/o improvement)
DEFAULT_SPACE = "log1p"       # switch to "counts" to mimic old behavior
DEFAULT_OPT = "adamw"         # or "sgd" to match legacy
MIN_DELTA = 1e-4

# -------------------------
# Tiny helpers
# -------------------------
def _device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but not available. Exiting.")
    return torch.device("cuda")

def _build_targets(high_resolution_samples, sample_size):
    """Center-crop HR to match the vanilla HiCPlus output size."""
    padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3  # 12
    half_padding = padding // 2
    output_length = sample_size - padding
    Y = []
    # high_resolution_samples: (N, 1, H, W)
    for i in range(high_resolution_samples.shape[0]):
        no_padding_sample = high_resolution_samples[i][0][
            half_padding:(sample_size - half_padding),
            half_padding:(sample_size - half_padding)
        ]
        Y.append(no_padding_sample)
    Y = np.array(Y, dtype=np.float32)  # (N, output_length, output_length)
    return Y, output_length

def _make_loader(low_np, y_np, batch_size):
    # Ensure (N,1,H,W)
    if low_np.ndim == 3:
        low_np = low_np[:, None, :, :]
    if y_np.ndim == 3:
        y_np = y_np[:, None, :, :]

    ds = data.TensorDataset(torch.from_numpy(low_np), torch.from_numpy(y_np))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    return dl

def _maybe_amp(enabled):
    return _AMP_AVAILABLE and enabled

# -------------------------
# Main training entry
# -------------------------
def train(
    lowres, highres, outModel,
    arch="hicplus",
    width=64, depth=8, use_aux=True,
    space=DEFAULT_SPACE,                 # "log1p" (recommended) or "counts"
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH,
    lr=DEFAULT_LR, weight_decay=DEFAULT_WD,
    optimizer_name=DEFAULT_OPT,          # "adamw" or "sgd"
    amp=True, accum_steps=DEFAULT_ACCUM,
    log_every=DEFAULT_LOG_EVERY,
    patience=DEFAULT_PATIENCE
):
    """
    Train either vanilla HiCPlus or FiLM model.

    Args:
        lowres, highres: np.ndarray (N,1,H,W) in counts (0..HiC_max_value after clipping)
        outModel: path to save best checkpoint
        arch: "hicplus" or "film"
        space: "log1p" (train in log1p space) or "counts"
        width, depth, use_aux: FiLM-only knobs (ignored by hicplus)
    """
    device = _device()

    # ---- Prep data (clip -> (optional) log1p) ----
    low = lowres.astype(np.float32) * down_sample_ratio
    hr  = highres.astype(np.float32)

    # clip to stabilize
    low = np.minimum(HiC_max_value, low)
    hr  = np.minimum(HiC_max_value,  hr)

    sample_size = int(low.shape[-1])  # D_in
    Y_np, D_out = _build_targets(hr, sample_size)  # center-cropped HR -> (N, D_out, D_out)

    # choose training space
    use_log1p = (space.lower() == "log1p")
    if use_log1p:
        low  = np.log1p(low)
        Y_np = np.log1p(Y_np)

    print("Data shapes (low, Y):", low.shape, (Y_np.shape[0], 1, Y_np.shape[1], Y_np.shape[2]))
    print("Space:", "log1p" if use_log1p else "counts")
    print("Ranges:",
          "LR[%.3f, %.3f]" % (float(low.min()), float(low.max())),
          "Y[%.3f, %.3f]" % (float(Y_np.min()), float(Y_np.max())))

    # ---- DataLoader ----
    train_loader = _make_loader(low, Y_np, batch_size=batch_size)

    # ---- Build model ----
    D_in = sample_size
    nonneg = (not use_log1p)  # in log space, allow negatives (no Softplus)
    if hasattr(model_mod, "build_model"):
        Net = model_mod.build_model(
            arch=arch, D_in=D_in, D_out=D_out,
            width=width, depth=depth, use_aux=use_aux, nonneg=nonneg
        )
    else:
        if arch == "hicplus":
            Net = model_mod.Net(D_in, D_out)
        elif hasattr(model_mod, "FiLMNet") and arch == "film":
            Net = model_mod.FiLMNet(D_in, D_out, width=width, depth=depth, use_aux=use_aux, nonneg=nonneg)
        else:
            raise ValueError("Unknown arch '%s'. Use 'hicplus' or 'film'." % arch)

    if use_gpu:
        Net = Net.to(device)

    # ---- Optimizer / scaler ----
    if optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(Net.parameters(), lr=(lr if lr is not None else 1e-5), momentum=0.9, weight_decay=weight_decay)
        # Reduce LR when the average epoch loss plateaus

    else:
        # default AdamW for faster convergence
        optimizer = optim.AdamW(Net.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99))

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=2,                 # drop LR sooner
        threshold=5e-4,        # ignore tiny “improvements”
        threshold_mode='abs',
        min_lr=1e-6,
        verbose=True,   
    )


    scaler = GradScaler(enabled=_maybe_amp(amp))
    criterion = nn.MSELoss()

    # ---- Train loop ----
    best_loss = float("inf")
    best_epoch = -1
    no_improve = 0

    os.makedirs(os.path.dirname(outModel) or ".", exist_ok=True)

    with open('HindIII_train.txt', 'w') as logf:
        for epoch in range(epochs):
            Net.train()
            running_loss = 0.0
            batches = 0

            for step, (lr_batch, y_batch) in enumerate(train_loader, start=1):
                lr_batch = lr_batch.to(device, non_blocking=True)
                y_batch  = y_batch.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=_maybe_amp(amp)):
                    pred = Net(lr_batch)            # in log or counts space, depending on use_log1p
                    loss = criterion(pred, y_batch)

                scaler.scale(loss / max(1, accum_steps)).backward()

                if (step % accum_steps) == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += float(loss.item())
                batches += 1

                if log_every and (step % log_every == 0):
                    # Optional: show counts-space MSE as a sanity metric when training in log1p
                    if use_log1p:
                        with torch.no_grad():
                            pred_c = torch.expm1(pred).clamp_min(0)
                            targ_c = torch.expm1(y_batch).clamp_min(0)
                            mse_counts = nn.functional.mse_loss(pred_c, targ_c).item()
                        print("[ep %d] it %d/%d  loss=%.4f (counts-mse=%.2f)  %s"
                              % (epoch, step, len(train_loader), float(loss.item()), mse_counts,
                                 strftime("%Y-%m-%d %H:%M:%S", gmtime())))
                    else:
                        print("[ep %d] it %d/%d  loss=%.4f  %s"
                              % (epoch, step, len(train_loader), float(loss.item()),
                                 strftime("%Y-%m-%d %H:%M:%S", gmtime())))

            avg_loss = running_loss / max(1, batches)


            # end-of-epoch log
            stamp = strftime("%Y-%m-%d %H:%M:%S", gmtime())
            print("------- epoch %d  avg_loss=%.4f  (%s)" % (epoch, avg_loss, stamp))
            logf.write("{}, {}, {}\n".format(epoch, avg_loss, stamp))
            logf.flush()

            # tell the scheduler our validation metric (here: avg training loss)
            scheduler.step(avg_loss)

            # save best
            if (best_loss - avg_loss) > MIN_DELTA:
                best_loss = avg_loss
                best_epoch = epoch
                ckpt = {
                    "state_dict": Net.state_dict(),
                    "arch": arch,
                    "space": space,
                    "D_in": D_in,
                    "D_out": D_out,
                    "film": {"width": width, "depth": depth, "use_aux": use_aux, "nonneg": nonneg}
                }
                torch.save(ckpt, outModel)
                print("✅ Saved best model at epoch %d with avg_loss %.4f -> %s" % (epoch, avg_loss, outModel))
                no_improve = 0
            else:
                no_improve += 1
                # optional early stopping
                if patience and no_improve >= patience:
                    print("Early stopping (no improvement for %d epochs). Best @ epoch %d (%.4f)."
                          % (no_improve, best_epoch, best_loss))
                    break

    print("Training finished. Best epoch %d with loss %.4f" % (best_epoch, best_loss))
