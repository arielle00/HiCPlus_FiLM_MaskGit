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

def create_offdiag_mask(shape, band, device=None):
    """
    Create a mask for pixels where abs(x-y) > band (off-diagonal region).
    
    Args:
        shape: Tuple of (H, W) or (C, H, W) or (B, C, H, W) - spatial dimensions
        band: Band width - pixels with |i-j| > band are masked
        device: torch device (optional)
    
    Returns:
        Boolean mask tensor of same shape as input (True for off-diagonal pixels)
    """
    # Extract spatial dimensions
    if len(shape) == 2:
        H, W = shape
        spatial_dims = (H, W)
    elif len(shape) == 3:
        _, H, W = shape
        spatial_dims = (H, W)
    elif len(shape) == 4:
        _, _, H, W = shape
        spatial_dims = (H, W)
    else:
        raise ValueError(f"Unsupported shape length: {len(shape)}")
    
    # Create coordinate grids
    i_coords = torch.arange(H, dtype=torch.float32, device=device)
    j_coords = torch.arange(W, dtype=torch.float32, device=device)
    
    # Create meshgrids
    i_grid, j_grid = torch.meshgrid(i_coords, j_coords, indexing='ij')
    
    # Compute |i - j| for each pixel
    diag_dist = torch.abs(i_grid - j_grid)
    
    # Mask: True where |i-j| > band (off-diagonal)
    mask = diag_dist > band
    
    # Expand to match input shape if needed
    if len(shape) == 3:
        mask = mask.unsqueeze(0)  # Add channel dimension
    elif len(shape) == 4:
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return mask

def weighted_mse_loss(pred_log, target_log, loss_weight_mode=None, loss_weight_k=10, lr_tau_counts=0.0, lr_batch=None):
    """
    Compute weighted MSE loss that emphasizes nonzero/high-evidence pixels.
    
    Args:
        pred_log: Predicted values in log1p space [B, C, H, W] or [B, H, W]
        target_log: Target values in log1p space [B, C, H, W] or [B, H, W]
        loss_weight_mode: "hr_nonzero" or "lr_evidence" or None (plain MSE)
        loss_weight_k: Weight multiplier for nonzero/evidence pixels (default: 10)
        lr_tau_counts: Threshold for LR evidence mode (default: 0.0)
        lr_batch: LR input batch for lr_evidence mode [B, C, H_lr, W_lr] (required if mode is lr_evidence)
                  Will be center-cropped to match target spatial dimensions
    
    Returns:
        Weighted MSE loss scalar
    """
    if loss_weight_mode is None:
        # Plain MSE
        return nn.functional.mse_loss(pred_log, target_log)
    
    # Compute squared error
    squared_error = (pred_log - target_log) ** 2
    
    if loss_weight_mode == "hr_nonzero":
        # Variant A: HR-nonzero weighted MSE in log-space
        # Convert HR log1p back to counts: hr_counts = expm1(hr_log)
        hr_counts = torch.expm1(target_log)
        # Weight mask: w = 1 + k * (hr_counts > 0)
        weight = 1.0 + loss_weight_k * (hr_counts > 0).float()
        # Loss: mean(w * (pred_log - hr_log)^2)
        weighted_error = weight * squared_error
        return weighted_error.mean()
    
    elif loss_weight_mode == "lr_evidence":
        # Variant B: LR-evidence weighted loss
        if lr_batch is None:
            raise ValueError("lr_batch is required for lr_evidence loss mode")
        
        # lr_batch may have different spatial dimensions than target (due to padding)
        # Center-crop lr_batch to match target spatial dimensions
        # Use the same padding logic as _build_targets
        padding = conv2d1_filters_size + conv2d2_filters_size + conv2d3_filters_size - 3  # 12
        half_padding = padding // 2  # 6
        
        target_shape = target_log.shape
        lr_shape = lr_batch.shape
        
        # Handle different shape cases
        if len(target_shape) == 4 and len(lr_shape) == 4:
            # Both are 4D: [B, C, H, W]
            if target_shape[2:] != lr_shape[2:]:
                # Center-crop lr_batch to match target
                _, _, H_tgt, W_tgt = target_shape
                lr_batch_cropped = lr_batch[:, :, half_padding:half_padding+H_tgt, half_padding:half_padding+W_tgt]
            else:
                lr_batch_cropped = lr_batch
        elif len(target_shape) == 3:
            # target is 3D [B, H, W], lr is 4D [B, C, H, W]
            H_tgt, W_tgt = target_shape[1], target_shape[2]
            lr_batch_cropped = lr_batch[:, 0, half_padding:half_padding+H_tgt, half_padding:half_padding+W_tgt]
        else:
            # Fallback: try to match shapes
            if lr_shape == target_shape:
                lr_batch_cropped = lr_batch
            else:
                raise ValueError(f"Cannot align lr_batch shape {lr_shape} with target shape {target_shape}")
        
        # Ensure shapes match exactly
        if lr_batch_cropped.shape != target_log.shape:
            raise ValueError(f"After cropping, lr_batch shape {lr_batch_cropped.shape} != target shape {target_log.shape}")
        
        # lr_counts = expm1(lr_log)
        lr_counts = torch.expm1(lr_batch_cropped)
        # w = 1 + k * (lr_counts > tau)
        weight = 1.0 + loss_weight_k * (lr_counts > lr_tau_counts).float()
        # Same weighted MSE
        weighted_error = weight * squared_error
        return weighted_error.mean()
    
    else:
        raise ValueError(f"Unknown loss_weight_mode: {loss_weight_mode}. Use 'hr_nonzero', 'lr_evidence', or None.")

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
    patience=DEFAULT_PATIENCE,
    loss_weight_mode=None,              # "hr_nonzero", "lr_evidence", or None
    loss_weight_k=10,                   # Weight multiplier for nonzero/evidence pixels
    lr_tau_counts=0.0,                  # Threshold for LR evidence mode
    offdiag_band=None,                  # Band width for off-diagonal penalty (None = disabled)
    lambda_bg=0.005                    # Weight for background penalty regularizer
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
    
    # Setup loss function
    if loss_weight_mode is not None:
        print(f"[INFO] Using weighted loss: mode={loss_weight_mode}, k={loss_weight_k}")
        if loss_weight_mode == "lr_evidence":
            print(f"       LR evidence threshold: {lr_tau_counts}")
    else:
        print("[INFO] Using plain MSE loss")
    
    # Setup off-diagonal regularizer
    if offdiag_band is not None:
        print(f"[INFO] Off-diagonal regularizer enabled: band={offdiag_band}, lambda_bg={lambda_bg}")
    else:
        print("[INFO] Off-diagonal regularizer disabled")

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
                    
                    # Use weighted loss if specified, otherwise plain MSE
                    if loss_weight_mode is not None:
                        if loss_weight_mode == "lr_evidence":
                            # Need to pass lr_batch for lr_evidence mode
                            mse_loss = weighted_mse_loss(
                                pred, y_batch,
                                loss_weight_mode=loss_weight_mode,
                                loss_weight_k=loss_weight_k,
                                lr_tau_counts=lr_tau_counts,
                                lr_batch=lr_batch
                            )
                        else:
                            # hr_nonzero mode doesn't need lr_batch
                            mse_loss = weighted_mse_loss(
                                pred, y_batch,
                                loss_weight_mode=loss_weight_mode,
                                loss_weight_k=loss_weight_k,
                                lr_tau_counts=lr_tau_counts
                            )
                    else:
                        # Plain MSE
                        mse_loss = nn.functional.mse_loss(pred, y_batch)
                    
                    # Add off-diagonal regularizer if enabled
                    if offdiag_band is not None:
                        # Build mask for pixels where abs(x-y) > band
                        pred_shape = pred.shape
                        offdiag_mask_base = create_offdiag_mask(pred_shape, offdiag_band, device=pred.device)
                        
                        # Expand mask to match pred shape exactly for boolean indexing
                        if len(pred_shape) == 4:
                            B, C, H, W = pred_shape
                            # Expand [1, 1, H, W] to [B, C, H, W]
                            offdiag_mask = offdiag_mask_base.expand(B, C, H, W)
                        elif len(pred_shape) == 3:
                            B, H, W = pred_shape
                            # Expand [1, H, W] to [B, H, W]
                            offdiag_mask = offdiag_mask_base.expand(B, H, W)
                        else:
                            offdiag_mask = offdiag_mask_base
                        
                        # Convert pred_log to counts: pred_counts = expm1(pred_log)
                        pred_counts = torch.expm1(pred)
                        
                        # Penalize mean predicted counts in off-diagonal region
                        # bg_pen = mean(pred_counts[offdiag_mask])
                        if offdiag_mask.any():
                            bg_pen = pred_counts[offdiag_mask].mean()
                        else:
                            bg_pen = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
                        
                        # Total loss: loss = weighted_mse + lambda_bg * bg_pen
                        loss = mse_loss + lambda_bg * bg_pen
                    else:
                        loss = mse_loss

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
