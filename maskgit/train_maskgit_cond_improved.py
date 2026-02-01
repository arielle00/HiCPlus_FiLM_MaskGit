#!/usr/bin/env python3
"""
Conditional MaskGit training with improved patch sampling.

Features:
- Diagonal filtering for HR and LR datasets
- Position jitter (data augmentation)

Usage:
    cd /home/012002744/hicplus_thesis/maskgit
    python3 train_maskgit_cond_improved.py
"""

import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["BEARTYPE_IS_BEARTYPE"] = "0"

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from pathlib import Path
import sys

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from improved_patch_sampling import FilteredHiCDataset
from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit, Transformer
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HR_PATH = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_hr.npy"
LR_PATH = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_lr.npy"
HR_COORDS = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_hr.npy.coords.npy"
LR_COORDS = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_lr.npy.coords.npy"

# Improved sampling parameters
DIAG_MAX_SEP_BINS = 512  # Only train on patches with |i-j| <= 512
ENABLE_JITTER = True     # Add random position jitter
JITTER_MAX = 8           # Maximum jitter in bins

# -------------------------
# Dataset: HR+LR paired with improved sampling
# -------------------------
class HiCCondPairedDatasetImproved(Dataset):
    """
    Paired HR+LR dataset using FilteredHiCDataset for both.
    Ensures both HR and LR are filtered to the same patches.
    """
    def __init__(
        self,
        hr_path: str,
        lr_path: str,
        hr_coords: str = None,
        lr_coords: str = None,
        diag_max_sep_bins: int = None,
        enable_jitter: bool = False,
        jitter_max: int = 8,
    ):
        # Create filtered datasets
        self.hr_dataset = FilteredHiCDataset(
            npy_path=hr_path,
            coords_path=hr_coords,
            diag_max_sep_bins=diag_max_sep_bins,
            enable_jitter=enable_jitter,
            jitter_max=jitter_max,
        )
        
        self.lr_dataset = FilteredHiCDataset(
            npy_path=lr_path,
            coords_path=lr_coords,
            diag_max_sep_bins=diag_max_sep_bins,
            enable_jitter=enable_jitter,
            jitter_max=jitter_max,
        )
        
        # Ensure same length
        if len(self.hr_dataset) != len(self.lr_dataset):
            min_len = min(len(self.hr_dataset), len(self.lr_dataset))
            print(f"[WARN] HR and LR datasets have different lengths!")
            print(f"       HR: {len(self.hr_dataset)}, LR: {len(self.lr_dataset)}")
            print(f"       Using first {min_len} samples")
            self.length = min_len
        else:
            self.length = len(self.hr_dataset)
        
        # Ensure same valid indices (if coordinates are available)
        if hasattr(self.hr_dataset, 'valid_indices') and hasattr(self.lr_dataset, 'valid_indices'):
            # Find intersection of valid indices
            hr_valid = set(self.hr_dataset.valid_indices)
            lr_valid = set(self.lr_dataset.valid_indices)
            common_valid = sorted(list(hr_valid & lr_valid))
            
            if len(common_valid) < self.length:
                print(f"[INFO] Using {len(common_valid)} common valid patches")
                # Create mapping from common indices to dataset indices
                hr_idx_map = {idx: i for i, idx in enumerate(self.hr_dataset.valid_indices)}
                lr_idx_map = {idx: i for i, idx in enumerate(self.lr_dataset.valid_indices)}
                
                self.common_indices = common_valid
                self.hr_idx_map = hr_idx_map
                self.lr_idx_map = lr_idx_map
                self.length = len(common_valid)
            else:
                self.common_indices = None
        else:
            self.common_indices = None

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.common_indices is not None:
            # Use common indices
            actual_idx = self.common_indices[idx]
            hr_idx = self.hr_idx_map[actual_idx]
            lr_idx = self.lr_idx_map[actual_idx]
            hr = self.hr_dataset[hr_idx]
            lr = self.lr_dataset[lr_idx]
        else:
            # Use same index for both (assumes aligned)
            hr = self.hr_dataset[idx]
            lr = self.lr_dataset[idx]
        
        return hr, lr


print("[INFO] Creating filtered paired dataset...")
print(f"       - Max diagonal separation: {DIAG_MAX_SEP_BINS} bins")
print(f"       - Position jitter: {'enabled' if ENABLE_JITTER else 'disabled'}")

dataset = HiCCondPairedDatasetImproved(
    hr_path=HR_PATH,
    lr_path=LR_PATH,
    hr_coords=HR_COORDS if Path(HR_COORDS).exists() else None,
    lr_coords=LR_COORDS if Path(LR_COORDS).exists() else None,
    diag_max_sep_bins=DIAG_MAX_SEP_BINS,
    enable_jitter=ENABLE_JITTER,
    jitter_max=JITTER_MAX,
)

# Split into train / val with aligned indices
N = len(dataset)
train_size = int(0.95 * N)
val_size = N - train_size

g = torch.Generator().manual_seed(42)
perm = torch.randperm(N, generator=g).tolist()

train_idx = perm[:train_size]
val_idx   = perm[train_size:]

train_data = Subset(dataset, train_idx)
val_data   = Subset(dataset, val_idx)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False, drop_last=False)

print(f"[INFO] Total samples: {N}")
print(f"[INFO] Training on {train_size} samples, validating on {val_size} samples")
print(f"[INFO] HR file: {HR_PATH}")
print(f"[INFO] LR file: {LR_PATH}")

# -------------------------
# Load frozen VQGAN (MUST match training)
# -------------------------
vae = VQGanVAE(
    dim=256,
    channels=1,
    layers=2,
    codebook_size=4096,
    lookup_free_quantization=False,
    use_vgg_and_gan=False,
    vq_kwargs=dict(
        codebook_dim=64,
        decay=0.95,
        commitment_weight=1.0,
        kmeans_init=True,
        use_cosine_sim=True,
    ),
).to(DEVICE)

vae.load("/home/012002744/hicplus_thesis/maskgit/muse_vqgan/vqgan_results_2layers_4096_removedOEsignal/vae.best.pt")
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# -------------------------
# Infer seq_len from VAE
# -------------------------
with torch.no_grad():
    hr0, _ = dataset[0]
    hr0 = hr0.unsqueeze(0).to(DEVICE)  # [1,1,H,W]
    _, ids0, _ = vae.encode(hr0)
    token_h, token_w = ids0.shape[1:]
    seq_len = token_h * token_w

print(f"[INFO] Token grid: {token_h} x {token_w} => seq_len={seq_len}")
print(f"[INFO] codebook_size={vae.codebook_size}")

# -------------------------
# Transformer
# -------------------------
transformer = Transformer(
    num_tokens=vae.codebook_size,
    seq_len=seq_len,
    dim=512,
    depth=8,
    heads=8,
    dim_head=64,
).to(DEVICE)

# -------------------------
# MaskGit (CONDITIONAL)
# -------------------------
maskgit = MaskGit(
    vae=vae,
    image_size=128,
    no_mask_token_prob=0.6,
    transformer=transformer
).to(DEVICE)

optimizer = torch.optim.AdamW(maskgit.parameters(), lr=2e-4)

# -------------------------
# Training loop with early stopping
# -------------------------
maskgit.train()

early_stopping_patience = 1000
best_val_loss = float("inf")
last_improvement_step = 0
step = 0
eval_every = 100

print(f"Early stopping patience: {early_stopping_patience} steps")
stop = False

for epoch in range(50):
    pbar = tqdm(train_loader, desc=f"[MaskGIT-Cond-Improved] Epoch {epoch}")
    for hr, lr in pbar:
        hr = hr.to(DEVICE)
        lr = lr.to(DEVICE)

        loss = maskgit(hr, lr_images=lr)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        if step % eval_every == 0:
            maskgit.eval()
            val_losses = []
            with torch.no_grad():
                for vhr, vlr in val_loader:
                    vhr = vhr.to(DEVICE)
                    vlr = vlr.to(DEVICE)
                    vloss = maskgit(vhr, lr_images=vlr)
                    val_losses.append(vloss.item())

            avg_val_loss = float(sum(val_losses) / max(1, len(val_losses)))
            maskgit.train()

            if avg_val_loss < best_val_loss:
                prev_best = best_val_loss
                prev_best_step = last_improvement_step
                best_val_loss = avg_val_loss
                last_improvement_step = step
                torch.save(maskgit.state_dict(), "maskgit_cond_improved.best.pt")

                if prev_best < float("inf"):
                    print(f"\n[Step {step}] ⭐ NEW BEST ⭐ Val Loss: {avg_val_loss:.6f} "
                          f"(Prev: {prev_best:.6f} at step {prev_best_step})")
                else:
                    print(f"\n[Step {step}] ⭐ FIRST BEST ⭐ Val Loss: {avg_val_loss:.6f}")
            else:
                print(f"\n[Step {step}] Val Loss: {avg_val_loss:.6f} "
                      f"(Best: {best_val_loss:.6f} at step {last_improvement_step})")

                if last_improvement_step > 0 and (step - last_improvement_step) >= early_stopping_patience:
                    print(f"\n[Step {step}] Early stopping: no improvement for "
                          f"{step - last_improvement_step} steps.")
                    stop = True

            pbar.set_postfix(
                train_loss=f"{loss.item():.4f}",
                val_loss=f"{avg_val_loss:.4f}",
                best_val_loss=f"{best_val_loss:.4f}"
            )

        if stop:
            break

    if stop:
        break

torch.save(maskgit.state_dict(), "maskgit_cond_improved.pt")

print("\n" + "=" * 80)
print("TRAINING SUMMARY - Best Model (Improved Sampling):")
print("=" * 80)
if best_val_loss < float("inf"):
    print(f"Best Model: Step {last_improvement_step} with Validation Loss: {best_val_loss:.6f}")
    print("Best model saved as: maskgit_cond_improved.best.pt")
else:
    print("No validation evaluations performed")
print("Final model saved as: maskgit_cond_improved.pt")
print("=" * 80)
