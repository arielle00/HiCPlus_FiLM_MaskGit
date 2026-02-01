#!/usr/bin/env python3
"""
Example: VQGAN training with improved patch sampling.

This shows how to use FilteredHiCDataset and DiagonalMaskedLoss
in your VQGAN training pipeline.
"""

import os, sys
import torch

# Ensure local muse_maskgit_pytorch is used
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from improved_patch_sampling import FilteredHiCDataset, DiagonalMaskedLoss
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from muse_maskgit_pytorch.trainers import VQGanVAETrainer


# ------------------------------------------------
# Config
# ------------------------------------------------

HIGHRES_NPY = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_hr.npy"
COORDS_NPY = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_hr.npy.coords.npy"
RESULTS_DIR = "./vqgan_results_improved_sampling"

BATCH_SIZE = 4
GRAD_ACCUM = 8
NUM_STEPS = 20_000
IMAGE_SIZE = 128

# Patch sampling parameters
DIAG_MAX_SEP_BINS = 512  # Only train on patches with |i-j| <= 512
ENABLE_JITTER = True     # Add random position jitter
JITTER_MAX = 8           # Maximum jitter in bins

# Loss masking parameters
MIN_DIAG_OFFSET = 2      # Ignore diagonal offsets < 2 bins in loss


# ------------------------------------------------
# Create filtered dataset
# ------------------------------------------------

print("[INFO] Creating filtered dataset...")
dataset = FilteredHiCDataset(
    npy_path=HIGHRES_NPY,
    coords_path=COORDS_NPY if os.path.exists(COORDS_NPY) else None,
    diag_max_sep_bins=DIAG_MAX_SEP_BINS,
    enable_jitter=ENABLE_JITTER,
    jitter_max=JITTER_MAX,
)

print(f"[INFO] Filtered dataset size: {len(dataset)}")


# ------------------------------------------------
# VQGAN model
# ------------------------------------------------

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
)


# ------------------------------------------------
# Custom trainer with masked loss
# ------------------------------------------------

# Note: The VQGanVAETrainer uses its own dataset loading.
# To use FilteredHiCDataset, you would need to modify the trainer
# or create a custom training loop. For now, this shows the concept.

# Option 1: Modify the trainer to accept a custom dataset
# Option 2: Use the filtered dataset in a custom training loop
# Option 3: Pre-filter the .npy file and use standard trainer

print("\n[INFO] To use FilteredHiCDataset with VQGanVAETrainer:")
print("       1. Modify VQGanVAETrainer.__init__ to accept custom dataset")
print("       2. Or create a custom training loop")
print("       3. Or pre-filter patches and save new .npy file")
print("\n[INFO] For now, using standard trainer with pre-filtered data")
print("       (you would need to create filtered .npy first)")


# ------------------------------------------------
# Example: Custom training loop with masked loss
# ------------------------------------------------

def custom_train_step_example(vae, dataset, loss_fn, optimizer, device):
    """
    Example training step using FilteredHiCDataset and DiagonalMaskedLoss.
    """
    vae.train()
    
    # Sample batch
    indices = torch.randint(0, len(dataset), (BATCH_SIZE,))
    batch = torch.stack([dataset[i] for i in indices]).to(device)
    
    # Forward pass
    recon, vq_loss = vae(batch, return_loss=True, return_recons=True)
    
    # Compute masked reconstruction loss
    recon_loss = loss_fn(recon, batch)
    
    # Total loss
    total_loss = recon_loss + vq_loss
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item(), recon_loss.item(), vq_loss.item()


# Create masked loss function
masked_loss_fn = DiagonalMaskedLoss(min_diag_offset=MIN_DIAG_OFFSET)

print(f"\n[INFO] Diagonal masked loss configured:")
print(f"       - Min diagonal offset: {MIN_DIAG_OFFSET} bins")
print(f"       - Off-diagonal interactions will be emphasized in training")


# ------------------------------------------------
# Summary
# ------------------------------------------------

print("\n" + "="*60)
print("IMPROVED PATCH SAMPLING CONFIGURATION")
print("="*60)
print(f"Dataset filtering:")
print(f"  - Max diagonal separation: {DIAG_MAX_SEP_BINS} bins")
print(f"  - Position jitter: {'enabled' if ENABLE_JITTER else 'disabled'}")
if ENABLE_JITTER:
    print(f"  - Jitter max: {JITTER_MAX} bins")
print(f"\nLoss masking:")
print(f"  - Min diagonal offset: {MIN_DIAG_OFFSET} bins")
print(f"  - Diagonal offsets < {MIN_DIAG_OFFSET} will be ignored in loss")
print("="*60)

print("\n[INFO] To integrate into training:")
print("       1. Use FilteredHiCDataset instead of HiCDataset")
print("       2. Use DiagonalMaskedLoss for reconstruction loss")
print("       3. Modify trainer or create custom training loop")
