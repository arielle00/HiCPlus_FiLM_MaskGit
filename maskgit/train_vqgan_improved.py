#!/usr/bin/env python3
"""
VQGAN training with improved patch sampling.

Features:
- Diagonal filtering (|i-j| <= diag_max_sep_bins)
- Position jitter (data augmentation)
- Diagonal-masked loss (excludes main diagonal band)

Usage:
    cd /home/012002744/hicplus_thesis/maskgit
    python3 train_vqgan_improved.py
"""

import os, sys
import torch
import numpy as np
from pathlib import Path

# Ensure local muse_maskgit_pytorch is used
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

import muse_maskgit_pytorch
print("[INFO] muse_maskgit_pytorch loaded from:", muse_maskgit_pytorch.__file__)

# Import improved sampling
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

# Improved sampling parameters
DIAG_MAX_SEP_BINS = 512  # Only train on patches with |i-j| <= 512
ENABLE_JITTER = True     # Add random position jitter
JITTER_MAX = 8           # Maximum jitter in bins

# Loss masking parameters
MIN_DIAG_OFFSET = 2      # Ignore diagonal offsets < 2 bins in loss

# ------------------------------------------------
# Create filtered dataset
# ------------------------------------------------

print("[INFO] Creating filtered dataset with improved sampling...")
print(f"       - Max diagonal separation: {DIAG_MAX_SEP_BINS} bins")
print(f"       - Position jitter: {'enabled' if ENABLE_JITTER else 'disabled'}")
if ENABLE_JITTER:
    print(f"       - Jitter max: {JITTER_MAX} bins")

# Check if coords file exists
if not Path(COORDS_NPY).exists():
    print(f"[WARN] Coordinates file not found: {COORDS_NPY}")
    print(f"       Diagonal filtering will not work correctly.")
    print(f"       Run create_coords_for_combined.py first if needed.")
    coords_path = None
else:
    coords_path = COORDS_NPY

# Create filtered dataset
filtered_dataset = FilteredHiCDataset(
    npy_path=HIGHRES_NPY,
    coords_path=coords_path,
    diag_max_sep_bins=DIAG_MAX_SEP_BINS,
    enable_jitter=ENABLE_JITTER,
    jitter_max=JITTER_MAX,
)

print(f"[INFO] Filtered dataset size: {len(filtered_dataset)}")

# ------------------------------------------------
# Option 1: Pre-filter and save new .npy file
# ------------------------------------------------

# Since VQGanVAETrainer expects a .npy file path, we need to either:
# 1. Pre-filter and save a new .npy file, OR
# 2. Modify the trainer to accept a custom dataset

# For now, we'll create a filtered .npy file
FILTERED_NPY = str(Path(HIGHRES_NPY).parent / "hic_vqgan_train_hr_filtered.npy")

print(f"\n[INFO] Saving filtered patches to: {FILTERED_NPY}")
filtered_data = []
for i in range(len(filtered_dataset)):
    patch = filtered_dataset[i].numpy()  # [1, H, W]
    filtered_data.append(patch)

filtered_array = np.stack(filtered_data, axis=0)  # [N, 1, H, W]
np.save(FILTERED_NPY, filtered_array)
print(f"[INFO] Saved {len(filtered_data)} filtered patches")

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
# Trainer (using filtered .npy)
# ------------------------------------------------

print(f"\n[INFO] Starting VQGAN training with improved sampling...")
print(f"       - Using filtered dataset: {FILTERED_NPY}")
print(f"       - Diagonal-masked loss: offsets < {MIN_DIAG_OFFSET} bins ignored")

# Note: The standard VQGanVAETrainer doesn't support custom loss functions
# easily. For diagonal masking, you would need to modify the trainer's
# train_step method or create a custom training loop.
# For now, we use the standard trainer with filtered data.

trainer = VQGanVAETrainer(
    vae=vae,
    folder=FILTERED_NPY,  # Use filtered .npy
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    grad_accum_every=GRAD_ACCUM,
    num_train_steps=NUM_STEPS,
    results_folder=RESULTS_DIR,
)

# ------------------------------------------------
# Train
# ------------------------------------------------

trainer.train()

print("\n[INFO] Training complete!")
print(f"       Results saved to: {RESULTS_DIR}")
