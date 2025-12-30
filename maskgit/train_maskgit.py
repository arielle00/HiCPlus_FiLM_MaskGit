#!/usr/bin/env python3
"""
MaskGIT training for Hi-C.

Trains conditional MaskGIT:
    p(tokens_high | tokens_low)

Assumes:
- pretrained LOW-res VQGAN (frozen)
- pretrained HIGH-res VQGAN (frozen)
- both trained with SAME vqgan_vae.py
- dataset returns (low_res_tile, high_res_tile)
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

# ---------------------
# Imports
# ---------------------


from muse_maskgit_pytorch.vqgan_vae import VQGanVAE

from muse_maskgit_pytorch.maskgit import MaskGit
from muse_maskgit_pytorch.maskgit_transformer import MaskGitTransformer

from datasets.hic_dataset import HiCDataset   # must return (lo, hi)


# =====================
# Config
# =====================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOW_SIZE  = 40
HIGH_SIZE = 80

BATCH_SIZE = 16
EPOCHS = 80
LR = 2e-4

VQGAN_LOW_CKPT  = "checkpoints/vqgan_low.pt"
VQGAN_HIGH_CKPT = "checkpoints/vqgan_high.pt"

OUT_DIR = Path("checkpoints")
OUT_DIR.mkdir(exist_ok=True)

OUT_MASKGIT = OUT_DIR / "maskgit_conditional.pt"


# =====================
# Dataset
# =====================

dataset = HiCDataset(
    low_size=LOW_SIZE,
    high_size=HIGH_SIZE
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    drop_last=True,
    pin_memory=True
)

print(f"[INFO] Dataset size: {len(dataset)}")
print(f"[INFO] Batches / epoch: {len(loader)}")


# =====================
# Load + freeze VQGANs
# =====================

vae_low = VQGanVAE(
    dim=128,
    channels=1,
    codebook_size=512,
    commitment_cost=0.25,
    use_vgg_and_gan=False,
).to(DEVICE)

vae_high = VQGanVAE(
    dim=128,
    channels=1,
    codebook_size=512,
    commitment_cost=0.25,
    use_vgg_and_gan=False,
).to(DEVICE)

vae_low.load_state_dict(torch.load(VQGAN_LOW_CKPT, map_location=DEVICE))
vae_high.load_state_dict(torch.load(VQGAN_HIGH_CKPT, map_location=DEVICE))

vae_low.eval()
vae_high.eval()

for p in vae_low.parameters():
    p.requires_grad = False
for p in vae_high.parameters():
    p.requires_grad = False


# =====================
# Sanity checks (CRITICAL)
# =====================

with torch.no_grad():
    dummy_lo = torch.zeros(1, 1, LOW_SIZE, LOW_SIZE).to(DEVICE)
    dummy_hi = torch.zeros(1, 1, HIGH_SIZE, HIGH_SIZE).to(DEVICE)

    _, ids_lo, _ = vae_low.encode(dummy_lo)
    _, ids_hi, _ = vae_high.encode(dummy_hi)

    # ids are flat [B * H * W]
    assert ids_lo.ndim == 1
    assert ids_hi.ndim == 1

    f_lo = int((ids_lo.numel()) ** 0.5)
    f_hi = int((ids_hi.numel()) ** 0.5)

    print(f"[Sanity] LOW token grid:  {f_lo} x {f_lo}")
    print(f"[Sanity] HIGH token grid: {f_hi} x {f_hi}")


# =====================
# MaskGIT Transformer
# =====================

transformer = MaskGitTransformer(
    dim=512,
    depth=8,
    heads=8,
    dim_head=64,
    dropout=0.1,
)

maskgit = MaskGit(
    transformer=transformer,
    vae=vae_high,          # predicts HIGH tokens
    cond_vae=vae_low,      # conditioned on LOW tokens
    image_size=HIGH_SIZE,
    channels=1,
    cond_image_size=LOW_SIZE,
    mask_prob=0.6,         # stable starting value
    steps=8,
).to(DEVICE)

optimizer = torch.optim.AdamW(
    maskgit.parameters(),
    lr=LR,
    betas=(0.9, 0.99),
    weight_decay=1e-4,
)


# =====================
# Training loop
# =====================

maskgit.train()

for epoch in range(EPOCHS):
    pbar = tqdm(loader, desc=f"[MaskGIT] epoch {epoch}")

    for lo, hi in pbar:
        lo = lo.to(DEVICE, non_blocking=True)
        hi = hi.to(DEVICE, non_blocking=True)

        loss = maskgit(
            hi,
            cond_images=lo,
            return_loss=True
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    print(f"[epoch {epoch}] loss: {loss.item():.4f}")


# =====================
# Save checkpoint
# =====================

torch.save(maskgit.state_dict(), OUT_MASKGIT)
print(f"[DONE] Saved MaskGIT → {OUT_MASKGIT}")
