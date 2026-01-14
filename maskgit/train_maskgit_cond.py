import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["BEARTYPE_IS_BEARTYPE"] = "0"

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit, Transformer
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HR_PATH = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_hr.npy"
LR_PATH = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_lr.npy"

# -------------------------
# Dataset: HR+LR paired from disk (no on-the-fly thinning)
# -------------------------
class HiCCondPairedDataset(Dataset):
    def __init__(self, hr_path: str, lr_path: str, dtype=torch.float32):
        hr = np.load(hr_path)  # expected [N,1,H,W]
        lr = np.load(lr_path)  # expected [N,1,H,W]

        if hr.shape != lr.shape:
            raise ValueError(f"HR and LR shapes must match. HR={hr.shape}, LR={lr.shape}")
        if hr.ndim != 4 or hr.shape[1] != 1:
            raise ValueError(f"Expected shape [N,1,H,W]. Got {hr.shape}")

        self.hr = torch.from_numpy(hr).to(dtype)
        self.lr = torch.from_numpy(lr).to(dtype)

    def __len__(self):
        return self.hr.shape[0]

    def __getitem__(self, idx):
        return self.hr[idx], self.lr[idx]


dataset = HiCCondPairedDataset(HR_PATH, LR_PATH)

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

vae.load("/home/012002744/hicplus_thesis/maskgit/muse_vqgan/vqgan_results_2layers_4096/vae.best.pt")
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# -------------------------
# Infer seq_len from VAE (DON'T hardcode 1024)
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
# Transformer (conditioning handled inside your modified MaskGit forward)
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
    pbar = tqdm(train_loader, desc=f"[MaskGIT-Cond] Epoch {epoch}")
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
                torch.save(maskgit.state_dict(), "maskgit_cond.best.pt")

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

torch.save(maskgit.state_dict(), "maskgit_cond_new.pt")

print("\n" + "=" * 80)
print("TRAINING SUMMARY - Best Model:")
print("=" * 80)
if best_val_loss < float("inf"):
    print(f"Best Model: Step {last_improvement_step} with Validation Loss: {best_val_loss:.6f}")
    print("Best model saved as: maskgit_cond.best.pt")
else:
    print("No validation evaluations performed")
print("Final model saved as: maskgit_cond.pt")
print("=" * 80)
