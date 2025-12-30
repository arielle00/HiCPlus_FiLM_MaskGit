import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["BEARTYPE_IS_BEARTYPE"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm

from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit, Transformer
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# LR corruption (define what "low-res" means)
# -------------------------
def make_lr(hr: torch.Tensor, down: int = 2) -> torch.Tensor:
    """
    hr: [1,H,W] or [B,1,H,W]
    returns: LR-like degraded version at same shape
    """
    if hr.dim() == 3:
        hr = hr.unsqueeze(0)

    # downsample -> upsample (removes high-frequency detail)
    lr = F.avg_pool2d(hr, kernel_size=down, stride=down)
    lr = F.interpolate(lr, size=hr.shape[-2:], mode="bilinear", align_corners=False)

    return lr.squeeze(0) if lr.shape[0] == 1 else lr

class HiCCondDataset(Dataset):
    def __init__(self, hr_tensor: torch.Tensor, down: int = 2):
        self.hr = hr_tensor
        self.down = down

    def __len__(self):
        return self.hr.shape[0]

    def __getitem__(self, idx):
        hr = self.hr[idx]          # [1,128,128]
        lr = make_lr(hr, self.down)
        return hr, lr

# -------------------------
# Load frozen VQGAN
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
# Dataset (HR only on disk; LR generated on-the-fly)
# -------------------------
data = np.load("hic_vqgan_train.npy")   # [N, 1, 128, 128]
data = torch.from_numpy(data).float()

dataset = HiCCondDataset(data, down=2)  # tune down=2/4 depending on how "low-res" you want

# Split into train and validation (5% validation)
train_size = int(0.95 * len(dataset))
val_size = len(dataset) - train_size
train_data, val_data = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
val_loader   = DataLoader(val_data,   batch_size=16, shuffle=False, drop_last=False)

# -------------------------
# Infer seq_len from VAE (DON'T hardcode 1024)
# -------------------------
with torch.no_grad():
    hr0, lr0 = dataset[0]
    hr0 = hr0.unsqueeze(0).to(DEVICE)  # [1,1,128,128]
    _, ids0, _ = vae.encode(hr0)
    token_h, token_w = ids0.shape[1:]
    seq_len = token_h * token_w

print(f"[INFO] Token grid: {token_h} x {token_w}  => seq_len={seq_len}")
print(f"[INFO] codebook_size={vae.codebook_size}")
print(f"[INFO] Training on {train_size} samples, validating on {val_size} samples")

# -------------------------
# Transformer (same class, now supports conditioning because you modified base code)
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

        # IMPORTANT: conditional call
        # If your modified MaskGit signature is forward(self, hr_images, lr_images=None),
        # this should work:
        loss = maskgit(hr, lr_images=lr)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Evaluate validation loss periodically
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

                if last_improvement_step > 0:
                    if (step - last_improvement_step) >= early_stopping_patience:
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

torch.save(maskgit.state_dict(), "maskgit_cond.pt")

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
