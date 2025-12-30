import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["BEARTYPE_IS_BEARTYPE"] = "0"

import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit, Transformer
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load frozen VQGAN
# -------------------------
vae = VQGanVAE(
    dim=256,
    channels=1,
    layers=2,
    codebook_size=4096,
    lookup_free_quantization=False,   # <-- big change
    use_vgg_and_gan=False,
    vq_kwargs=dict(
        codebook_dim=64,              # try 64 or 128
        decay=0.95,
        commitment_weight=1.0,
        kmeans_init=True,
        use_cosine_sim=True,
    ),
).to(DEVICE)

vae.load(
    "/home/012002744/hicplus_thesis/maskgit/muse_vqgan/vqgan_results_2layers_4096/vae.best.pt"
)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# -------------------------
# Dataset
# -------------------------
data = np.load("hic_vqgan_train.npy")   # [N, 1, 128, 128]
data = torch.from_numpy(data).float()

# Split into train and validation (5% validation)
train_size = int(0.95 * len(data))
val_size = len(data) - train_size
train_data, val_data = random_split(data, [train_size, val_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(
    train_data,
    batch_size=16,
    shuffle=True,
    drop_last=True
)

val_loader = DataLoader(
    val_data,
    batch_size=16,
    shuffle=False,
    drop_last=False
)



transformer = Transformer(
    num_tokens = vae.codebook_size,   # e.g. 4096
    seq_len    = 1024,              # e.g. 32 * 32 = 1024
    dim        = 512,
    depth      = 8,
    heads      = 8,
    dim_head   = 64,
).to(DEVICE)

# -------------------------
# MaskGit (UNCONDITIONAL)
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

# Early stopping parameters
early_stopping_patience = 1000  # Stop if no improvement for 1000 iterations
best_val_loss = float('inf')
last_improvement_step = 0
step = 0
eval_every = 100  # Evaluate validation loss every 100 iterations

print(f"Training on {train_size} samples, validating on {val_size} samples")
print(f"Early stopping patience: {early_stopping_patience} steps")

for epoch in range(50):
    pbar = tqdm(train_loader, desc=f"[MaskGIT] Epoch {epoch}")
    for x in pbar:
        x = x.to(DEVICE)

        loss = maskgit(x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Evaluate validation loss periodically
        if step % eval_every == 0:
            maskgit.eval()
            val_losses = []
            with torch.no_grad():
                for val_x in val_loader:
                    val_x = val_x.to(DEVICE)
                    val_loss = maskgit(val_x)
                    val_losses.append(val_loss.item())
            
            avg_val_loss = sum(val_losses) / len(val_losses)
            maskgit.train()
            
            # Check for improvement
            if avg_val_loss < best_val_loss:
                prev_best = best_val_loss
                prev_best_step = last_improvement_step
                best_val_loss = avg_val_loss
                last_improvement_step = step
                
                # Save best model
                torch.save(maskgit.state_dict(), "maskgit_uncond.best.pt")
                
                if prev_best < float('inf'):
                    print(f"\n[Step {step}] ⭐ NEW BEST MODEL ⭐ - Val Loss: {avg_val_loss:.6f} "
                          f"(Previous best: {prev_best:.6f} at step {prev_best_step})")
                else:
                    print(f"\n[Step {step}] ⭐ NEW BEST MODEL ⭐ - Val Loss: {avg_val_loss:.6f} (First evaluation)")
            else:
                print(f"\n[Step {step}] Val Loss: {avg_val_loss:.6f} "
                      f"(Best: {best_val_loss:.6f} at step {last_improvement_step})")
                
                # Check for early stopping
                if last_improvement_step > 0:
                    steps_since_improvement = step - last_improvement_step
                    if steps_since_improvement >= early_stopping_patience:
                        print(f"\n[Step {step}] Early stopping triggered: No improvement for {steps_since_improvement} steps "
                              f"(last improvement at step {last_improvement_step})")
                        print("Training stopped early due to no improvement.")
                        break
            
            pbar.set_postfix(
                train_loss=f"{loss.item():.4f}",
                val_loss=f"{avg_val_loss:.4f}",
                best_val_loss=f"{best_val_loss:.4f}"
            )
    
    # Break outer loop if early stopping was triggered
    if last_improvement_step > 0 and (step - last_improvement_step) >= early_stopping_patience:
        break

# Save final model
torch.save(maskgit.state_dict(), "maskgit_uncond.pt")
print("\n" + "="*80)
print("TRAINING SUMMARY - Best Model:")
print("="*80)
if best_val_loss < float('inf'):
    print(f"Best Model: Step {last_improvement_step} with Validation Loss: {best_val_loss:.6f}")
    print(f"Best model saved as: maskgit_uncond.best.pt")
else:
    print("No validation evaluations performed")
print("Final model saved as: maskgit_uncond.pt")
print("="*80)
