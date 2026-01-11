import os
os.environ["BEARTYPE_IS_BEARTYPE"] = "0"

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torchvision.utils import make_grid, save_image
from einops import rearrange

from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit, Transformer
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE

# -------------------------
# Config
# -------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VQGAN_CKPT   = "/home/012002744/hicplus_thesis/maskgit/muse_vqgan/vqgan_results_2layers_4096/vae.best.pt"
MASKGIT_CKPT = "/home/012002744/hicplus_thesis/maskgit/maskgit_cond.best.pt"
DATA_NPY     = "/home/012002744/hicplus_thesis/maskgit/hic_vqgan_train_hr.npy"

OUT_DIR = Path("cond_samples")
OUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE   = 8
TIMESTEPS    = 12
TEMPERATURE  = 1.0
THIN_FRAC    = 0.10
IMAGE_SIZE   = 128
CODEBOOK_SIZE = 4096

# -------------------------
# LR corruption (same idea as training)
# -------------------------
def make_lr(hr, frac=0.1):
    counts = hr.clamp(min=0)
    dist = torch.distributions.Binomial(
        total_count=torch.round(counts),
        probs=frac,
    )
    return dist.sample().to(hr.dtype)

# -------------------------
# Robust checkpoint load
# -------------------------
def load_state_dict_flex(path: str):
    ckpt = torch.load(path, map_location=DEVICE)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt

# -------------------------
# SAFE decode that works for your VQ-trained VAE
# (does NOT call vae.decode_from_ids)
# -------------------------
@torch.no_grad()
def decode_ids_vq_safe(vae: VQGanVAE, ids_hw: torch.Tensor) -> torch.Tensor:
    """
    ids_hw: [B, H, W] long
    returns: [B, C, H_img, W_img]
    """
    q = vae.quantizer

    # Find codebook embedding/tensor inside vector_quantize_pytorch quantizer
    cb = None

    # common patterns
    if hasattr(q, "codebook"):
        cb = getattr(q, "codebook")
        # sometimes codebook is a module with `.embed`
        if hasattr(cb, "embed"):
            cb = cb.embed

    if cb is None and hasattr(q, "embed"):
        cb = q.embed
    if cb is None and hasattr(q, "embedding"):
        cb = q.embedding
    if cb is None and hasattr(q, "_codebook") and hasattr(q._codebook, "embed"):
        cb = q._codebook.embed

    if cb is None:
        raise AttributeError("Could not locate VQ codebook inside vae.quantizer")

    # ids -> codes [B,H,W,D]
    if isinstance(cb, torch.nn.Embedding):
        codes = cb(ids_hw)
    else:
        # tensor / parameter shaped [K,D]
        codes = cb[ids_hw]

    # project to encoded dim if needed
    if hasattr(q, "project_out"):
        fmap = q.project_out(codes)
    else:
        fmap = codes

    fmap = rearrange(fmap, "b h w c -> b c h w")
    return vae.decode(fmap)

# -------------------------
# Fixed conditional generation (no model.generate)
# -------------------------
@torch.no_grad()
def fixed_generate_conditional(model: MaskGit, lr_images: torch.Tensor, timesteps=12, temperature=1.0):
    fmap = model.vae.get_encoded_fmap_size(model.image_size)
    seq_len = fmap * fmap
    device = next(model.parameters()).device
    B = lr_images.shape[0]

    # encode LR -> ids for conditioning
    _, ids_lr_grid, _ = model.vae.encode(lr_images.to(device))
    ids_lr = rearrange(ids_lr_grid, "b h w -> b (h w)")
    assert ids_lr.shape == (B, seq_len)

    # start fully masked
    ids = torch.full((B, seq_len), model.transformer.mask_id, device=device, dtype=torch.long)
    scores = torch.zeros_like(ids, dtype=torch.float)

    for t in torch.linspace(0, 1, timesteps, device=device):
        mask_prob = model.noise_schedule(t)
        num_mask = max(int(mask_prob * seq_len), 1)

        masked_idx = scores.topk(num_mask, dim=-1).indices
        ids = ids.scatter(1, masked_idx, model.transformer.mask_id)

        logits = model.transformer(ids, cond=ids_lr)         # [B, N, V]
        probs  = F.softmax(logits / temperature, dim=-1)

        # multinomial needs 2D -> flatten B*N
        B2, N, V = probs.shape
        sampled = torch.multinomial(probs.reshape(B2 * N, V), 1).reshape(B2, N)

        is_mask = ids == model.transformer.mask_id
        ids = torch.where(is_mask, sampled, ids)

        scores = 1 - probs.gather(2, sampled[..., None]).squeeze(-1)
        scores = scores.masked_fill(~is_mask, -1e5)

    ids_hw = ids.view(B, fmap, fmap)

    # IMPORTANT: use VQ-safe decoder
    return decode_ids_vq_safe(model.vae, ids_hw)

# -------------------------
# Load VQGAN EXACTLY like training (VQ, not LFQ)
# -------------------------
vae = VQGanVAE(
    dim=256,
    channels=1,
    layers=2,
    codebook_size=CODEBOOK_SIZE,
    lookup_free_quantization=False,   # MUST match training
    use_vgg_and_gan=False,
    vq_kwargs=dict(
        codebook_dim=64,
        decay=0.95,
        commitment_weight=1.0,
        kmeans_init=True,
        use_cosine_sim=True,
    ),
).to(DEVICE)

vae.load_state_dict(load_state_dict_flex(VQGAN_CKPT), strict=False)
vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# infer seq_len from encode
with torch.no_grad():
    dummy = torch.zeros(1, 1, IMAGE_SIZE, IMAGE_SIZE, device=DEVICE)
    _, ids0, _ = vae.encode(dummy)
    token_h, token_w = ids0.shape[1:]
    seq_len = token_h * token_w

print(f"[INFO] Token grid: {token_h} x {token_w} => seq_len={seq_len}")

# -------------------------
# Transformer + MaskGit
# -------------------------
transformer = Transformer(
    num_tokens=vae.codebook_size,
    seq_len=seq_len,
    dim=512,
    depth=8,
    heads=8,
    dim_head=64,
    self_cond=False,
    cond_tokens=True,
).to(DEVICE)

maskgit = MaskGit(
    image_size=IMAGE_SIZE,
    transformer=transformer,
    vae=vae,
    no_mask_token_prob=0.1,
).to(DEVICE)

maskgit.load_state_dict(torch.load(MASKGIT_CKPT, map_location=DEVICE), strict=False)
maskgit.eval()

# -------------------------
# Load HR data and make LR condition
# -------------------------
data = torch.from_numpy(np.load(DATA_NPY)).float()
idx = torch.randperm(len(data))[:BATCH_SIZE]

hr = data[idx].to(DEVICE)              # [B,1,128,128]
lr = make_lr(hr, frac=THIN_FRAC).to(DEVICE)

# -------------------------
# Generate
# -------------------------
gen = fixed_generate_conditional(maskgit, lr, timesteps=TIMESTEPS, temperature=TEMPERATURE)

# -------------------------
# Normalize for visualization
# -------------------------
def norm(x):
    x = x - x.min()
    x = x / (x.max() + 1e-8)
    return x

hr_v  = norm(hr)
lr_v  = norm(lr)
gen_v = norm(gen)

# Grid: [LR | HR | GEN]
rows = []
for i in range(BATCH_SIZE):
    rows.append(torch.cat([lr_v[i], hr_v[i], gen_v[i]], dim=-1))

grid = make_grid(rows, nrow=1, padding=4)
save_image(grid, OUT_DIR / "conditional_comparison.png")

print(f"[OK] Saved: {OUT_DIR / 'conditional_comparison.png'}")
