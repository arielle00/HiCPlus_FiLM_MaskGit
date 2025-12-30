import torch
from pathlib import Path
from torchvision.utils import make_grid, save_image
import torch.nn.functional as F

from muse_maskgit_pytorch.vqgan_vae import VQGanVAE
from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit, Transformer

# ======================================================
# CONFIG
# ======================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VQGAN_CKPT   = "vqgan_results_2layers_4096/vae.best.pt"
MASKGIT_CKPT = "../maskgit_uncond.pt"

OUT_DIR = Path("maskgit_samples")
OUT_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 128
NUM_SAMPLES = 32
TIMESTEPS = 18
TEMPERATURE = 1.0

TOKEN_H = 32
TOKEN_W = 32
SEQ_LEN = TOKEN_H * TOKEN_W
CODEBOOK_SIZE = 4096

# ======================================================
# LOAD VQGAN (LFQ SAFE)
# ======================================================

vae = VQGanVAE(
    dim=256,
    channels=1,
    layers=2,
    codebook_size=CODEBOOK_SIZE,
    use_vgg_and_gan=False,
).to(DEVICE)

ckpt = torch.load(VQGAN_CKPT, map_location=DEVICE)
state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

missing, unexpected = vae.load_state_dict(state_dict, strict=False)
print("[VQGAN] Missing keys:", missing)
print("[VQGAN] Unexpected keys:", unexpected)

vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# sanity check
with torch.no_grad():
    x = torch.randn(1, 1, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)
    _, ids, _ = vae.encode(x)
    recon = vae.decode_from_ids(ids)
    print("[VQGAN] Recon OK:", recon.shape)

# ======================================================
# TRANSFORMER
# ======================================================

transformer = Transformer(
    num_tokens=CODEBOOK_SIZE,
    seq_len=SEQ_LEN,
    dim=512,
    depth=8,
    heads=8,
    dim_head=64,
).to(DEVICE)

# ======================================================
# MASKGIT
# ======================================================

maskgit = MaskGit(
    transformer=transformer,
    vae=vae,
    image_size=IMAGE_SIZE,
).to(DEVICE)

missing, unexpected = maskgit.load_state_dict(
    torch.load(MASKGIT_CKPT, map_location=DEVICE),
    strict=False
)

print("[MaskGit] Missing keys:", missing)
print("[MaskGit] Unexpected keys:", unexpected)

maskgit.eval()

# ======================================================
# FIXED GENERATE (PATCH)
# ======================================================

@torch.no_grad()
def fixed_generate(model, batch_size, timesteps=12, temperature=1.0):
    fmap = model.vae.get_encoded_fmap_size(model.image_size)
    seq_len = fmap * fmap
    device = next(model.parameters()).device

    ids = torch.full(
        (batch_size, seq_len),
        model.transformer.mask_id,
        device=device,
        dtype=torch.long,
    )

    scores = torch.zeros_like(ids, dtype=torch.float)

    for t in torch.linspace(0, 1, timesteps, device=device):
        mask_prob = model.noise_schedule(t)
        num_mask = max(int(mask_prob * seq_len), 1)

        masked_idx = scores.topk(num_mask, dim=-1).indices
        ids = ids.scatter(1, masked_idx, model.transformer.mask_id)

        logits = model.transformer(ids)
        probs = F.softmax(logits / temperature, dim=-1)

        B, N, V = probs.shape
        sampled = torch.multinomial(
            probs.view(B * N, V),
            1
        ).view(B, N)

        is_mask = ids == model.transformer.mask_id
        ids = torch.where(is_mask, sampled, ids)

        scores = 1 - probs.gather(2, sampled[..., None]).squeeze(-1)
        scores = scores.masked_fill(~is_mask, -1e5)

    ids = ids.view(batch_size, fmap, fmap)
    return model.vae.decode_from_ids(ids)

# ======================================================
# SAMPLE
# ======================================================

with torch.no_grad():
    images = fixed_generate(
        maskgit,
        batch_size=NUM_SAMPLES,
        timesteps=TIMESTEPS,
        temperature=TEMPERATURE,
    )

images = images.clamp(0, 1)

grid = make_grid(images, nrow=8, normalize=True)
save_image(grid, OUT_DIR / "samples.png")

print(f"[OK] Saved samples to {OUT_DIR / 'samples.png'}")
