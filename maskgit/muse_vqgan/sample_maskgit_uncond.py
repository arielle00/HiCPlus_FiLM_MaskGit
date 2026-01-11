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
# helper: attach VQ codebook so decode_from_ids works
# ======================================================

def attach_vq_codebook_if_needed(vae: VQGanVAE):
    """
    Your repo's VQGanVAE.decode_from_ids() expects `vae.codebook` when
    lookup_free_quantization=False, but VQGanVAE.__init__ does not set it.

    We attach it from the quantizer so decode_from_ids works WITHOUT changing the library.
    """
    if getattr(vae, "lookup_free_quantization", True):
        return  # LFQ path doesn't need codebook

    if hasattr(vae, "codebook"):
        return

    q = getattr(vae, "quantizer", None)
    if q is None:
        raise AttributeError("VAE has no .quantizer; cannot attach codebook.")

    # Most common in vector_quantize_pytorch: q._codebook is an nn.Embedding or similar
    candidates = []
    if hasattr(q, "_codebook"):
        candidates.append(q._codebook)
    if hasattr(q, "codebook"):
        candidates.append(q.codebook)
    if hasattr(q, "embedding"):
        candidates.append(q.embedding)

    codebook = None
    for cb in candidates:
        if cb is None:
            continue
        # nn.Embedding or similar
        if hasattr(cb, "weight") and torch.is_tensor(cb.weight):
            codebook = cb.weight
            break
        # tensor-like
        if torch.is_tensor(cb):
            codebook = cb
            break

    if codebook is None:
        raise AttributeError(
            "Could not find codebook on vae.quantizer. "
            "Print dir(vae.quantizer) and tell me what attributes it has."
        )

    vae.codebook = codebook  # <- what decode_from_ids expects

# ======================================================
# LOAD VQGAN (MUST match training)
# ======================================================

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

ckpt = torch.load(VQGAN_CKPT, map_location=DEVICE)
state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

missing, unexpected = vae.load_state_dict(state_dict, strict=False)
print("[VQGAN] Missing keys:", missing)
print("[VQGAN] Unexpected keys:", unexpected)

vae.eval()
for p in vae.parameters():
    p.requires_grad = False

# ✅ patch so decode_from_ids won't crash in VQ mode
attach_vq_codebook_if_needed(vae)

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
# FIXED GENERATE (PATCH)  (unconditional)
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
        sampled = torch.multinomial(probs.view(B * N, V), 1).view(B, N)

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
