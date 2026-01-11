#!/usr/bin/env python3
"""
infer_gated_maskgit_log1p_v2.py

Fixes included (vs your current script):
1) BUGFIX: Avoid idx padding masking token 0 repeatedly (was corrupting outputs).
2) Better lr_zero gating: compute "no evidence" in COUNT space with SUM pooling.
3) Gentler refinement: cap % of editable tokens masked per step (default 10%).
4) Optional mass calibration after merge_in_counts: match final sum to proposal sum.

Pipeline:
  LR (log1p) + proposal (log1p) -> VQGAN encode -> MaskGit refine (GATED) -> VQGAN decode
  Optional merge:
    final = (1-G_up)*proposal + G_up*refined   (in counts or log space)
  Optional mass calibration (counts): enforce sum(final)=sum(proposal) per sample.

Expected .npy shapes:
- proposal_npy: [N, 1, 128, 128] or [N, 128, 128]  (log1p)
- lr_npy:       [N, 1, 128, 128] or [N, 128, 128]  (log1p)
- hr_npy: optional (log1p) for viz / sim LR if lr_npy missing
"""

import os
os.environ["BEARTYPE_IS_BEARTYPE"] = "0"

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import make_grid, save_image

from muse_maskgit_pytorch.muse_maskgit_pytorch import MaskGit, Transformer
from muse_maskgit_pytorch.vqgan_vae import VQGanVAE


# -------------------------
# Utils
# -------------------------
def load_state_dict_flex(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        return ckpt["model"]
    return ckpt


def ensure_nchw(x: np.ndarray) -> np.ndarray:
    """Ensure [N,1,H,W] float32."""
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D array, got shape {x.shape}")
    return x.astype(np.float32, copy=False)


def norm_for_vis(x: torch.Tensor) -> torch.Tensor:
    x = x.detach()
    x = x - x.amin(dim=(-2, -1), keepdim=True)
    x = x / (x.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return x


# -------------------------
# Log1p-aware LR simulation (ONLY if needed)
# -------------------------
@torch.no_grad()
def make_lr_from_log1p(hr_log1p: torch.Tensor, frac: float) -> torch.Tensor:
    """
    hr_log1p: [B,1,H,W] in log1p space
    returns: lr_log1p in log1p space
    """
    hr_counts = torch.expm1(hr_log1p).clamp(min=0)
    dist = torch.distributions.Binomial(
        total_count=torch.round(hr_counts),
        probs=frac,
    )
    lr_counts = dist.sample().to(hr_counts.dtype)
    return torch.log1p(lr_counts)


# -------------------------
# VQ safe decode
# -------------------------
@torch.no_grad()
def decode_ids_vq_safe(vae: VQGanVAE, ids_hw: torch.Tensor) -> torch.Tensor:
    """
    ids_hw: [B, Ht, Wt] long
    returns: [B, C, H_img, W_img]
    """
    q = vae.quantizer

    cb = None
    if hasattr(q, "codebook"):
        cb = getattr(q, "codebook")
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

    if isinstance(cb, torch.nn.Embedding):
        codes = cb(ids_hw)          # [B,Ht,Wt,D]
    else:
        codes = cb[ids_hw]          # [B,Ht,Wt,D]

    if hasattr(q, "project_out"):
        fmap = q.project_out(codes)
    else:
        fmap = codes

    fmap = rearrange(fmap, "b h w c -> b c h w")
    return vae.decode(fmap)


# -------------------------
# Gating helpers
# -------------------------
@torch.no_grad()
def downsample_to_tokens_logspace(x_log1p: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
    """
    x_log1p: [B,1,H,W] in log1p space
    returns: [B,1,token_h,token_w] using avg pooling in log space (heuristic)
    """
    B, C, H, W = x_log1p.shape
    assert C == 1
    assert H % token_h == 0 and W % token_w == 0
    kH = H // token_h
    kW = W // token_w
    return F.avg_pool2d(x_log1p, kernel_size=(kH, kW), stride=(kH, kW))


@torch.no_grad()
def downsample_sum_counts_from_log1p(x_log1p: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
    """
    x_log1p: [B,1,H,W] log1p
    returns: [B,1,token_h,token_w] SUM of counts inside each token cell
    """
    x_counts = torch.expm1(x_log1p).clamp(min=0)
    B, C, H, W = x_counts.shape
    assert C == 1
    assert H % token_h == 0 and W % token_w == 0
    kH = H // token_h
    kW = W // token_w
    pooled = F.avg_pool2d(x_counts, kernel_size=(kH, kW), stride=(kH, kW))
    return pooled * float(kH * kW)


@torch.no_grad()
def compute_gate(
    lr_log1p: torch.Tensor,
    proposal_log1p: torch.Tensor,
    token_h: int,
    token_w: int,
    gate_mode: str,
    lr_tau_counts: float,
    topk_frac: float,
) -> torch.Tensor:
    """
    Returns:
      gate_hw: [B, token_h, token_w] bool  (True => editable)
    """
    if gate_mode == "lr_zero":
        # Robust: decide "no evidence" in COUNT space using sum pooling.
        lr32_sum = downsample_sum_counts_from_log1p(lr_log1p, token_h, token_w).squeeze(1)  # [B,th,tw]
        gate = (lr32_sum <= lr_tau_counts)

    elif gate_mode == "topk_diff":
        # Diff in log space (heuristic) between proposal and LR
        lr32 = downsample_to_tokens_logspace(lr_log1p, token_h, token_w).squeeze(1)
        p32  = downsample_to_tokens_logspace(proposal_log1p, token_h, token_w).squeeze(1)
        diff = (p32 - lr32).abs()  # [B,th,tw]

        B = diff.shape[0]
        flat = diff.view(B, -1)
        k = max(int(topk_frac * flat.shape[1]), 1)
        topk_idx = flat.topk(k, dim=-1).indices
        gate_flat = torch.zeros_like(flat, dtype=torch.bool)
        gate_flat.scatter_(1, topk_idx, True)
        gate = gate_flat.view(B, token_h, token_w)

    else:
        raise ValueError(f"Unknown gate_mode: {gate_mode}")

    return gate


@torch.no_grad()
def upsample_gate_to_image(gate_hw: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    gate_hw: [B,th,tw] bool
    returns: [B,1,H,W] float in {0,1}
    """
    gate = gate_hw.float().unsqueeze(1)
    return F.interpolate(gate, size=(H, W), mode="nearest")


# -------------------------
# GATED conditional generation
# -------------------------
@torch.no_grad()
def generate_gated_conditional(
    model: MaskGit,
    lr_log1p: torch.Tensor,         # [B,1,128,128]
    proposal_log1p: torch.Tensor,   # [B,1,128,128]
    gate_hw: torch.Tensor,          # [B,th,tw] bool
    timesteps: int,
    temperature: float,
    max_edit_frac: float,
) -> torch.Tensor:
    """
    Returns refined image in the same space the VAE operates on.
    """
    device = next(model.parameters()).device
    fmap = model.vae.get_encoded_fmap_size(model.image_size)
    seq_len = fmap * fmap
    B = lr_log1p.shape[0]

    assert gate_hw.shape == (B, fmap, fmap), f"gate_hw must be [B,{fmap},{fmap}]"

    # Encode LR -> conditioning tokens
    _, ids_lr_grid, _ = model.vae.encode(lr_log1p.to(device))
    ids_lr = rearrange(ids_lr_grid, "b h w -> b (h w)")  # [B,N]

    # Encode proposal -> init tokens
    _, ids_p_grid, _ = model.vae.encode(proposal_log1p.to(device))
    ids_init = rearrange(ids_p_grid, "b h w -> b (h w)")  # [B,N]

    gate_flat = gate_hw.view(B, seq_len)
    editable_count = gate_flat.sum(dim=-1)  # [B]

    # Start from proposal
    ids = ids_init.clone()

    # scores used for selecting which tokens to mask next (uncertainty)
    # Initialize with tiny random noise so we don't always pick the same indices when all scores tie.
    scores = (torch.rand((B, seq_len), device=device) * 1e-3).float()
    scores = scores.masked_fill(~gate_flat, -1e5)

    for t in torch.linspace(0, 1, timesteps, device=device):
        # If no editable tokens for a sample, skip it naturally (editable_count=0)
        if int(editable_count.max().item()) == 0:
            break

        mask_prob = model.noise_schedule(t)

        # baseline number to mask among editable positions
        num_mask = (mask_prob * editable_count.float()).long()

        # ensure valid
        num_mask = torch.clamp(num_mask, min=0)
        num_mask = torch.minimum(num_mask, editable_count)

        # cap for gentle refinement
        cap = (max_edit_frac * editable_count.float()).long()
        cap = torch.clamp(cap, min=1)
        num_mask = torch.minimum(num_mask, cap)
        num_mask = torch.where(editable_count > 0, torch.clamp(num_mask, min=1), num_mask)

        # choose indices to mask: top-k by score within editable
        masked_idx_list = []
        for b in range(B):
            k = int(num_mask[b].item())
            if k <= 0 or int(editable_count[b].item()) == 0:
                masked_idx = torch.empty((0,), device=device, dtype=torch.long)
            else:
                masked_idx = scores[b].topk(k, dim=-1).indices
            masked_idx_list.append(masked_idx)

        # BUGFIX: build boolean mask per-sample (no padding -> no accidental token-0 masking)
        is_mask = torch.zeros((B, seq_len), device=device, dtype=torch.bool)
        for b in range(B):
            mi = masked_idx_list[b]
            if mi.numel() > 0:
                is_mask[b, mi] = True

        # apply mask tokens only at those positions
        ids = torch.where(is_mask, model.transformer.mask_id, ids)

        # predict tokens
        logits = model.transformer(ids, cond=ids_lr)  # [B,N,V]
        probs = F.softmax(logits / temperature, dim=-1)

        # sample tokens
        B2, N, V = probs.shape
        sampled = torch.multinomial(probs.reshape(B2 * N, V), 1).reshape(B2, N)

        # only update masked positions
        ids = torch.where(is_mask, sampled, ids)

        # update scores (uncertainty) for masked positions
        conf = probs.gather(2, sampled[..., None]).squeeze(-1)  # [B,N]
        new_scores = 1.0 - conf
        scores = torch.where(is_mask, new_scores, scores)
        scores = scores.masked_fill(~gate_flat, -1e5)

    ids_hw = ids.view(B, fmap, fmap)
    refined = decode_ids_vq_safe(model.vae, ids_hw)
    return refined


# -------------------------
# Main
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vqgan_ckpt", required=True)
    p.add_argument("--maskgit_ckpt", required=True)

    p.add_argument("--proposal_npy", required=True, help="HiCPlus/FiLM proposal in log1p space")
    p.add_argument("--lr_npy", default=None, help="LR input in log1p space (recommended)")
    p.add_argument("--hr_npy", default=None, help="Optional HR target in log1p space (for visualization)")

    p.add_argument("--out_dir", default="cond_samples_gated")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    # VQGAN config (must match training)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--codebook_size", type=int, default=4096)

    # MaskGit sampling
    p.add_argument("--timesteps", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--max_edit_frac", type=float, default=0.10, help="Max fraction of editable tokens masked per step")

    # If lr_npy is missing, we can simulate from hr_npy in COUNT space
    p.add_argument("--thin_frac", type=float, default=0.10, help="Used only if simulating LR from HR")

    # Gating
    p.add_argument("--gate_mode", choices=["lr_zero", "topk_diff"], default="lr_zero")
    p.add_argument("--lr_tau_counts", type=float, default=0.0,
                   help="COUNT-space sum threshold per token-cell; 0.0 means exactly no reads in that cell")
    p.add_argument("--topk_frac", type=float, default=0.20, help="Fraction of tokens editable for topk_diff mode")

    # Merge controls
    p.add_argument("--do_merge", action="store_true", help="Merge proposal/refined using gate mask")
    p.add_argument("--merge_in_counts", action="store_true", help="Merge in count space (recommended)")
    p.add_argument("--soft_gate", action="store_true", help="Slightly soften gate (avg blur) before merging")
    p.add_argument("--mass_calibrate", action="store_true",
                   help="After merge_in_counts, rescale final to match proposal total counts per sample")

    # Save arrays
    p.add_argument("--save_npy", action="store_true", help="Save proposal/refined/final arrays as .npy")

    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load VQGAN (must match training)
    # -------------------------
    vae = VQGanVAE(
        dim=256,
        channels=1,
        layers=2,
        codebook_size=args.codebook_size,
        lookup_free_quantization=False,   # MUST match training
        use_vgg_and_gan=False,
        vq_kwargs=dict(
            codebook_dim=64,
            decay=0.95,
            commitment_weight=1.0,
            kmeans_init=True,
            use_cosine_sim=True,
        ),
    ).to(device)

    vae.load_state_dict(load_state_dict_flex(args.vqgan_ckpt, device), strict=False)
    vae.eval()
    for pp in vae.parameters():
        pp.requires_grad = False

    # infer token grid
    with torch.no_grad():
        dummy = torch.zeros(1, 1, args.image_size, args.image_size, device=device)
        _, ids0, _ = vae.encode(dummy)
        token_h, token_w = ids0.shape[1:]
        seq_len = token_h * token_w
    print(f"[INFO] Token grid: {token_h} x {token_w} => seq_len={seq_len}")

    # -------------------------
    # Build MaskGit
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
    ).to(device)

    maskgit = MaskGit(
        image_size=args.image_size,
        transformer=transformer,
        vae=vae,
        no_mask_token_prob=0.1,
    ).to(device)

    maskgit.load_state_dict(torch.load(args.maskgit_ckpt, map_location=device), strict=False)
    maskgit.eval()

    # -------------------------
    # Load arrays (LOG1P)
    # -------------------------
    proposal_all = torch.from_numpy(ensure_nchw(np.load(args.proposal_npy))).float()
    N = proposal_all.shape[0]

    lr_all = None
    if args.lr_npy is not None and Path(args.lr_npy).exists():
        lr_all = torch.from_numpy(ensure_nchw(np.load(args.lr_npy))).float()
        if lr_all.shape[0] != N:
            raise ValueError(f"lr_npy N={lr_all.shape[0]} does not match proposal N={N}")

    hr_all = None
    if args.hr_npy is not None and Path(args.hr_npy).exists():
        hr_all = torch.from_numpy(ensure_nchw(np.load(args.hr_npy))).float()
        if hr_all.shape[0] != N:
            raise ValueError(f"hr_npy N={hr_all.shape[0]} does not match proposal N={N}")

    # sample batch
    idx = torch.randperm(N)[: args.batch_size]
    proposal = proposal_all[idx].to(device)

    if lr_all is not None:
        lr = lr_all[idx].to(device)
    else:
        if hr_all is not None:
            hr_tmp = hr_all[idx].to(device)
            lr = make_lr_from_log1p(hr_tmp, frac=args.thin_frac).to(device)
            print("[INFO] lr_npy not provided; simulated LR by thinning HR in count space.")
        else:
            lr = make_lr_from_log1p(proposal, frac=args.thin_frac).to(device)
            print("[WARN] lr_npy and hr_npy not provided; simulated LR by thinning the PROPOSAL. (OK only for smoke tests)")

    hr = hr_all[idx].to(device) if hr_all is not None else None

    # -------------------------
    # Gate in token space
    # -------------------------
    gate_hw = compute_gate(
        lr_log1p=lr,
        proposal_log1p=proposal,
        token_h=token_h,
        token_w=token_w,
        gate_mode=args.gate_mode,
        lr_tau_counts=args.lr_tau_counts,
        topk_frac=args.topk_frac,
    )  # [B,th,tw] bool

    # -------------------------
    # Run gated MaskGit refinement
    # -------------------------
    refined = generate_gated_conditional(
        model=maskgit,
        lr_log1p=lr,
        proposal_log1p=proposal,
        gate_hw=gate_hw,
        timesteps=args.timesteps,
        temperature=args.temperature,
        max_edit_frac=args.max_edit_frac,
    )

    # -------------------------
    # Merge (optional)
    # -------------------------
    if args.do_merge:
        G_up = upsample_gate_to_image(gate_hw, args.image_size, args.image_size)  # [B,1,H,W]
        if args.soft_gate:
            G_up = F.avg_pool2d(G_up, kernel_size=3, stride=1, padding=1)

        if args.merge_in_counts:
            prop_c = torch.expm1(proposal).clamp(min=0)
            ref_c  = torch.expm1(refined).clamp(min=0)
            final_c = (1.0 - G_up) * prop_c + G_up * ref_c

            if args.mass_calibrate:
                prop_sum = prop_c.sum(dim=(-2, -1), keepdim=True)
                final_sum = final_c.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-6)
                alpha = prop_sum / final_sum
                final_c = final_c * alpha

            final = torch.log1p(final_c)
        else:
            final = (1.0 - G_up) * proposal + G_up * refined
    else:
        final = refined

    # -------------------------
    # Save visuals
    # -------------------------
    lr_v       = norm_for_vis(lr.cpu())
    prop_v     = norm_for_vis(proposal.cpu())
    ref_v      = norm_for_vis(refined.cpu())
    final_v    = norm_for_vis(final.cpu())
    gate_up    = upsample_gate_to_image(gate_hw, args.image_size, args.image_size).cpu()

    rows = []
    for i in range(args.batch_size):
        parts = [lr_v[i], prop_v[i], ref_v[i], final_v[i]]
        if hr is not None:
            hr_v = norm_for_vis(hr.cpu())
            parts.append(hr_v[i])
        rows.append(torch.cat(parts, dim=-1))

    grid = make_grid(rows, nrow=1, padding=4)
    grid_path = out_dir / "gated_comparison.png"
    save_image(grid, grid_path)

    gate_path = out_dir / "gate_upsampled.png"
    save_image(gate_up, gate_path)

    print(f"[OK] Saved: {grid_path}")
    print(f"[OK] Saved: {gate_path}")

    # -------------------------
    # Save arrays (optional)
    # -------------------------
    if args.save_npy:
        np.save(out_dir / "idx.npy", idx.cpu().numpy())
        np.save(out_dir / "proposal_log1p.npy", proposal.detach().cpu().numpy())
        np.save(out_dir / "lr_log1p.npy", lr.detach().cpu().numpy())
        np.save(out_dir / "refined_log1p.npy", refined.detach().cpu().numpy())
        np.save(out_dir / "final_log1p.npy", final.detach().cpu().numpy())
        np.save(out_dir / "gate_hw.npy", gate_hw.detach().cpu().numpy())
        print(f"[OK] Saved .npy outputs to: {out_dir}")


if __name__ == "__main__":
    main()
