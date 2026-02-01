#!/usr/bin/env python3
"""
infer_gated_maskgit_log1p_v6.py

Key fixes vs v5:
1) Proposal calibration to LR evidence (so proposal isn't globally inflated vs HR/LR).
   --mass_calibrate_to lr is the default recommendation for test-time.
2) LR-evidence-aware merge weight (merge refined mostly where LR has evidence).
   --merge_use_lr_evidence with --merge_lr_tau_counts, --merge_evidence_beta.
3) Refined clamp in counts space to prevent "dark hallucinated blocks".

All arrays are assumed log1p(counts), shape [N,1,H,W], float32.
"""

import os
os.environ["BEARTYPE_IS_BEARTYPE"] = "0"

import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torchvision.utils import save_image
from scipy.stats import spearmanr, pearsonr
from scipy.ndimage import sobel
from skimage.metrics import structural_similarity as ssim

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
    if x.ndim == 3:
        x = x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"Expected 3D or 4D array, got shape {x.shape}")
    return x.astype(np.float32, copy=False)


def same_device(*tensors, device):
    out = []
    for t in tensors:
        out.append(None if t is None else t.to(device))
    return out


def load_coords(path: str):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return np.load(str(p))


@torch.no_grad()
def make_lr_from_log1p(hr_log1p: torch.Tensor, frac: float) -> torch.Tensor:
    hr_counts = torch.expm1(hr_log1p).clamp(min=0)
    dist = torch.distributions.Binomial(
        total_count=torch.round(hr_counts),
        probs=frac,
    )
    lr_counts = dist.sample().to(hr_counts.dtype)
    return torch.log1p(lr_counts)


# -------------------------
# VQ decode safe
# -------------------------
@torch.no_grad()
def decode_ids_vq_safe(vae: VQGanVAE, ids_hw: torch.Tensor) -> torch.Tensor:
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
        codes = cb(ids_hw)
    else:
        codes = cb[ids_hw]

    fmap = q.project_out(codes) if hasattr(q, "project_out") else codes
    fmap = rearrange(fmap, "b h w c -> b c h w")
    return vae.decode(fmap)


# -------------------------
# Token-grid helpers
# -------------------------
@torch.no_grad()
def downsample_to_tokens_logspace(x: torch.Tensor, token_h: int, token_w: int) -> torch.Tensor:
    B, C, H, W = x.shape
    assert C == 1
    assert H % token_h == 0 and W % token_w == 0
    kH = H // token_h
    kW = W // token_w
    return F.avg_pool2d(x, kernel_size=(kH, kW), stride=(kH, kW))


@torch.no_grad()
def compute_low_evidence_mask(lr_log: torch.Tensor, token_h: int, token_w: int, lr_tau_log: float) -> torch.Tensor:
    lr_tok = downsample_to_tokens_logspace(lr_log, token_h, token_w).squeeze(1)
    return (lr_tok <= lr_tau_log)


@torch.no_grad()
def compute_gate(
    lr_log: torch.Tensor,
    proposal_log: torch.Tensor,
    token_h: int,
    token_w: int,
    gate_mode: str,
    lr_tau_log: float,
    topk_frac: float,
    topk_only_where_low_evidence: bool,
    max_edit_frac: float,
):
    B = lr_log.shape[0]
    low_ev = compute_low_evidence_mask(lr_log, token_h, token_w, lr_tau_log)
    low_ev_frac = float(low_ev.float().mean().item())

    if gate_mode == "lr_zero":
        gate = low_ev.clone()

    elif gate_mode == "topk_diff":
        lr32 = downsample_to_tokens_logspace(lr_log, token_h, token_w).squeeze(1)
        p32  = downsample_to_tokens_logspace(proposal_log, token_h, token_w).squeeze(1)
        diff = (p32 - lr32).abs()

        flat = diff.view(B, -1)
        if topk_only_where_low_evidence:
            mask_flat = low_ev.view(B, -1)
            flat = flat.masked_fill(~mask_flat, float("-inf"))

        N = flat.shape[1]
        k_req = max(int(topk_frac * N), 1)
        k_cap = max(int(max_edit_frac * N), 1) if max_edit_frac > 0 else k_req
        k = min(k_req, k_cap)

        gate_flat = torch.zeros((B, N), device=lr_log.device, dtype=torch.bool)
        for b in range(B):
            vals = flat[b]
            if torch.isinf(vals).all():
                continue
            cand = (~torch.isinf(vals)).sum().item()
            k_b = min(k, int(cand))
            if k_b <= 0:
                continue
            idx = vals.topk(k_b, dim=-1).indices
            gate_flat[b, idx] = True

        gate = gate_flat.view(B, token_h, token_w)
    else:
        raise ValueError(gate_mode)

    return gate, low_ev_frac


@torch.no_grad()
def maybe_add_outside_gate_edits(gate_hw: torch.Tensor, frac_outside: float, rng: torch.Generator) -> torch.Tensor:
    if frac_outside <= 0:
        return gate_hw
    B, th, tw = gate_hw.shape
    N = th * tw
    add_k = int(frac_outside * N)
    if add_k <= 0:
        return gate_hw

    gate_flat = gate_hw.view(B, -1)
    new_gate = gate_flat.clone()

    for b in range(B):
        outside = (~gate_flat[b]).nonzero(as_tuple=False).squeeze(1)
        if outside.numel() == 0:
            continue
        k = min(add_k, outside.numel())
        perm = outside[torch.randperm(outside.numel(), generator=rng, device=outside.device)[:k]]
        new_gate[b, perm] = True

    return new_gate.view(B, th, tw)


@torch.no_grad()
def upsample_gate_to_image(gate_hw: torch.Tensor, H: int, W: int) -> torch.Tensor:
    return F.interpolate(gate_hw.float().unsqueeze(1), size=(H, W), mode="nearest")


# -------------------------
# Proposal calibration (to LR evidence)
# -------------------------
@torch.no_grad()
def calibrate_proposal_to_target(
    proposal_log: torch.Tensor,
    target_log: torch.Tensor,
    mode: str,
    eps: float = 1e-8,
    alpha_min: float = 1e-3,
    alpha_max: float = 1e3,
    lr_pos_mask: bool = True,
):
    """
    mode:
      - "none"
      - "hr_sum": match total sum in counts
      - "lr_sum_pos": match sum over pixels where LR>0 (recommended for sparse LR)
    """
    if mode == "none":
        return proposal_log, None

    prop_c = torch.expm1(proposal_log).clamp(min=0)
    tgt_c  = torch.expm1(target_log).clamp(min=0)

    if mode == "hr_sum":
        prop_sum = prop_c.flatten(1).sum(dim=1).clamp(min=eps)
        tgt_sum  = tgt_c.flatten(1).sum(dim=1).clamp(min=eps)

    elif mode == "lr_sum_pos":
        if lr_pos_mask:
            m = (tgt_c > 0).float()
            prop_sum = (prop_c * m).flatten(1).sum(dim=1).clamp(min=eps)
            tgt_sum  = (tgt_c  * m).flatten(1).sum(dim=1).clamp(min=eps)
            # if a sample is all-zero in LR, fallback to full sum
            bad = (m.flatten(1).sum(dim=1) < 10)
            if bad.any():
                prop_sum2 = prop_c.flatten(1).sum(dim=1).clamp(min=eps)
                tgt_sum2  = tgt_c.flatten(1).sum(dim=1).clamp(min=eps)
                prop_sum = torch.where(bad, prop_sum2, prop_sum)
                tgt_sum  = torch.where(bad, tgt_sum2,  tgt_sum)
        else:
            prop_sum = prop_c.flatten(1).sum(dim=1).clamp(min=eps)
            tgt_sum  = tgt_c.flatten(1).sum(dim=1).clamp(min=eps)
    else:
        raise ValueError(mode)

    alpha = (tgt_sum / prop_sum).clamp(min=alpha_min, max=alpha_max)
    prop_scaled = torch.log1p(prop_c * alpha.view(-1, 1, 1, 1))
    return prop_scaled, alpha


# -------------------------
# MaskGit init + sampling
# -------------------------
@torch.no_grad()
def init_ids(model: MaskGit, ids_proposal: torch.Tensor, gate_flat: torch.Tensor, init_mode: str,
            rng: torch.Generator, seed: int = None):
    B, N = ids_proposal.shape
    V = model.transformer.num_tokens
    device = ids_proposal.device

    rng_device = torch.Generator(device=device)
    if seed is not None:
        rng_device.manual_seed(seed)
    rng = rng_device

    if init_mode == "proposal":
        return ids_proposal.clone()
    if init_mode == "noise":
        return torch.randint(low=0, high=V, size=(B, N), device=device, generator=rng)
    if init_mode == "mixed":
        ids = ids_proposal.clone()
        rand = torch.randint(low=0, high=V, size=(B, N), device=device, generator=rng)
        return torch.where(gate_flat, rand, ids)
    raise ValueError(init_mode)


@torch.no_grad()
def generate_gated_conditional(
    model: MaskGit,
    lr_log: torch.Tensor,
    proposal_log: torch.Tensor,
    gate_hw: torch.Tensor,
    timesteps: int,
    temperature: float,
    init_mode: str,
    rng: torch.Generator,
    seed: int = None,
):
    device = next(model.parameters()).device
    rng_device = torch.Generator(device=device)
    if seed is not None:
        rng_device.manual_seed(seed)
    rng = rng_device

    fmap = model.vae.get_encoded_fmap_size(model.image_size)
    seq_len = fmap * fmap
    B = lr_log.shape[0]
    assert gate_hw.shape == (B, fmap, fmap)

    _, ids_lr_grid, _ = model.vae.encode(lr_log.to(device))
    ids_lr = rearrange(ids_lr_grid, "b h w -> b (h w)")

    _, ids_p_grid, _ = model.vae.encode(proposal_log.to(device))
    ids_prop = rearrange(ids_p_grid, "b h w -> b (h w)")

    gate_flat = gate_hw.view(B, seq_len)

    ids = init_ids(model, ids_prop, gate_flat, init_mode=init_mode, rng=rng, seed=seed)
    ids_init = ids.clone()

    scores = torch.zeros((B, seq_len), device=device, dtype=torch.float)
    scores = scores.masked_fill(~gate_flat, -1e5)
    editable_count = gate_flat.sum(dim=-1)

    for t in torch.linspace(0, 1, timesteps, device=device):
        mask_prob = model.noise_schedule(t)
        num_mask = (mask_prob * editable_count.float()).long().clamp(min=1)
        num_mask = torch.minimum(num_mask, editable_count.clamp(min=1))

        masked_idx_list = []
        max_k = 0
        for b in range(B):
            if editable_count[b].item() == 0:
                masked_idx = torch.empty((0,), device=device, dtype=torch.long)
            else:
                k = int(num_mask[b].item())
                masked_idx = scores[b].topk(k, dim=-1).indices
            masked_idx_list.append(masked_idx)
            max_k = max(max_k, masked_idx.numel())
        if max_k == 0:
            break

        idx_pad = torch.zeros((B, max_k), device=device, dtype=torch.long)
        for b in range(B):
            mi = masked_idx_list[b]
            if mi.numel() > 0:
                idx_pad[b, :mi.numel()] = mi

        ids = ids.scatter(1, idx_pad, model.transformer.mask_id)

        logits = model.transformer(ids, cond=ids_lr)
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)

        B2, N, V = probs.shape
        sampled = torch.multinomial(probs.reshape(B2 * N, V), 1, generator=rng).reshape(B2, N)

        is_mask = ids == model.transformer.mask_id
        ids = torch.where(is_mask, sampled, ids)

        conf = probs.gather(2, sampled[..., None]).squeeze(-1)
        new_scores = 1.0 - conf
        scores = torch.where(is_mask, new_scores, scores)
        scores = scores.masked_fill(~gate_flat, -1e5)

    ids_final = ids.clone()
    ids_hw = ids.view(B, fmap, fmap)
    refined = decode_ids_vq_safe(model.vae, ids_hw)

    changed = (ids_final != ids_prop) & gate_flat
    return {
        "refined_log": refined,
        "ids_prop": ids_prop,
        "ids_init": ids_init,
        "ids_final": ids_final,
        "gate_flat": gate_flat,
        "changed_in_gate": changed.sum(dim=1),
        "gate_counts": gate_flat.sum(dim=1),
    }


# -------------------------
# Merge helpers
# -------------------------
@torch.no_grad()
def lr_evidence_weight_map(lr_log: torch.Tensor, token_h: int, token_w: int,
                          beta: float, tau_counts: float,
                          min_w: float, max_w: float) -> torch.Tensor:
    """
    Returns per-pixel LR evidence weight in [min_w, max_w].
    High when LR has evidence (lr_tok > log1p(tau_counts)).
    """
    tau_log = float(np.log1p(tau_counts))
    lr_tok = downsample_to_tokens_logspace(lr_log, token_h, token_w).squeeze(1)  # [B,th,tw]
    w_tok = torch.sigmoid(beta * (lr_tok - tau_log))
    w_tok = w_tok.clamp(min=min_w, max=max_w)
    return w_tok


@torch.no_grad()
def clamp_refined_counts(refined_log: torch.Tensor, proposal_log: torch.Tensor,
                         cap_factor: float, cap_add: float):
    """
    Prevent refined from exploding into very dark regions:
      refined_c <= cap_factor * proposal_c + cap_add
    """
    if cap_factor <= 0 and cap_add <= 0:
        return refined_log

    ref_c = torch.expm1(refined_log).clamp(min=0)
    prop_c = torch.expm1(proposal_log).clamp(min=0)
    cap = prop_c * max(cap_factor, 0.0) + max(cap_add, 0.0)
    ref_c = torch.minimum(ref_c, cap)
    return torch.log1p(ref_c)


# -------------------------
# Args
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--vqgan_ckpt", required=True)
    p.add_argument("--maskgit_ckpt", required=True)

    p.add_argument("--proposal_npy", required=True)
    p.add_argument("--lr_npy", default=None)
    p.add_argument("--hr_npy", default=None)

    p.add_argument("--proposal_coords", default=None)
    p.add_argument("--lr_coords", default=None)
    p.add_argument("--hr_coords", default=None)

    p.add_argument("--out_dir", default="cond_samples_gated_v6")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)

    # VQGAN config
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--codebook_size", type=int, default=4096)

    # MaskGit sampling
    p.add_argument("--timesteps", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.7)

    # LR simulation
    p.add_argument("--thin_frac", type=float, default=0.10)

    # Gating
    p.add_argument("--gate_mode", choices=["lr_zero", "topk_diff"], default="topk_diff")
    p.add_argument("--lr_tau_counts", type=float, default=0.0)
    p.add_argument("--topk_frac", type=float, default=0.05)
    p.add_argument("--topk_only_where_low_evidence", action="store_true")
    p.add_argument("--max_edit_frac", type=float, default=0.05)

    # Init / edits
    p.add_argument("--init_mode", choices=["proposal", "noise", "mixed"], default="proposal")
    p.add_argument("--allow_edit_outside_gate_frac", type=float, default=0.0)

    # Proposal calibration
    p.add_argument("--mass_calibrate_to", choices=["none", "lr", "hr"], default="lr",
                   help="Calibrate proposal scale. Use 'lr' for test-time, 'hr' only for eval runs.")
    p.add_argument("--mass_alpha_min", type=float, default=1e-3)
    p.add_argument("--mass_alpha_max", type=float, default=1e3)

    # Merge
    p.add_argument("--do_merge", action="store_true")
    p.add_argument("--merge_in_counts", action="store_true")
    p.add_argument("--soft_gate", action="store_true")

    p.add_argument("--merge_use_lr_evidence", action="store_true")
    p.add_argument("--merge_evidence_beta", type=float, default=2.0)
    p.add_argument("--merge_lr_tau_counts", type=float, default=1.0)
    p.add_argument("--merge_lr_min_w", type=float, default=0.05)
    p.add_argument("--merge_lr_max_w", type=float, default=1.0)

    # Refined clamp
    p.add_argument("--refined_cap_factor", type=float, default=2.5)
    p.add_argument("--refined_cap_add", type=float, default=2.0)

    # Save arrays
    p.add_argument("--save_npy", action="store_true")

    return p.parse_args()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    proposal_all = torch.from_numpy(ensure_nchw(np.load(args.proposal_npy))).float()
    N = proposal_all.shape[0]

    lr_all = None
    if args.lr_npy and Path(args.lr_npy).exists():
        lr_all = torch.from_numpy(ensure_nchw(np.load(args.lr_npy))).float()
        if lr_all.shape[0] != N:
            raise ValueError("lr_npy N mismatch")

    hr_all = None
    if args.hr_npy and Path(args.hr_npy).exists():
        hr_all = torch.from_numpy(ensure_nchw(np.load(args.hr_npy))).float()
        if hr_all.shape[0] != N:
            raise ValueError("hr_npy N mismatch")

    idx = torch.from_numpy(np.random.permutation(N)[:args.batch_size]).long()
    proposal = proposal_all[idx]
    lr = lr_all[idx] if lr_all is not None else None
    hr = hr_all[idx] if hr_all is not None else None

    if lr is None:
        if hr is not None:
            lr = make_lr_from_log1p(hr, frac=args.thin_frac)
            print("[INFO] lr_npy missing; simulated LR by thinning HR.")
        else:
            lr = make_lr_from_log1p(proposal, frac=args.thin_frac)
            print("[WARN] lr_npy & hr_npy missing; simulated LR by thinning proposal.")

    # Load VQGAN
    vae = VQGanVAE(
        dim=256,
        channels=1,
        layers=2,
        codebook_size=args.codebook_size,
        lookup_free_quantization=False,
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

    with torch.no_grad():
        dummy = torch.zeros(1, 1, args.image_size, args.image_size, device=device)
        _, ids0, _ = vae.encode(dummy)
        token_h, token_w = ids0.shape[1:]
        seq_len = token_h * token_w
    print(f"[INFO] Token grid: {token_h} x {token_w} => seq_len={seq_len}")

    # MaskGit
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

    # Move to device
    proposal, lr, hr = same_device(proposal, lr, hr, device=device)

    # Proposal calibration
    alpha = None
    if args.mass_calibrate_to == "hr":
        if hr is None:
            raise RuntimeError("--mass_calibrate_to hr requires --hr_npy")
        proposal, alpha = calibrate_proposal_to_target(
            proposal, hr,
            mode="hr_sum",
            alpha_min=args.mass_alpha_min,
            alpha_max=args.mass_alpha_max,
        )
    elif args.mass_calibrate_to == "lr":
        proposal, alpha = calibrate_proposal_to_target(
            proposal, lr,
            mode="lr_sum_pos",
            alpha_min=args.mass_alpha_min,
            alpha_max=args.mass_alpha_max,
        )
    if alpha is not None:
        print("[INFO] proposal mass-cal alpha (per-sample):", alpha.detach().cpu().tolist())

    # Gate
    lr_tau_log = float(np.log1p(args.lr_tau_counts))
    gate_hw, low_ev_frac = compute_gate(
        lr_log=lr,
        proposal_log=proposal,
        token_h=token_h,
        token_w=token_w,
        gate_mode=args.gate_mode,
        lr_tau_log=lr_tau_log,
        topk_frac=args.topk_frac,
        topk_only_where_low_evidence=args.topk_only_where_low_evidence,
        max_edit_frac=args.max_edit_frac,
    )
    gate_hw = maybe_add_outside_gate_edits(gate_hw, args.allow_edit_outside_gate_frac, rng=rng)

    print(f"[INFO] low-evidence frac (token space): {low_ev_frac:.4f}")
    print(f"[INFO] gate frac (token space): {float(gate_hw.float().mean().item()):.4f}")

    # Refine
    out = generate_gated_conditional(
        model=maskgit,
        lr_log=lr,
        proposal_log=proposal,
        gate_hw=gate_hw,
        timesteps=args.timesteps,
        temperature=args.temperature,
        init_mode=args.init_mode,
        rng=rng,
        seed=args.seed,
    )
    refined = out["refined_log"]

    # Clamp refined to avoid dark blocks
    refined = clamp_refined_counts(refined, proposal, args.refined_cap_factor, args.refined_cap_add)

    # Merge
    G_up = upsample_gate_to_image(gate_hw, args.image_size, args.image_size)
    if args.soft_gate:
        G_up = F.avg_pool2d(G_up, kernel_size=3, stride=1, padding=1)

    # LR evidence weight (token->pixel)
    lr_w_tok = None
    lr_w_up = None
    gate_eff = G_up
    if args.merge_use_lr_evidence:
        lr_w_tok = lr_evidence_weight_map(
            lr, token_h, token_w,
            beta=args.merge_evidence_beta,
            tau_counts=args.merge_lr_tau_counts,
            min_w=args.merge_lr_min_w,
            max_w=args.merge_lr_max_w,
        )
        lr_w_up = upsample_gate_to_image(lr_w_tok, args.image_size, args.image_size)
        gate_eff = G_up * lr_w_up  # only inject refined where LR has evidence

    if args.do_merge:
        if args.merge_in_counts:
            prop_c = torch.expm1(proposal).clamp(min=0)
            ref_c  = torch.expm1(refined).clamp(min=0)
            final_c = (1.0 - gate_eff) * prop_c + gate_eff * ref_c
            final = torch.log1p(final_c)
        else:
            final = (1.0 - gate_eff) * proposal + gate_eff * refined
    else:
        final = refined

    # -------------------------
    # Save diagnostics
    # -------------------------
    def save_norm(name, x):
        x = x.detach().cpu()
        x = x / (x.max() + 1e-8)
        save_image(x, out_dir / name)

    save_image(G_up.detach().cpu(), out_dir / "gate.png")
    save_image(gate_eff.detach().cpu(), out_dir / "gate_eff.png")
    if lr_w_up is not None:
        save_image(lr_w_up.detach().cpu(), out_dir / "lr_weight.png")

    with torch.no_grad():
        prop_c = torch.expm1(proposal).clamp(min=0)
        ref_c  = torch.expm1(refined).clamp(min=0)
        diff_full = (ref_c - prop_c).abs()
        diff_in_gate = diff_full * gate_eff
    save_norm("diff_full.png", diff_full)
    save_norm("diff_in_gate.png", diff_in_gate)

    # -------------------------
    # Visualization + metrics
    # -------------------------
    import matplotlib.pyplot as plt

    def save_panel_grid(path, lr_img, prop_img, ref_img, final_img, gate_img, gate_eff_img,
                        lr_w_img=None, hr_img=None):
        cols = 8 if (hr_img is not None and lr_w_img is not None) else (7 if hr_img is not None else 6)
        rows = lr_img.shape[0]

        stack = [lr_img, prop_img, ref_img, final_img]
        if hr_img is not None:
            stack.append(hr_img)
        X = torch.cat(stack, dim=0)
        vmax = max(torch.quantile(X.flatten(), 0.95).item(), 1.0)
        vmin = 0.0

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.8))
        if rows == 1:
            axes = np.expand_dims(axes, 0)

        titles = ["LR", "Proposal", "MaskGit refined", "Final (merged)", "Gate", "Gate_eff"]
        if lr_w_img is not None:
            titles.append("LR_weight")
        if hr_img is not None:
            titles.append("HR ref")

        for r in range(rows):
            imgs = [lr_img[r], prop_img[r], ref_img[r], final_img[r], gate_img[r], gate_eff_img[r]]
            if lr_w_img is not None:
                imgs.append(lr_w_img[r])
            if hr_img is not None:
                imgs.append(hr_img[r])

            for c in range(cols):
                ax = axes[r, c]
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(titles[c], fontsize=9)

                t = titles[c]
                data = imgs[c][0].cpu().numpy()

                if t in ("Gate", "Gate_eff", "LR_weight"):
                    ax.imshow(data, cmap="gray", vmin=0, vmax=1)
                else:
                    ax.imshow(data, cmap="Reds", vmin=vmin, vmax=vmax)

        plt.tight_layout()
        fig.savefig(str(path), dpi=200, bbox_inches="tight")
        plt.close(fig)

    lr_vis = lr.detach().cpu()
    prop_vis = proposal.detach().cpu()
    ref_vis = refined.detach().cpu()
    final_vis = final.detach().cpu()
    gate_vis = G_up.detach().cpu()
    gate_eff_vis = gate_eff.detach().cpu()
    lr_w_vis = lr_w_up.detach().cpu() if lr_w_up is not None else None
    hr_vis = hr.detach().cpu() if hr is not None else None

    save_panel_grid(out_dir / "gated_comparison.png",
                    lr_vis, prop_vis, ref_vis, final_vis, gate_vis, gate_eff_vis,
                    lr_w_img=lr_w_vis, hr_img=hr_vis)
    print(f"[OK] Saved: {out_dir/'gated_comparison.png'}")

    # Metrics (masked diagonal band + ref>0)
    def compute_metrics(pred_img, ref_img, name, diag_band=16, only_where_ref_pos=True):
        B, _, H, W = pred_img.shape
        spearman_vals, pearson_vals, ssim_vals, mse_vals, grad_corr_vals = [], [], [], [], []
        for b in range(B):
            pred_np = pred_img[b, 0].numpy()
            ref_np  = ref_img[b, 0].numpy()

            mask = np.isfinite(pred_np) & np.isfinite(ref_np)
            ii = np.arange(H)[:, None]
            jj = np.arange(W)[None, :]
            mask &= (np.abs(ii - jj) <= int(diag_band))
            if only_where_ref_pos:
                mask &= (ref_np > 0)

            pf = pred_np[mask].reshape(-1)
            rf = ref_np[mask].reshape(-1)

            if pf.size < 50 or np.std(pf) < 1e-8 or np.std(rf) < 1e-8:
                spearman_vals.append(0.0); pearson_vals.append(0.0); ssim_vals.append(0.0)
                mse_vals.append(float(np.mean((pf - rf) ** 2)) if pf.size else 0.0)
                grad_corr_vals.append(0.0)
                continue

            sp, _ = spearmanr(pf, rf)
            pr, _ = pearsonr(pf, rf)
            spearman_vals.append(0.0 if np.isnan(sp) else float(sp))
            pearson_vals.append(0.0 if np.isnan(pr) else float(pr))
            mse_vals.append(float(np.mean((pf - rf) ** 2)))

            # Sobel grad corr
            pgx = sobel(pred_np, axis=1); pgy = sobel(pred_np, axis=0)
            rgx = sobel(ref_np, axis=1);  rgy = sobel(ref_np, axis=0)
            pg = np.sqrt(pgx**2 + pgy**2); rg = np.sqrt(rgx**2 + rgy**2)
            pgf = pg[mask].reshape(-1); rgf = rg[mask].reshape(-1)
            if pgf.size >= 10 and np.std(pgf) > 1e-8 and np.std(rgf) > 1e-8:
                gc, _ = pearsonr(pgf, rgf)
                grad_corr_vals.append(0.0 if np.isnan(gc) else float(gc))
            else:
                grad_corr_vals.append(0.0)

            ref_masked = ref_np[mask]
            vmin = 0.0
            vmax = np.percentile(ref_masked, 99.5) if ref_masked.size else 1.0
            vmax = max(float(vmax), 1e-6)

            pred_norm = np.clip((pred_np - vmin) / (vmax - vmin), 0.0, 1.0)
            ref_norm  = np.clip((ref_np  - vmin) / (vmax - vmin), 0.0, 1.0)
            pred_norm[~mask] = 0.0
            ref_norm[~mask]  = 0.0
            s = ssim(ref_norm, pred_norm, data_range=1.0)
            ssim_vals.append(0.0 if np.isnan(s) else float(s))

        print(f"\n[METRICS] {name} vs HR:")
        print(f"  Pearson : {np.mean(pearson_vals):.4f}")
        print(f"  Spearman: {np.mean(spearman_vals):.4f}")
        print(f"  SSIM    : {np.mean(ssim_vals):.4f}")
        print(f"  MSE     : {np.mean(mse_vals):.6f}")
        print(f"  GradCorr: {np.mean(grad_corr_vals):.4f}")
        return dict(pearson=pearson_vals, spearman=spearman_vals, ssim=ssim_vals, mse=mse_vals, grad_corr=grad_corr_vals)

    if hr_vis is not None:
        m_lr   = compute_metrics(lr_vis,   hr_vis, "LR")
        m_prop = compute_metrics(prop_vis, hr_vis, "Proposal")
        m_ref  = compute_metrics(ref_vis,  hr_vis, "MaskGit")
        m_fin  = compute_metrics(final_vis,hr_vis, "Final")

        with open(out_dir / "metrics_vs_hr.txt", "w") as f:
            f.write("# Method\tPearson\tSpearman\tMSE\tSSIM\tGradCorr\n")
            def line(name, m):
                f.write(f"{name}\t{np.mean(m['pearson']):.6f}\t{np.mean(m['spearman']):.6f}\t"
                        f"{np.mean(m['mse']):.6f}\t{np.mean(m['ssim']):.6f}\t{np.mean(m['grad_corr']):.6f}\n")
            line("LR", m_lr); line("Proposal", m_prop); line("MaskGit", m_ref); line("Final", m_fin)

    if args.save_npy:
        np.save(out_dir / "idx.npy", idx.cpu().numpy())
        np.save(out_dir / "proposal_log1p.npy", proposal.detach().cpu().numpy())
        np.save(out_dir / "lr_log1p.npy", lr.detach().cpu().numpy())
        np.save(out_dir / "refined_log1p.npy", refined.detach().cpu().numpy())
        np.save(out_dir / "final_log1p.npy", final.detach().cpu().numpy())
        np.save(out_dir / "gate_hw.npy", gate_hw.detach().cpu().numpy())
        np.save(out_dir / "gate_up.npy", G_up.detach().cpu().numpy())
        np.save(out_dir / "gate_eff.npy", gate_eff.detach().cpu().numpy())
        if lr_w_up is not None:
            np.save(out_dir / "lr_weight.npy", lr_w_up.detach().cpu().numpy())
        if alpha is not None:
            np.save(out_dir / "mass_alpha.npy", alpha.detach().cpu().numpy())
        print(f"[OK] Saved .npy outputs to: {out_dir}")


if __name__ == "__main__":
    main()
