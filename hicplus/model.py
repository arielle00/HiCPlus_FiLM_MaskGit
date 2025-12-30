import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# =========================
# Original HiCPlus constants
# =========================
conv2d1_filters_numbers = 8
conv2d1_filters_size = 9
conv2d2_filters_numbers = 8
conv2d2_filters_size = 1
conv2d3_filters_numbers = 1
conv2d3_filters_size = 5


# =========================
# Original HiCPlus (kept intact)
# =========================
class Net(nn.Module):
    def __init__(self, D_in, D_out):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, conv2d1_filters_numbers, conv2d1_filters_size)
        self.conv2 = nn.Conv2d(conv2d1_filters_numbers, conv2d2_filters_numbers, conv2d2_filters_size)
        self.conv3 = nn.Conv2d(conv2d2_filters_numbers, 1, conv2d3_filters_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x


# =========================
# FiLM + MaskGIT extensions
# =========================

def _same_padding(k, d=1):
    return (k + (k - 1) * (d - 1)) // 2


def _center_crop2d(x, out_h, out_w):
    _, _, H, W = x.shape
    y0 = (H - out_h) // 2
    x0 = (W - out_w) // 2
    return x[:, :, y0:y0 + out_h, x0:x0 + out_w]


@torch.no_grad()
def _diagonal_distance_map(n, device):
    i = torch.arange(n, device=device).float()
    d = (i[None, :] - i[:, None]).abs()
    return d / d.max().clamp(min=1.0)


_ACT = nn.SiLU if hasattr(nn, "SiLU") else nn.ReLU


# =========================
# FiLM blocks (MaskGIT-gated)
# =========================
class _FiLM(nn.Module):
    """
    Feature-wise linear modulation with external scalar gate alpha.
    """
    def __init__(self, cond_channels, out_channels):
        super().__init__()
        self.gamma = nn.Conv2d(cond_channels, out_channels, kernel_size=1)
        self.beta  = nn.Conv2d(cond_channels, out_channels, kernel_size=1)

    def forward(self, x, cond, alpha=None):
        g = self.gamma(cond)
        b = self.beta(cond)
        if alpha is not None:
            g = g * alpha
            b = b * alpha
        return (1 + g) * x + b


class _CondResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_ch, k=3, dilation=1, groups=8):
        super().__init__()
        pad = _same_padding(k, dilation)

        self.norm1 = nn.GroupNorm(groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)

        self.norm2 = nn.GroupNorm(groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=pad, dilation=dilation)

        self.film1 = _FiLM(cond_ch, in_ch)
        self.film2 = _FiLM(cond_ch, out_ch)

        self.proj = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)
        self.act = _ACT()

    def forward(self, x, cond, alpha=None):
        y = self.film1(self.norm1(x), cond, alpha)
        y = self.conv1(self.act(y))
        y = self.film2(self.norm2(y), cond, alpha)
        y = self.conv2(self.act(y))
        return self.act(self.proj(x) + y)


class _LREncoder(nn.Module):
    """
    Shallow LR encoder used only for FiLM conditioning.
    """
    def __init__(self, in_ch=1, base_ch=32, depth=3):
        super().__init__()
        layers = []
        ch = in_ch
        for _ in range(depth):
            layers += [
                nn.Conv2d(ch, base_ch, 3, padding=1), _ACT(),
                nn.Conv2d(base_ch, base_ch, 3, padding=1), _ACT()
            ]
            ch = base_ch
        self.net = nn.Sequential(*layers)
        self.out_ch = base_ch

    def forward(self, x):
        return self.net(x)


# =========================
# HiCPlus + FiLM + MaskGIT
# =========================
class HiCPlusFiLM(nn.Module):
    def __init__(
        self,
        width=64,
        depth=8,
        dilations=(1, 1, 2, 2, 4, 4, 8, 8),
        use_aux=True,
        predict_residual=True,
        sym_enforce=True,
        nonneg=True,
        groups=8
    ):
        super().__init__()
        self.use_aux = use_aux
        self.predict_residual = predict_residual
        self.sym_enforce = sym_enforce
        self.nonneg = nonneg

        stem_in = 1 + (1 if use_aux else 0)
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, width, 3, padding=1), _ACT(),
            nn.Conv2d(width, width, 3, padding=1), _ACT()
        )

        self.lr_encoder = _LREncoder(in_ch=1, base_ch=width // 2, depth=3)
        cond_ch = self.lr_encoder.out_ch

        self.blocks = nn.ModuleList([
            _CondResBlock(width, width, cond_ch, k=3,
                          dilation=(dilations[i] if i < len(dilations) else 1),
                          groups=groups)
            for i in range(depth)
        ])

        self.head = nn.Sequential(
            nn.GroupNorm(groups, width),
            _ACT(),
            nn.Conv2d(width, 1, 3, padding=1)
        )

        self.softplus = nn.Softplus()

    def forward(self, x, maskgit_map=None):
        """
        x: [B,1,H,W]
        maskgit_map: [B,1,f,f] or None
        """
        B, _, H, W = x.shape

        # --- MaskGIT scalar gate ---
        alpha = None
        if maskgit_map is not None:
            alpha = maskgit_map.mean(dim=[2, 3], keepdim=True)  # [B,1,1,1]

        # --- auxiliary diagonal channel ---
        if self.use_aux:
            dmap = _diagonal_distance_map(H, x.device)
            dmap = dmap.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
            z = torch.cat([x, dmap], dim=1)
        else:
            z = x

        z = self.stem(z)
        cond = self.lr_encoder(x)

        for blk in self.blocks:
            z = blk(z, cond, alpha)

        y = self.head(z)

        if self.predict_residual:
            y = x + y
        if self.sym_enforce:
            y = 0.5 * (y + y.transpose(-1, -2))
        if self.nonneg:
            y = self.softplus(y)

        return y


class FiLMNet(nn.Module):
    """
    Drop-in replacement matching original Net(D_in, D_out) behavior.
    """
    def __init__(
        self,
        D_in,
        D_out,
        width=64,
        depth=8,
        dilations=(1, 1, 2, 2, 4, 4, 8, 8),
        use_aux=True,
        predict_residual=True,
        sym_enforce=True,
        nonneg=True,
        groups=8
    ):
        super().__init__()
        self.core = HiCPlusFiLM(
            width=width,
            depth=depth,
            dilations=dilations,
            use_aux=use_aux,
            predict_residual=predict_residual,
            sym_enforce=sym_enforce,
            nonneg=nonneg,
            groups=groups
        )

        self._shrink = int(D_in) - int(D_out)
        if self._shrink < 0:
            raise ValueError("D_out must be <= D_in")

    def forward(self, x, maskgit_map=None):
        y = self.core(x, maskgit_map)
        if self._shrink > 0:
            B, C, H, W = y.shape
            y = _center_crop2d(y, H - self._shrink, W - self._shrink)
        return y
