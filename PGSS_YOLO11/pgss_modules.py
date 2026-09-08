"""
PGSS-YOLO11s custom modules, written to integrate genuinely with Ultralytics.

Design constraints so they wire cleanly into a YAML via parse_model's channel
passthrough (`c2 = ch[f]` for non-base modules):
  * every module is CHANNEL-PRESERVING (out_channels == in_channels)
  * every module is SINGLE-INPUT and takes its channel count `c` as the first arg
  * registered into ultralytics.nn.tasks globals so a custom YAML can name them

Modules:
  GLSA  — Global-Local Spatial Attention (SFAM): fixed, efficient, channel-preserving
  AWFF  — Adaptive Weighted Feature Fusion refinement (AFFM idea), post-concat, channel-preserving
  DEConv— Detail-Enhanced Convolution (DE-SCSH detail block), channel-preserving
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GlobalSpatialAttention(nn.Module):
    """Efficient global attention: channel-affinity (linear) attention — O(N·C²),
    no N×N matrix, so it is memory-safe on large feature maps. Long-range context
    is captured through the C×C channel affinity, then re-projected spatially."""

    def __init__(self, c: int):
        super().__init__()
        self.q = nn.Conv2d(c, c, 1, bias=False)
        self.k = nn.Conv2d(c, c, 1, bias=False)
        self.v = nn.Conv2d(c, c, 1, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)
        self.scale = c ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.q(x).flatten(2)                     # (B, C, N)
        k = self.k(x).flatten(2)                     # (B, C, N)
        v = self.v(x).flatten(2)                     # (B, C, N)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=-1)  # (B, C, C)
        out = (attn @ v).view(B, C, H, W)            # (B, C, N) -> (B,C,H,W)
        return x + self.proj(out)                    # residual


class LocalSpatialAttention(nn.Module):
    """Fine-grained local attention via cascaded depthwise convs + sigmoid gate."""

    def __init__(self, c: int, n_layers: int = 3):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers += [nn.Conv2d(c, c, 1, bias=False),
                       nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False)]
        self.fc = nn.Sequential(*layers)
        self.gate = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x):
        att = torch.sigmoid(self.gate(self.fc(x) + x))
        return att * x + x


class GLSA(nn.Module):
    """Global-Local Spatial Attention (SFAM). CHANNEL-PRESERVING: splits channels
    in half, one half through global attention, one through local attention, then
    concatenates back to `c` and fuses with a 1x1 (output channels == input)."""

    def __init__(self, c: int):
        super().__init__()
        assert c % 2 == 0, "GLSA needs even channels"
        half = c // 2
        self.gsa = GlobalSpatialAttention(half)
        self.lsa = LocalSpatialAttention(half)
        self.fuse = nn.Conv2d(c, c, 1, bias=False)   # keep channels (fixed: was c//2)
        self.bn = nn.BatchNorm2d(c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        out = torch.cat([self.gsa(x1), self.lsa(x2)], dim=1)
        return self.act(self.bn(self.fuse(out))) + x   # residual keeps it stable


class AWFF(nn.Module):
    """Adaptive Weighted Feature-fusion refinement (AFFM idea), applied AFTER a
    stock Concat. Learns per-channel adaptive gating (SE-style) then a 3x3 CBS,
    channel-preserving. Genuinely re-weights fused features instead of treating
    the concatenated halves equally."""

    def __init__(self, c: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, max(8, c // 8), 1), nn.SiLU(inplace=True),
            nn.Conv2d(max(8, c // 8), c, 1), nn.Sigmoid(),
        )
        self.post = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c), nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.post(x * self.gate(x))


class DEConv(nn.Module):
    """Detail-Enhanced Convolution (DE-SCSH detail block). A central 3x3 conv plus
    four 1x1 'detail' branches placed at kernel corners, averaged; GroupNorm + SiLU.
    CHANNEL-PRESERVING (c -> c)."""

    def __init__(self, c: int, groups: int = 16):
        super().__init__()
        self.main = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.d_tl = nn.Conv2d(c, c, 1, bias=False)
        self.d_tr = nn.Conv2d(c, c, 1, bias=False)
        self.d_bl = nn.Conv2d(c, c, 1, bias=False)
        self.d_br = nn.Conv2d(c, c, 1, bias=False)
        self.norm = nn.GroupNorm(max(1, min(groups, c)), c)
        self.act = nn.SiLU(inplace=True)

    def _as_3x3(self, conv1x1, pos):
        w = torch.zeros_like(self.main.weight)
        r, cc = {"tl": (0, 0), "tr": (0, 2), "bl": (2, 0), "br": (2, 2)}[pos]
        w[:, :, r, cc] = conv1x1.weight[:, :, 0, 0]
        return w

    def forward(self, x):
        out = (self.main(x)
               + F.conv2d(x, self._as_3x3(self.d_tl, "tl"), padding=1)
               + F.conv2d(x, self._as_3x3(self.d_tr, "tr"), padding=1)
               + F.conv2d(x, self._as_3x3(self.d_bl, "bl"), padding=1)
               + F.conv2d(x, self._as_3x3(self.d_br, "br"), padding=1)) / 5.0
        return self.act(self.norm(out))


def register_pgss_modules():
    """Register PGSS modules into Ultralytics namespaces so custom YAMLs can name
    them and torch.load can resolve them."""
    import ultralytics.nn.tasks as tasks
    import ultralytics.nn.modules as modules
    for cls in (GLSA, AWFF, DEConv, GlobalSpatialAttention, LocalSpatialAttention):
        # parse_model resolves module names via globals() of the tasks module,
        # so injecting into tasks.__dict__ is what makes YAML names work.
        tasks.__dict__[cls.__name__] = cls
        modules.__dict__[cls.__name__] = cls


register_pgss_modules()
