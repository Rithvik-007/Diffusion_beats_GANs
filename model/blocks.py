"""
Core building blocks for the ADM UNet.

Implements:
- AdaGN: Adaptive Group Normalization (timestep + class conditioning)
- BigGANResBlock: Residual block with BigGAN-style up/downsampling
- Upsample / Downsample modules

Reference: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    """
    2x spatial upsampling: nearest-neighbor interpolation followed by a 3x3 conv.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    2x spatial downsampling via strided 3x3 convolution.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AdaGN(nn.Module):
    """
    Adaptive Group Normalization.

    Injects timestep + class conditioning into residual blocks by predicting
    per-channel scale (w_s) and shift (w_b) from the conditioning embedding.

    Formula:
        [w_s, w_b] = Linear(SiLU(emb))
        AdaGN(h, emb) = w_s * GroupNorm(h) + w_b

    Note: GroupNorm uses affine=False because AdaGN provides its own scale/shift.
    """
    def __init__(self, num_groups: int, channels: int, emb_dim: int):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels, affine=False)
        # Project conditioning to scale + shift
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * channels),
        )

    def forward(self, h: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Feature tensor (B, C, H, W)
            emb: Conditioning embedding (B, emb_dim)
        Returns:
            Normalized and conditioned features (B, C, H, W)
        """
        h = self.norm(h)

        # Predict scale and shift from embedding
        params = self.proj(emb)                   # (B, 2*C)
        params = params.unsqueeze(-1).unsqueeze(-1)  # (B, 2*C, 1, 1)
        w_s, w_b = params.chunk(2, dim=1)          # Each: (B, C, 1, 1)

        return w_s * h + w_b


class BigGANResBlock(nn.Module):
    """
    BigGAN-style residual block with up/downsampling integrated inside the block.

    Architecture:
        Main path:   AdaGN₁ → SiLU → [Up/Down] → Conv3x3 → AdaGN₂ → SiLU → Dropout → Conv3x3
        Skip path:   [Up/Down] → [Conv1x1 if channel change]
        Output:      main + skip

    Key: up/downsampling is applied to BOTH the main path and the skip connection,
    matching the original BigGAN design.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        dropout: float = 0.0,
        up: bool = False,
        down: bool = False,
        num_groups: int = 32,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down

        # --- Main path ---
        self.norm1 = AdaGN(num_groups, in_channels, emb_dim)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = AdaGN(num_groups, out_channels, emb_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # --- Up/Down sampling layers (shared between main and skip) ---
        if up:
            self.resample = Upsample(in_channels)
            self.skip_resample = Upsample(in_channels)
        elif down:
            self.resample = Downsample(in_channels)
            self.skip_resample = Downsample(in_channels)
        else:
            self.resample = nn.Identity()
            self.skip_resample = nn.Identity()

        # --- Skip connection ---
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_proj = nn.Identity()

        # NOTE: No zero-init on conv2. When used with LoRA (frozen base),
        # zero-init would make every ResBlock's main path permanently dead.

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (B, in_channels, H, W)
            emb: Conditioning embedding (B, emb_dim)
        Returns:
            Output features (B, out_channels, H', W')
            where H', W' may differ from H, W if up/down=True
        """
        # --- Main path ---
        h = self.norm1(x, emb)
        h = F.silu(h)
        h = self.resample(h)     # Up/down before first conv
        h = self.conv1(h)

        h = self.norm2(h, emb)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # --- Skip path ---
        skip = self.skip_resample(x)
        skip = self.skip_proj(skip)

        return h + skip
