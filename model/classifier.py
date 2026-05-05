"""
Auxiliary classifier for classifier-guided diffusion.

A small ResNet-style network trained to predict class labels from noisy
images/latents at arbitrary timesteps t. Its gradients are used during
sampling to steer the diffusion process toward a target class.

Reference: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding for timesteps."""
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb_scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)


class ClassifierResBlock(nn.Module):
    """Simple residual block with timestep conditioning for the classifier."""
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, down: bool = False):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        # Timestep projection
        self.emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_ch),
        )

        # Skip connection
        if in_ch != out_ch:
            self.skip = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.skip = nn.Identity()

        # Downsampling
        self.down = nn.AvgPool2d(2) if down else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add timestep embedding
        emb_out = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)
        h = h + emb_out

        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)

        h = h + self.skip(x)
        h = self.down(h)

        return h


class NoisyClassifier(nn.Module):
    """
    Classifier for noisy images/latents at arbitrary timesteps.

    Architecture:
        Conv3x3 → [ResBlock, ResBlock(down)] x 3 → AdaptiveAvgPool → Linear

    The classifier is conditioned on the timestep t so it can handle
    varying noise levels during the diffusion process.

    During sampling, ∇_{x_t} log p(y|x_t) is used for classifier guidance.
    """
    def __init__(
        self,
        in_channels: int = 4,
        hidden_channels: int = 64,
        num_classes: int = 10,
    ):
        super().__init__()
        emb_dim = hidden_channels * 4  # 256

        # Timestep embedding
        self.time_dim = hidden_channels
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_channels, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # Initial conv
        self.input_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)

        # ResNet backbone
        ch = hidden_channels  # 64
        self.blocks = nn.ModuleList([
            # 32x32 → 32x32
            ClassifierResBlock(ch, ch, emb_dim, down=False),
            # 32x32 → 16x16
            ClassifierResBlock(ch, ch * 2, emb_dim, down=True),
            # 16x16 → 16x16
            ClassifierResBlock(ch * 2, ch * 2, emb_dim, down=False),
            # 16x16 → 8x8
            ClassifierResBlock(ch * 2, ch * 4, emb_dim, down=True),
            # 8x8 → 8x8
            ClassifierResBlock(ch * 4, ch * 4, emb_dim, down=False),
        ])

        # Output head
        self.output_norm = nn.GroupNorm(32, ch * 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch * 4, num_classes)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noisy input (B, in_channels, H, W)
            timesteps: (B,) integer timesteps
        Returns:
            Class logits (B, num_classes)
        """
        # Timestep embedding
        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        # Forward through backbone
        h = self.input_conv(x)
        for block in self.blocks:
            h = block(h, t_emb)

        # Classification head
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.pool(h).flatten(1)
        logits = self.classifier(h)

        return logits
