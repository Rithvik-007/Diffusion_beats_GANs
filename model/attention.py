"""
Multi-head self-attention module for the ADM UNet.

Uses a fixed head width of 64 channels (not a fixed number of heads),
applied at resolutions 32x32, 16x16, and 8x8.

Reference: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi-head self-attention with fixed 64-channel head width.

    Architecture:
        GroupNorm → reshape (B,C,H,W) → (B,H*W,C) → QKV projection
        → split into num_heads (C//64) → scaled_dot_product_attention
        → output projection → reshape back → residual connection

    The Q, K, V, and output projections are the layers where LoRA
    will be injected.
    """
    def __init__(self, channels: int, head_channels: int = 64, num_groups: int = 32):
        super().__init__()
        assert channels % head_channels == 0, (
            f"channels ({channels}) must be divisible by head_channels ({head_channels})"
        )
        self.channels = channels
        self.head_channels = head_channels
        self.num_heads = channels // head_channels

        # Pre-norm
        self.norm = nn.GroupNorm(num_groups, channels)

        # QKV projections (separate for LoRA injection compatibility)
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)

        # Output projection
        self.out_proj = nn.Linear(channels, channels)

        # NOTE: No zero-init on out_proj. When used with LoRA (frozen base),
        # zero-init would make attention permanently output zero.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
        Returns:
            Output tensor (B, C, H, W) with residual connection
        """
        B, C, H, W = x.shape
        residual = x

        # Pre-norm
        h = self.norm(x)

        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        h = h.view(B, C, H * W).permute(0, 2, 1)  # (B, N, C) where N = H*W

        # QKV projections
        q = self.q_proj(h)  # (B, N, C)
        k = self.k_proj(h)
        v = self.v_proj(h)

        # Reshape for multi-head attention: (B, N, C) -> (B, num_heads, N, head_channels)
        q = q.view(B, H * W, self.num_heads, self.head_channels).permute(0, 2, 1, 3)
        k = k.view(B, H * W, self.num_heads, self.head_channels).permute(0, 2, 1, 3)
        v = v.view(B, H * W, self.num_heads, self.head_channels).permute(0, 2, 1, 3)

        # Scaled dot-product attention (uses FlashAttention when available)
        attn_out = F.scaled_dot_product_attention(q, k, v)

        # Reshape back: (B, num_heads, N, head_channels) -> (B, N, C)
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, H * W, C)

        # Output projection
        attn_out = self.out_proj(attn_out)

        # Reshape back to spatial: (B, N, C) -> (B, C, H, W)
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)

        # Residual connection
        return residual + attn_out
