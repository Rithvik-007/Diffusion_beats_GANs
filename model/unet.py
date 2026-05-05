"""
ADM-style UNet for diffusion models.

Implements the full architecture from "Diffusion Models Beat GANs on Image Synthesis":
- Sinusoidal timestep embedding + class embedding
- Encoder with BigGAN ResBlocks + multi-head self-attention
- Bottleneck with attention
- Decoder with skip connections
- Optional gradient checkpointing for VRAM savings

Reference: Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .blocks import BigGANResBlock, Downsample, Upsample
from .attention import SelfAttention


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional embedding for timesteps.

    Args:
        timesteps: (B,) tensor of integer timesteps
        dim: Embedding dimension (must be even)
    Returns:
        (B, dim) embedding tensor
    """
    assert dim % 2 == 0, f"Embedding dim must be even, got {dim}"
    half_dim = dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb_scale)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, dim)
    return emb


class TimestepEmbedding(nn.Module):
    """
    Timestep + class label conditioning.

    Timestep: sinusoidal(128) → Linear(128, 512) → SiLU → Linear(512, 512)
    Class:    nn.Embedding(num_classes, 512), added to timestep embedding
    """
    def __init__(self, base_channels: int, num_classes: int):
        super().__init__()
        time_dim = base_channels  # 128
        emb_dim = base_channels * 4  # 512

        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.emb_dim = emb_dim
        self.time_dim = time_dim

    def forward(self, timesteps: torch.Tensor, class_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: (B,) integer timesteps
            class_labels: (B,) integer class labels
        Returns:
            (B, emb_dim) combined conditioning embedding
        """
        t_emb = sinusoidal_embedding(timesteps, self.time_dim)  # (B, 128)
        t_emb = self.time_mlp(t_emb)                           # (B, 512)
        c_emb = self.class_emb(class_labels)                    # (B, 512)
        return t_emb + c_emb


class UNet(nn.Module):
    """
    ADM-style UNet with BigGAN residual blocks and multi-head self-attention.

    The encoder/decoder are built as flat lists of modules. Each encoder module
    produces a skip connection, and each decoder module consumes one (in reverse).

    Encoder structure (per level):
        [ResBlock (+Attn)] x num_res_blocks, then Downsample (except last level)

    Decoder structure (per level):
        [ResBlock (+Attn, +skip concat)] x (num_res_blocks + 1), then Upsample (except last level)
    """
    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        base_channels: int = 128,
        channel_mult: tuple = (1, 2, 3, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple = (32, 16, 8),
        head_channels: int = 64,
        num_classes: int = 10,
        dropout: float = 0.0,
        input_size: int = 32,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.input_size = input_size
        num_groups = 32

        # Channel sizes at each level
        channels = [base_channels * m for m in channel_mult]
        emb_dim = base_channels * 4  # 512

        # --- Timestep + class conditioning ---
        self.time_embed = TimestepEmbedding(base_channels, num_classes)

        # --- Input convolution ---
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # ============================================================
        # ENCODER — build flat list, track skip connection channels
        # ============================================================
        self.encoder_blocks = nn.ModuleList()
        skip_channels = [base_channels]  # First skip is after input_conv

        current_ch = base_channels
        current_res = input_size

        for level, ch in enumerate(channels):
            for block_idx in range(num_res_blocks):
                block_in_ch = current_ch
                layers = [BigGANResBlock(block_in_ch, ch, emb_dim, dropout, num_groups=num_groups)]

                if current_res in attention_resolutions:
                    layers.append(SelfAttention(ch, head_channels, num_groups))

                self.encoder_blocks.append(nn.ModuleList(layers))
                current_ch = ch
                skip_channels.append(current_ch)

            # Downsample (except at the last level)
            if level < len(channels) - 1:
                self.encoder_blocks.append(nn.ModuleList([Downsample(current_ch)]))
                skip_channels.append(current_ch)
                current_res //= 2

        # ============================================================
        # BOTTLENECK
        # ============================================================
        self.bottleneck = nn.ModuleList([
            BigGANResBlock(current_ch, current_ch, emb_dim, dropout, num_groups=num_groups),
            SelfAttention(current_ch, head_channels, num_groups),
            BigGANResBlock(current_ch, current_ch, emb_dim, dropout, num_groups=num_groups),
        ])

        # ============================================================
        # DECODER — mirror of encoder, consumes skip connections in reverse
        # ============================================================
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(len(channels))):
            ch = channels[level]

            # Resolution at this decoder level
            if level == len(channels) - 1:
                dec_res = current_res
            else:
                dec_res = input_size // (2 ** level)

            for block_idx in range(num_res_blocks + 1):
                # Pop skip channel (reverse order)
                skip_ch = skip_channels.pop()
                block_in_ch = current_ch + skip_ch  # Concatenated skip

                layers = [BigGANResBlock(block_in_ch, ch, emb_dim, dropout, num_groups=num_groups)]

                if dec_res in attention_resolutions:
                    layers.append(SelfAttention(ch, head_channels, num_groups))

                self.decoder_blocks.append(nn.ModuleList(layers))
                current_ch = ch

            # Upsample (except at the first encoder level = last decoder level)
            if level > 0:
                self.decoder_blocks.append(nn.ModuleList([
                    BigGANResBlock(current_ch, current_ch, emb_dim, dropout, up=True, num_groups=num_groups)
                ]))
                current_res *= 2

        # Verify all skips consumed
        assert len(skip_channels) == 0, f"Leftover skip channels: {skip_channels}"

        # --- Output ---
        self.output_norm = nn.GroupNorm(num_groups, current_ch)
        self.output_conv = nn.Conv2d(current_ch, out_channels, kernel_size=3, padding=1)

        # NOTE: No zero-init on output conv. Zero-init is standard for pretrained
        # diffusion models but fatal when training from scratch with LoRA, since
        # the frozen output conv would permanently output zeros.

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy input (B, in_channels, H, W)
            timesteps: (B,) integer timesteps
            class_labels: (B,) integer class labels
        Returns:
            Predicted noise (B, out_channels, H, W)
        """
        # Conditioning embedding
        emb = self.time_embed(timesteps, class_labels)  # (B, 512)

        # Input conv
        h = self.input_conv(x)

        # --- Encoder with skip connections ---
        skips = [h]  # First skip is after input_conv
        for layers in self.encoder_blocks:
            if isinstance(layers[0], Downsample):
                h = layers[0](h)
            else:
                resblock = layers[0]
                if self.use_gradient_checkpointing and self.training:
                    h = checkpoint(resblock, h, emb, use_reentrant=False)
                else:
                    h = resblock(h, emb)

                # Apply attention if present
                for layer in layers[1:]:
                    h = layer(h)

            skips.append(h)

        # --- Bottleneck ---
        if self.use_gradient_checkpointing and self.training:
            h = checkpoint(self.bottleneck[0], h, emb, use_reentrant=False)
        else:
            h = self.bottleneck[0](h, emb)
        h = self.bottleneck[1](h)  # Attention
        if self.use_gradient_checkpointing and self.training:
            h = checkpoint(self.bottleneck[2], h, emb, use_reentrant=False)
        else:
            h = self.bottleneck[2](h, emb)

        # --- Decoder with skip connections ---
        for layers in self.decoder_blocks:
            if isinstance(layers[0], BigGANResBlock) and layers[0].up:
                # Upsample block (no skip connection)
                if self.use_gradient_checkpointing and self.training:
                    h = checkpoint(layers[0], h, emb, use_reentrant=False)
                else:
                    h = layers[0](h, emb)
            else:
                # Regular decoder ResBlock — pop and concat skip
                skip = skips.pop()
                h = torch.cat([h, skip], dim=1)

                resblock = layers[0]
                if self.use_gradient_checkpointing and self.training:
                    h = checkpoint(resblock, h, emb, use_reentrant=False)
                else:
                    h = resblock(h, emb)

                # Apply attention if present
                for layer in layers[1:]:
                    h = layer(h)

        # --- Output ---
        h = self.output_norm(h)
        h = F.silu(h)
        h = self.output_conv(h)

        return h
