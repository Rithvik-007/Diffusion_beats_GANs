"""
Centralized configuration for the ADM Latent Diffusion Model.
All hyperparameters in one place for easy tuning.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ModelConfig:
    """ADM UNet architecture configuration."""
    image_size: int = 256               # CIFAR-10 upscaled to 256x256
    latent_size: int = 32               # 256 // 8 (SD-VAE 8x downscale)
    in_channels: int = 4                # 4 for latent mode, 3 for pixel mode
    out_channels: int = 4              # Must match in_channels
    base_channels: int = 128           # Scaled down from 256 for 6GB VRAM
    channel_mult: Tuple[int, ...] = (1, 2, 3, 4)  # -> 128, 256, 384, 512
    num_res_blocks: int = 2            # Residual blocks per resolution level
    attention_resolutions: Tuple[int, ...] = (32, 16, 8)  # Multi-resolution attention
    head_channels: int = 64            # Fixed head width per ADM spec
    num_classes: int = 10              # CIFAR-10 classes
    dropout: float = 0.0
    use_gradient_checkpointing: bool = True


@dataclass
class LoRAConfig:
    """Low-Rank Adaptation configuration."""
    rank: int = 16
    alpha: int = 16                    # Scaling: alpha / rank = 1.0


@dataclass
class ClassifierConfig:
    """Auxiliary noisy-image classifier configuration."""
    in_channels: int = 4               # 4 for latent, 3 for pixel
    hidden_channels: int = 64
    num_classes: int = 10
    learning_rate: float = 3e-4
    num_epochs: int = 20
    batch_size: int = 32


@dataclass
class DiffusionConfig:
    """DDPM diffusion process configuration."""
    timesteps: int = 1000
    schedule: str = "linear"           # "linear" or "cosine"
    guidance_scale: float = 2.0        # Classifier guidance strength


@dataclass
class TrainConfig:
    """Training loop configuration."""
    batch_size: int = 4                # For latent mode; use 1-2 for pixel
    learning_rate: float = 1e-4
    num_epochs: int = 50
    use_amp: bool = True               # FP16 mixed precision
    use_8bit_adam: bool = True          # bitsandbytes 8-bit Adam
    gradient_checkpointing: bool = True
    log_interval: int = 50             # Print loss every N steps
    sample_interval: int = 500         # Generate samples every N steps
    save_interval: int = 1000          # Save checkpoint every N steps
    num_workers: int = 0               # DataLoader workers


@dataclass
class VAEConfig:
    """Pre-trained StabilityAI SD-VAE configuration."""
    model_id: str = "stabilityai/sd-vae-ft-mse"
    scale_factor: float = 0.18215      # Latent scaling factor


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset: str = "cifar10"
    images_per_class: int = 1000       # Subset size
    data_dir: str = "data"
    latent_dir: str = "data/latents"
    pixel_dir: str = "data/pixels"
