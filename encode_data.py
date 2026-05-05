"""
CIFAR-10 Data Encoding Script.

Pre-encodes CIFAR-10 images into:
  1. Upscaled 256x256 pixel tensors (for pixel baseline training)
  2. 32x32 VAE latent tensors (for latent-LoRA training)

This runs ONCE before training. After encoding, training scripts
load directly from .pt files without needing the VAE.

Usage:
    python encode_data.py [--images_per_class 1000]
"""

import os
import sys
import time
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from diffusers import AutoencoderKL
from tqdm import tqdm
import numpy as np


def get_cifar10_subset(images_per_class: int = 1000, data_dir: str = "data"):
    """
    Download CIFAR-10 and select a balanced subset.

    Args:
        images_per_class: Number of images per class
        data_dir: Root directory for CIFAR-10 download

    Returns:
        Subset of CIFAR-10 dataset with raw PIL images
    """
    # Download CIFAR-10
    dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=None  # Raw PIL images
    )

    # Select balanced subset
    class_counts = {}
    selected_indices = []

    for idx, (_, label) in enumerate(dataset):
        if label not in class_counts:
            class_counts[label] = 0
        if class_counts[label] < images_per_class:
            selected_indices.append(idx)
            class_counts[label] += 1

        # Stop early if we have enough
        if all(c >= images_per_class for c in class_counts.values()):
            break

    subset = Subset(dataset, selected_indices)
    print(f"Selected {len(subset)} images ({images_per_class} per class, "
          f"{len(class_counts)} classes)")
    return subset


def encode_pixels(subset, pixel_dir: str, image_size: int = 256, batch_size: int = 64):
    """
    Resize CIFAR-10 images to 256x256 and save as pixel tensors.

    Args:
        subset: CIFAR-10 subset
        pixel_dir: Output directory for pixel .pt files
        image_size: Target image size
        batch_size: Processing batch size
    """
    os.makedirs(pixel_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])

    all_pixels = []
    all_labels = []

    print("Resizing CIFAR-10 -> 256x256 pixels...")
    for idx in tqdm(range(len(subset)), desc="Processing pixels"):
        img, label = subset[idx]
        pixel = transform(img)  # (3, 256, 256)
        all_pixels.append(pixel)
        all_labels.append(label)

    all_pixels = torch.stack(all_pixels)    # (N, 3, 256, 256)
    all_labels = torch.tensor(all_labels)    # (N,)

    # Save as float16 to reduce disk usage
    all_pixels = all_pixels.half()

    torch.save(all_pixels, os.path.join(pixel_dir, "train_pixels.pt"))
    torch.save(all_labels, os.path.join(pixel_dir, "train_labels.pt"))

    size_mb = all_pixels.element_size() * all_pixels.nelement() / (1024 * 1024)
    print(f"Saved pixels: shape={all_pixels.shape}, dtype={all_pixels.dtype}, "
          f"size={size_mb:.1f} MB")


def encode_latents(subset, latent_dir: str, image_size: int = 256,
                   vae_model: str = "stabilityai/sd-vae-ft-mse",
                   scale_factor: float = 0.18215, batch_size: int = 8):
    """
    Encode CIFAR-10 images into VAE latents and save as .pt files.

    Args:
        subset: CIFAR-10 subset
        latent_dir: Output directory for latent .pt files
        image_size: Input image size (256)
        vae_model: HuggingFace model ID for the VAE
        scale_factor: Latent scaling factor
        batch_size: Encoding batch size (small due to VRAM)
    """
    os.makedirs(latent_dir, exist_ok=True)

    # Preprocessing to match VAE expectations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC,
                          antialias=True),
        transforms.ToTensor(),           # [0, 1]
        transforms.Normalize([0.5], [0.5]),  # [-1, 1]
    ])

    # Load VAE
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SD-VAE from '{vae_model}' on {device}...")
    vae = AutoencoderKL.from_pretrained(vae_model)
    vae = vae.to(device, dtype=torch.float16)
    vae.eval()

    # Process in batches
    all_latents = []
    all_labels = []

    # Prepare all images
    all_images = []
    for idx in range(len(subset)):
        img, label = subset[idx]
        all_images.append(transform(img))
        all_labels.append(label)

    all_labels = torch.tensor(all_labels)

    print("Encoding images -> VAE latents...")
    for i in tqdm(range(0, len(all_images), batch_size), desc="Encoding batches"):
        batch = torch.stack(all_images[i:i + batch_size]).to(device, dtype=torch.float16)

        with torch.no_grad():
            posterior = vae.encode(batch).latent_dist
            latents = posterior.sample() * scale_factor  # (B, 4, 32, 32)

        all_latents.append(latents.cpu().half())

    all_latents = torch.cat(all_latents, dim=0)  # (N, 4, 32, 32)

    torch.save(all_latents, os.path.join(latent_dir, "train_latents.pt"))
    torch.save(all_labels, os.path.join(latent_dir, "train_labels.pt"))

    size_mb = all_latents.element_size() * all_latents.nelement() / (1024 * 1024)
    print(f"Saved latents: shape={all_latents.shape}, dtype={all_latents.dtype}, "
          f"size={size_mb:.1f} MB")

    # Cleanup VAE from VRAM
    del vae
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Encode CIFAR-10 data for diffusion training")
    parser.add_argument("--images_per_class", type=int, default=1000,
                        help="Number of images per class (default: 1000)")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Root data directory")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Target image size (default: 256)")
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse",
                        help="HuggingFace VAE model ID")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="VAE encoding batch size")
    args = parser.parse_args()

    pixel_dir = os.path.join(args.data_dir, "pixels")
    latent_dir = os.path.join(args.data_dir, "latents")

    print("=" * 60)
    print("CIFAR-10 Data Encoding Pipeline")
    print("=" * 60)
    print(f"  Images per class: {args.images_per_class}")
    print(f"  Total images:     {args.images_per_class * 10}")
    print(f"  Image size:       {args.image_size}×{args.image_size}")
    print(f"  VAE model:        {args.vae_model}")
    print()

    start = time.time()

    # Step 1: Get CIFAR-10 subset
    subset = get_cifar10_subset(args.images_per_class, args.data_dir)

    # Step 2: Encode pixels (resize to 256x256)
    print("\n--- Step 1/2: Encoding Pixels ---")
    encode_pixels(subset, pixel_dir, args.image_size)

    # Step 3: Encode latents (VAE)
    print("\n--- Step 2/2: Encoding Latents (VAE) ---")
    encode_latents(subset, latent_dir, args.image_size, args.vae_model,
                   batch_size=args.batch_size)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"Encoding complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Pixels: {pixel_dir}/")
    print(f"  Latents: {latent_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
