"""
VAE Sanity Check — Run this BEFORE training.

Loads a single CIFAR-10 image, encodes it through the SD-VAE,
immediately decodes it, and saves the result as sanity_check.png.

Expected result: a blurry version of the original image.
If the output is BLACK, the VAE pipeline is broken — stop and fix it.

Usage:
    python sanity_check.py
"""

import torch
import torchvision
from diffusers import AutoencoderKL
from torchvision import transforms
from torchvision.utils import save_image

print("Loading VAE...")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").cuda().half()

print("Loading CIFAR-10 sample...")
dataset = torchvision.datasets.CIFAR10(
    root="./data", download=True,
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
)

# Grab first image
img = dataset[0][0].unsqueeze(0).cuda().half() * 2 - 1  # [0,1] -> [-1,1]

print("Encoding -> Decoding...")
with torch.no_grad():
    latent = vae.encode(img).latent_dist.sample() * 0.18215
    decoded = vae.decode(latent / 0.18215).sample
    out = (decoded / 2 + 0.5).clamp(0, 1).float().cpu()

save_image(out, "sanity_check.png")
print(f"\nSaved sanity_check.png")
print(f"  Output range: [{out.min():.3f}, {out.max():.3f}]")
print(f"  Output mean: {out.mean():.3f}")

if out.mean() < 0.01:
    print("\n  *** WARNING: Image appears BLACK. VAE pipeline may be broken! ***")
else:
    print("\n  Looks good! Open sanity_check.png to verify it's a blurry image.")

del vae
torch.cuda.empty_cache()
