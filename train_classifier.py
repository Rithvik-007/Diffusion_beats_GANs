"""
Auxiliary Classifier Training Script.

Trains a small ResNet classifier to predict class labels from noisy
latent inputs at arbitrary timesteps. This classifier is used for
classifier guidance during sampling.

Run AFTER encode_data.py and BEFORE sampling.

Usage:
    python train_classifier.py [--epochs 20] [--batch_size 32]
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.classifier import NoisyClassifier
from diffusion import GaussianDiffusion
from config import ClassifierConfig, DiffusionConfig


def main():
    parser = argparse.ArgumentParser(description="Train auxiliary classifier on noisy latents")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--latent_dir", type=str, default="data/latents")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--timesteps", type=int, default=1000)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load pre-encoded latents ---
    print("Loading pre-encoded latents...")
    latents = torch.load(os.path.join(args.latent_dir, "train_latents.pt"),
                         map_location="cpu", weights_only=True).float()
    labels = torch.load(os.path.join(args.latent_dir, "train_labels.pt"),
                        map_location="cpu", weights_only=True).long()
    print(f"  Latents: {latents.shape}, Labels: {labels.shape}")

    dataset = TensorDataset(latents, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)

    # --- Build classifier ---
    classifier = NoisyClassifier(
        in_channels=4,  # Latent channels
        hidden_channels=64,
        num_classes=10,
    ).to(device)

    total_params = sum(p.numel() for p in classifier.parameters())
    print(f"  Classifier params: {total_params:,}")

    # --- Diffusion process (for creating noisy inputs) ---
    diffusion = GaussianDiffusion(schedule_name=args.schedule, timesteps=args.timesteps)

    # --- Optimizer ---
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    # --- Training ---
    os.makedirs(args.save_dir, exist_ok=True)
    log_path = os.path.join(args.save_dir, "classifier_log.json")
    logs = []

    print(f"\nTraining classifier for {args.epochs} epochs...")
    print(f"{'='*60}")

    for epoch in range(args.epochs):
        classifier.train()
        epoch_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_latents, batch_labels in pbar:
            batch_latents = batch_latents.to(device)
            batch_labels = batch_labels.to(device)

            # Sample random timesteps
            t = torch.randint(0, args.timesteps, (batch_latents.shape[0],), device=device)

            # Create noisy latents
            x_t = diffusion.q_sample(batch_latents, t)

            # Forward
            logits = classifier(x_t, t)
            loss = F.cross_entropy(logits, batch_labels)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Stats
            epoch_loss += loss.item()
            preds = logits.argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()
            total += batch_labels.shape[0]

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100*correct/total:.1f}%")

        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total
        logs.append({"epoch": epoch + 1, "loss": avg_loss, "accuracy": accuracy})

        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")

    # --- Save ---
    save_path = os.path.join(args.save_dir, "classifier.pt")
    torch.save(classifier.state_dict(), save_path)
    print(f"\nClassifier saved to {save_path}")

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"Training log saved to {log_path}")


if __name__ == "__main__":
    main()
