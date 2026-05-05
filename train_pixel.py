"""
Pixel-Space Baseline Training Script.

Trains the ADM UNet directly on 256×256 RGB images (3-channel).
ALL parameters are trainable (~70M). Uses standard FP32 Adam.

This is the "SLOW" baseline for comparison with latent-LoRA training.
Supports --resume to continue from the latest checkpoint.

Usage:
    python train_pixel.py [--epochs 3] [--batch_size 1] [--resume]
"""

import os
import sys
import json
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.unet import UNet
from diffusion import GaussianDiffusion
from config import ModelConfig, DiffusionConfig, TrainConfig


def main():
    parser = argparse.ArgumentParser(description="Pixel-space baseline diffusion training")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Small batch size due to 256x256 full-res training")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pixel_dir", type=str, default="data/pixels")
    parser.add_argument("--save_dir", type=str, default="runs/pixel")
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--sample_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=200)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load pixel data ---
    print("Loading pixel data (256×256)...")
    pixels = torch.load(os.path.join(args.pixel_dir, "train_pixels.pt"),
                        map_location="cpu", weights_only=True).float()
    labels = torch.load(os.path.join(args.pixel_dir, "train_labels.pt"),
                        map_location="cpu", weights_only=True).long()
    print(f"  Pixels: {pixels.shape}, Labels: {labels.shape}")

    dataset = TensorDataset(pixels, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=True)

    # --- Build UNet (pixel mode: 3ch in/out, 256x256 input) ---
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=128,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),  # Relative to 256: at 256/8=32, 256/16=16, 256/32=8
        head_channels=64,
        num_classes=10,
        dropout=0.0,
        input_size=256,
        use_gradient_checkpointing=False,  # No optimization for baseline
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # --- Diffusion process ---
    diffusion = GaussianDiffusion(schedule_name=args.schedule, timesteps=args.timesteps)

    # --- Standard FP32 Adam (no optimizations) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # --- Training ---
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "samples"), exist_ok=True)
    speed_log = []
    global_step = 0
    start_epoch = 0

    # --- Resume from checkpoint ---
    if args.resume:
        # Find latest checkpoint
        ckpts = [f for f in os.listdir(args.save_dir)
                 if f.startswith("checkpoint_") and f.endswith(".pt") and "final" not in f]
        if ckpts:
            # Parse step numbers and pick the highest
            steps = []
            for c in ckpts:
                try:
                    s = int(c.replace("checkpoint_", "").replace(".pt", ""))
                    steps.append((s, c))
                except ValueError:
                    pass
            if steps:
                steps.sort()
                latest_step, latest_ckpt = steps[-1]
                ckpt_path = os.path.join(args.save_dir, latest_ckpt)
                print(f"  Resuming from {ckpt_path} (step {latest_step})")
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                if isinstance(state, dict) and "model" in state:
                    model.load_state_dict(state["model"])
                    optimizer.load_state_dict(state["optimizer"])
                    global_step = state["global_step"]
                    start_epoch = state["epoch"]
                    speed_log = state.get("speed_log", [])
                else:
                    # Legacy checkpoint (just model state_dict)
                    model.load_state_dict(state)
                    global_step = latest_step
                print(f"  Resumed at global_step={global_step}")
        else:
            print("  No checkpoint found, starting fresh.")

        # Load existing speed log
        log_path = os.path.join(args.save_dir, "speed_log.json")
        if os.path.exists(log_path) and not speed_log:
            with open(log_path, "r") as f:
                speed_log = json.load(f)
            print(f"  Loaded {len(speed_log)} existing log entries.")

    print(f"\n{'='*60}")
    print(f"PIXEL BASELINE TRAINING")
    print(f"  Mode: Full FP32, all params trainable")
    print(f"  Input: 256×256×3 RGB")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs} (starting from {start_epoch})")
    print(f"  Global step: {global_step}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_pixels, batch_labels in pbar:
            step_start = time.time()

            batch_pixels = batch_pixels.to(device)
            batch_labels = batch_labels.to(device)

            # Random timesteps
            t = torch.randint(0, args.timesteps, (batch_pixels.shape[0],), device=device)

            # Training loss (MSE on noise prediction)
            loss = diffusion.training_loss(model, batch_pixels, t, batch_labels)

            # Standard backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step_time = time.time() - step_start
            epoch_loss += loss.item()
            epoch_steps += 1
            global_step += 1

            # Logging
            if global_step % args.log_interval == 0:
                entry = {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss.item(),
                    "sec_per_iter": step_time,
                }
                speed_log.append(entry)
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    speed=f"{step_time:.2f}s/it"
                )

            # Generate samples
            if global_step % args.sample_interval == 0:
                _generate_samples(model, diffusion, device, args, global_step)

            # Save checkpoint (full state for resume)
            if global_step % args.save_interval == 0:
                ckpt_path = os.path.join(args.save_dir, f"checkpoint_{global_step}.pt")
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "global_step": global_step,
                    "epoch": epoch,
                    "speed_log": speed_log,
                }, ckpt_path)
                # Also flush speed log to disk
                _flush_speed_log(speed_log, args.save_dir)
                print(f"\n  Checkpoint saved: {ckpt_path} (step {global_step})")

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save final checkpoint and logs
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": args.epochs,
        "speed_log": speed_log,
    }, os.path.join(args.save_dir, "checkpoint_final.pt"))

    _flush_speed_log(speed_log, args.save_dir)
    print(f"\nSpeed log saved.")

    # Print summary
    if speed_log:
        avg_speed = sum(e["sec_per_iter"] for e in speed_log) / len(speed_log)
        print(f"Average speed: {avg_speed:.3f} sec/iteration")

    # VRAM usage
    if torch.cuda.is_available():
        peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"Peak VRAM: {peak_mb:.1f} MB")


def _flush_speed_log(speed_log, save_dir):
    """Flush speed log to disk so it survives crashes."""
    log_path = os.path.join(save_dir, "speed_log.json")
    with open(log_path, "w") as f:
        json.dump(speed_log, f, indent=2)


def _generate_samples(model, diffusion, device, args, step):
    """Generate a small batch of sample images for monitoring."""
    model.eval()
    with torch.no_grad():
        # Generate one sample per class
        class_labels = torch.arange(10, device=device)
        shape = (10, 3, 256, 256)

        # Only do 50 steps for fast preview (not full 1000)
        samples = diffusion.p_sample_loop(
            model, shape, class_labels, device, verbose=False
        )

        # Save as grid (simple concatenation)
        samples = (samples.clamp(-1, 1) + 1) / 2  # [-1,1] -> [0,1]

    save_path = os.path.join(args.save_dir, "samples", f"sample_{step}.pt")
    torch.save(samples.cpu(), save_path)
    model.train()


if __name__ == "__main__":
    main()
