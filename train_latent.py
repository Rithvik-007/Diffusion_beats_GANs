"""
Latent-Space Full UNet Training Script.

Trains the ADM UNet on 32x32 VAE latents (4-channel) with ALL parameters.
Uses FP16 AMP + 8-bit Adam + gradient checkpointing for VRAM efficiency.
Includes Classifier-Free Guidance (CFG) dropout during training.

Supports --resume to continue from the latest checkpoint.

Usage:
    python train_latent.py [--epochs 50] [--batch_size 4] [--resume]
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
import torchvision.utils as vutils

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.unet import UNet
from diffusion import GaussianDiffusion


def get_optimizer(params, lr, use_8bit):
    """
    Get optimizer with 8-bit Adam fallback.
    """
    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(params, lr=lr)
            print("  Using 8-bit Adam (bitsandbytes)")
            return optimizer
        except ImportError:
            print("  WARNING: bitsandbytes not available, falling back to AdamW")
        except Exception as e:
            print(f"  WARNING: bitsandbytes error ({e}), falling back to AdamW")

    optimizer = torch.optim.AdamW(params, lr=lr)
    print("  Using standard AdamW")
    return optimizer


def main():
    parser = argparse.ArgumentParser(description="Latent-space full UNet diffusion training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--latent_dir", type=str, default="data/latents")
    parser.add_argument("--save_dir", type=str, default="runs/latent")
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--use_8bit_adam", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--p_uncond", type=float, default=0.1,
                        help="Probability of dropping class label for CFG training")
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--sample_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
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

    # --- Build UNet (latent mode: 4ch in/out, 32x32 input) ---
    # num_classes=11: 10 CIFAR-10 classes + 1 null class (index 10) for CFG
    model = UNet(
        in_channels=4,
        out_channels=4,
        base_channels=128,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        head_channels=64,
        num_classes=11,
        dropout=0.0,
        input_size=32,
        use_gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    # --- Diffusion process ---
    diffusion = GaussianDiffusion(schedule_name=args.schedule, timesteps=args.timesteps)

    # --- 8-bit Adam optimizer (ALL parameters) ---
    optimizer = get_optimizer(model.parameters(), args.lr, args.use_8bit_adam)

    # --- AMP setup ---
    scaler = torch.amp.GradScaler("cuda") if args.use_amp and device.type == "cuda" else None
    autocast_ctx = torch.amp.autocast("cuda") if args.use_amp and device.type == "cuda" else torch.nullcontext()

    # --- Training ---
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "samples"), exist_ok=True)
    speed_log = []
    global_step = 0
    start_epoch = 0

    # --- Resume from checkpoint ---
    if args.resume:
        ckpts = [f for f in os.listdir(args.save_dir)
                 if f.startswith("checkpoint_") and f.endswith(".pt") and "final" not in f]
        if ckpts:
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
                model.load_state_dict(state["model"])
                optimizer.load_state_dict(state["optimizer"])
                if scaler is not None and "scaler" in state:
                    scaler.load_state_dict(state["scaler"])
                global_step = state["global_step"]
                start_epoch = state["epoch"]
                speed_log = state.get("speed_log", [])
                print(f"  Resumed at global_step={global_step}, epoch={start_epoch}")
        else:
            print("  No checkpoint found, starting fresh.")

        log_path = os.path.join(args.save_dir, "speed_log.json")
        if os.path.exists(log_path) and not speed_log:
            with open(log_path, "r") as f:
                speed_log = json.load(f)
            print(f"  Loaded {len(speed_log)} existing log entries.")

    print(f"\n{'='*60}")
    print(f"LATENT FULL-UNET TRAINING")
    print(f"  Mode: FP16 AMP + {'8-bit Adam' if args.use_8bit_adam else 'AdamW'}")
    print(f"  Input: 32x32x4 VAE latents")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs} (starting from {start_epoch})")
    print(f"  Global step: {global_step}")
    print(f"  Timesteps: {args.timesteps}")
    print(f"  CFG p_uncond: {args.p_uncond}")
    print(f"  Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0
        epoch_steps = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_latents, batch_labels in pbar:
            step_start = time.time()

            batch_latents = batch_latents.to(device)
            batch_labels = batch_labels.to(device)

            # --- CFG dropout: randomly replace class labels with null class (10) ---
            mask = torch.rand(batch_labels.shape[0], device=device) < args.p_uncond
            batch_labels[mask] = 10  # null class index

            # Random timesteps
            t = torch.randint(0, args.timesteps, (batch_latents.shape[0],), device=device)

            # Forward with AMP
            optimizer.zero_grad()

            with autocast_ctx:
                loss = diffusion.training_loss(model, batch_latents, t, batch_labels)

            # Backward with gradient scaling
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
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
                    speed=f"{step_time:.3f}s/it"
                )

            # Generate sample images
            if global_step % args.sample_interval == 0:
                _generate_samples(model, diffusion, device, args, global_step)

            # Save checkpoint (full model state for resume)
            if global_step % args.save_interval == 0:
                _save_checkpoint(model, optimizer, scaler, global_step, epoch, speed_log, args.save_dir)
                _flush_speed_log(speed_log, args.save_dir)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save final checkpoint and logs
    _save_checkpoint(model, optimizer, scaler, global_step, args.epochs, speed_log, args.save_dir, tag="final")

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


def _save_checkpoint(model, optimizer, scaler, global_step, epoch, speed_log, save_dir, tag=None):
    """Save full model checkpoint for resume capability."""
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "speed_log": speed_log,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    if tag:
        save_path = os.path.join(save_dir, f"checkpoint_{tag}.pt")
    else:
        save_path = os.path.join(save_dir, f"checkpoint_{global_step}.pt")
    torch.save(ckpt, save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n  Checkpoint saved: {save_path} ({size_mb:.1f} MB)")


def _flush_speed_log(speed_log, save_dir):
    """Flush speed log to disk so it survives crashes."""
    log_path = os.path.join(save_dir, "speed_log.json")
    with open(log_path, "w") as f:
        json.dump(speed_log, f, indent=2)


def _generate_samples(model, diffusion, device, args, step):
    """Generate sample latents, decode via VAE, and save as PNG grid."""
    model.eval()

    with torch.no_grad():
        # Generate one sample per class (classes 0-9, no null class)
        class_labels = torch.arange(10, device=device)
        shape = (10, 4, 32, 32)

        samples = diffusion.p_sample_loop(
            model, shape, class_labels, device, verbose=False
        )

    # Decode latents to images via SD-VAE
    try:
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae = vae.to(device, dtype=torch.float16)
        vae.eval()

        with torch.no_grad():
            decoded = vae.decode(
                samples.to(device, dtype=torch.float16) / 0.18215
            ).sample

        images = (decoded / 2 + 0.5).clamp(0, 1).float().cpu()

        del vae
        torch.cuda.empty_cache()

        # Save as PNG grid
        save_path = os.path.join(args.save_dir, "samples", f"step_{step}.png")
        vutils.save_image(images, save_path, nrow=5, padding=2)
        print(f"\n  Samples saved: {save_path}")

    except Exception as e:
        # If VAE fails (e.g., not downloaded yet), save raw latents as fallback
        print(f"\n  WARNING: VAE decode failed ({e}), saving raw latents")
        save_path = os.path.join(args.save_dir, "samples", f"latent_sample_{step}.pt")
        torch.save(samples.cpu(), save_path)

    model.train()


if __name__ == "__main__":
    main()
