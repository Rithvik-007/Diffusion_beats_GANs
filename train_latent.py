"""
Latent-Space LoRA Training Script.

Trains the ADM UNet on 32×32 VAE latents (4-channel) with LoRA adapters.
Only LoRA weights are trainable (~1.5M params). Uses FP16 AMP + 8-bit Adam.

This is the "FAST" evolution for comparison with pixel baseline training.
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.unet import UNet
from model.lora import inject_lora
from diffusion import GaussianDiffusion
from config import ModelConfig, LoRAConfig, DiffusionConfig, TrainConfig


def get_optimizer(lora_params, lr, use_8bit):
    """
    Get optimizer with 8-bit Adam fallback.
    """
    if use_8bit:
        try:
            import bitsandbytes as bnb
            optimizer = bnb.optim.Adam8bit(lora_params, lr=lr)
            print("  Using 8-bit Adam (bitsandbytes)")
            return optimizer
        except ImportError:
            print("  WARNING: bitsandbytes not available, falling back to AdamW")
        except Exception as e:
            print(f"  WARNING: bitsandbytes error ({e}), falling back to AdamW")

    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    print("  Using standard AdamW")
    return optimizer


def main():
    parser = argparse.ArgumentParser(description="Latent-space LoRA diffusion training")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--latent_dir", type=str, default="data/latents")
    parser.add_argument("--save_dir", type=str, default="runs/latent")
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--use_8bit_adam", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
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
    model = UNet(
        in_channels=4,
        out_channels=4,
        base_channels=128,
        channel_mult=(1, 2, 3, 4),
        num_res_blocks=2,
        attention_resolutions=(32, 16, 8),
        head_channels=64,
        num_classes=10,
        dropout=0.0,
        input_size=32,
        use_gradient_checkpointing=args.gradient_checkpointing,
    ).to(device)

    # --- Save base model weights (BEFORE LoRA injection) ---
    # This is critical: sample.py needs the exact same base weights
    # that LoRA was trained on top of. Without this, sampling creates
    # a new random UNet and the LoRA weights are useless.
    base_model_path = os.path.join(args.save_dir, "base_model.pt")
    if not os.path.exists(base_model_path):
        torch.save(model.state_dict(), base_model_path)
        size_mb = os.path.getsize(base_model_path) / (1024 * 1024)
        print(f"  Saved base model weights: {base_model_path} ({size_mb:.1f} MB)")
    else:
        # Resume: load the original base model weights
        model.load_state_dict(torch.load(base_model_path, map_location=device, weights_only=True))
        print(f"  Loaded base model weights from {base_model_path}")

    # --- Inject LoRA ---
    print(f"\nInjecting LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    model, lora_params = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    # --- Diffusion process ---
    diffusion = GaussianDiffusion(schedule_name=args.schedule, timesteps=args.timesteps)

    # --- 8-bit Adam optimizer (LoRA params only) ---
    optimizer = get_optimizer(lora_params, args.lr, args.use_8bit_adam)

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
                 if f.startswith("lora_ckpt_") and f.endswith(".pt")]
        if ckpts:
            steps = []
            for c in ckpts:
                try:
                    s = int(c.replace("lora_ckpt_", "").replace(".pt", ""))
                    steps.append((s, c))
                except ValueError:
                    pass
            if steps:
                steps.sort()
                latest_step, latest_ckpt = steps[-1]
                ckpt_path = os.path.join(args.save_dir, latest_ckpt)
                print(f"  Resuming from {ckpt_path} (step {latest_step})")
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                # Load LoRA weights into model
                model_state = model.state_dict()
                for key, val in state["lora_state"].items():
                    if key in model_state:
                        model_state[key] = val
                model.load_state_dict(model_state)
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
    print(f"LATENT-LoRA TRAINING")
    print(f"  Mode: FP16 AMP + {'8-bit Adam' if args.use_8bit_adam else 'AdamW'} + LoRA")
    print(f"  Input: 32×32×4 VAE latents")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Epochs: {args.epochs} (starting from {start_epoch})")
    print(f"  Global step: {global_step}")
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

            # Generate samples
            if global_step % args.sample_interval == 0:
                _generate_samples(model, diffusion, device, args, global_step)

            # Save LoRA weights + optimizer state for resume
            if global_step % args.save_interval == 0:
                _save_resumable_checkpoint(model, optimizer, scaler, global_step, epoch, speed_log, args.save_dir)
                _flush_speed_log(speed_log, args.save_dir)

        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"  Epoch {epoch+1} avg loss: {avg_loss:.4f}")

    # Save final LoRA weights and logs
    _save_lora_weights(model, args.save_dir, "final")
    _save_resumable_checkpoint(model, optimizer, scaler, global_step, args.epochs, speed_log, args.save_dir)

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


def _save_lora_weights(model, save_dir, tag):
    """Save only the LoRA parameters (very small files)."""
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu()

    save_path = os.path.join(save_dir, f"lora_weights_{tag}.pt")
    torch.save(lora_state, save_path)
    size_kb = os.path.getsize(save_path) / 1024
    print(f"\n  LoRA weights saved: {save_path} ({size_kb:.1f} KB)")


def _save_resumable_checkpoint(model, optimizer, scaler, global_step, epoch, speed_log, save_dir):
    """Save full checkpoint for resume capability."""
    lora_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            lora_state[name] = param.data.cpu()

    ckpt = {
        "lora_state": lora_state,
        "optimizer": optimizer.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "speed_log": speed_log,
    }
    if scaler is not None:
        ckpt["scaler"] = scaler.state_dict()

    save_path = os.path.join(save_dir, f"lora_ckpt_{global_step}.pt")
    torch.save(ckpt, save_path)
    print(f"  Resume checkpoint saved: {save_path}")


def _flush_speed_log(speed_log, save_dir):
    """Flush speed log to disk so it survives crashes."""
    log_path = os.path.join(save_dir, "speed_log.json")
    with open(log_path, "w") as f:
        json.dump(speed_log, f, indent=2)


def _generate_samples(model, diffusion, device, args, step):
    """Generate sample latents for monitoring."""
    model.eval()
    with torch.no_grad():
        class_labels = torch.arange(10, device=device)
        shape = (10, 4, 32, 32)

        samples = diffusion.p_sample_loop(
            model, shape, class_labels, device, verbose=False
        )

    save_path = os.path.join(args.save_dir, "samples", f"latent_sample_{step}.pt")
    torch.save(samples.cpu(), save_path)
    model.train()


if __name__ == "__main__":
    main()
