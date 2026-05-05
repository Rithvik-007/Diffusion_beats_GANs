"""
Comparison Script — Pixel vs Latent-LoRA Speed & Quality.

Reads speed logs from both training runs and generates presentation-ready
comparison charts and statistics.

Usage:
    python compare.py
"""

import os
import sys
import json
import argparse

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]


def load_speed_log(log_path: str) -> list:
    """Load speed log JSON file."""
    if not os.path.exists(log_path):
        print(f"WARNING: {log_path} not found")
        return []
    with open(log_path, "r") as f:
        return json.load(f)


def plot_speed_comparison(pixel_log, latent_log, output_dir):
    """Bar chart: average seconds/iteration for pixel vs latent."""
    fig, ax = plt.subplots(figsize=(8, 5))

    pixel_speed = np.mean([e["sec_per_iter"] for e in pixel_log]) if pixel_log else 0
    latent_speed = np.mean([e["sec_per_iter"] for e in latent_log]) if latent_log else 0

    bars = ax.bar(
        ["Pixel Baseline\n(256×256, FP32, 70M params)",
         "Latent + LoRA\n(32×32, FP16, 1.5M params)"],
        [pixel_speed, latent_speed],
        color=["#e74c3c", "#2ecc71"],
        edgecolor="white",
        linewidth=2,
        width=0.5,
    )

    # Add value labels on bars
    for bar, val in zip(bars, [pixel_speed, latent_speed]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}s", ha="center", va="bottom", fontsize=14, fontweight="bold")

    if pixel_speed > 0 and latent_speed > 0:
        speedup = pixel_speed / latent_speed
        ax.set_title(f"Training Speed Comparison\n(Latent-LoRA is {speedup:.1f}× faster)",
                     fontsize=16, fontweight="bold")
    else:
        ax.set_title("Training Speed Comparison", fontsize=16, fontweight="bold")

    ax.set_ylabel("Seconds per Iteration", fontsize=13)
    ax.set_ylim(0, max(pixel_speed, latent_speed) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "speed_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")
    return pixel_speed, latent_speed


def plot_loss_curves(pixel_log, latent_log, output_dir):
    """Overlaid loss curves for both training runs."""
    fig, ax = plt.subplots(figsize=(10, 5))

    if pixel_log:
        steps = [e["step"] for e in pixel_log]
        losses = [e["loss"] for e in pixel_log]
        ax.plot(steps, losses, color="#e74c3c", label="Pixel Baseline", alpha=0.8, linewidth=2)

    if latent_log:
        steps = [e["step"] for e in latent_log]
        losses = [e["loss"] for e in latent_log]
        ax.plot(steps, losses, color="#2ecc71", label="Latent + LoRA", alpha=0.8, linewidth=2)

    ax.set_xlabel("Training Step", fontsize=13)
    ax.set_ylabel("Loss (MSE)", fontsize=13)
    ax.set_title("Training Loss Curves", fontsize=16, fontweight="bold")
    ax.legend(fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "loss_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def plot_stats_table(pixel_log, latent_log, output_dir):
    """Generate a comparison statistics table as an image."""
    pixel_speed = np.mean([e["sec_per_iter"] for e in pixel_log]) if pixel_log else float("nan")
    latent_speed = np.mean([e["sec_per_iter"] for e in latent_log]) if latent_log else float("nan")
    pixel_final_loss = pixel_log[-1]["loss"] if pixel_log else float("nan")
    latent_final_loss = latent_log[-1]["loss"] if latent_log else float("nan")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    table_data = [
        ["Metric", "Pixel Baseline", "Latent + LoRA"],
        ["Input Resolution", "256×256×3", "32×32×4"],
        ["Trainable Params", "~70M (all)", "~1.5M (LoRA)"],
        ["Precision", "FP32", "FP16 (AMP)"],
        ["Optimizer", "Adam", "8-bit Adam"],
        ["Avg sec/iter", f"{pixel_speed:.3f}s", f"{latent_speed:.3f}s"],
        ["Final Loss", f"{pixel_final_loss:.4f}", f"{latent_final_loss:.4f}"],
        [
            "Speedup",
            "1.0×",
            f"{pixel_speed/latent_speed:.1f}×" if latent_speed > 0 and pixel_speed > 0 else "N/A",
        ],
    ]

    table = ax.table(
        cellText=table_data,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.8)

    # Style header row
    for j in range(3):
        table[0, j].set_facecolor("#34495e")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = "#f7f9fc" if i % 2 == 0 else "#ffffff"
        for j in range(3):
            table[i, j].set_facecolor(color)

    ax.set_title("Training Comparison Summary", fontsize=16, fontweight="bold", pad=20)

    plt.tight_layout()
    save_path = os.path.join(output_dir, "comparison_table.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots")
    parser.add_argument("--pixel_dir", type=str, default="runs/pixel")
    parser.add_argument("--latent_dir", type=str, default="runs/latent")
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("PIXEL vs LATENT-LoRA COMPARISON")
    print("=" * 60)

    # Load logs
    pixel_log = load_speed_log(os.path.join(args.pixel_dir, "speed_log.json"))
    latent_log = load_speed_log(os.path.join(args.latent_dir, "speed_log.json"))

    if not pixel_log and not latent_log:
        print("ERROR: No speed logs found. Run train_pixel.py and/or train_latent.py first.")
        return

    print(f"  Pixel log: {len(pixel_log)} entries")
    print(f"  Latent log: {len(latent_log)} entries")

    # Generate plots
    print("\nGenerating charts...")
    pixel_speed, latent_speed = plot_speed_comparison(pixel_log, latent_log, args.output_dir)
    plot_loss_curves(pixel_log, latent_log, args.output_dir)
    plot_stats_table(pixel_log, latent_log, args.output_dir)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    if pixel_speed > 0:
        print(f"  Pixel baseline:  {pixel_speed:.3f} sec/iter")
    if latent_speed > 0:
        print(f"  Latent + LoRA:   {latent_speed:.3f} sec/iter")
    if pixel_speed > 0 and latent_speed > 0:
        print(f"  Speedup:         {pixel_speed/latent_speed:.1f}×")
    print(f"\nAll plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
