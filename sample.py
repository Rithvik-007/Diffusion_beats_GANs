"""
Sampling Script with Classifier Guidance.

Generates images using the trained UNet + classifier guidance.
Supports both pixel-space and latent-space modes.

Usage:
    python sample.py --mode latent --guidance_scale 2.0 --num_samples 10
    python sample.py --mode pixel --guidance_scale 0.0 --num_samples 10
"""

import os
import sys
import argparse

import torch
import torchvision.utils as vutils
from diffusers import AutoencoderKL

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.unet import UNet
from model.lora import inject_lora
from model.classifier import NoisyClassifier
from diffusion import GaussianDiffusion


def load_latent_model(args, device):
    """Load UNet with LoRA weights for latent-space generation."""
    model = UNet(
        in_channels=4, out_channels=4,
        base_channels=128, channel_mult=(1, 2, 3, 4),
        num_res_blocks=2, attention_resolutions=(32, 16, 8),
        head_channels=64, num_classes=10, input_size=32,
    ).to(device)

    # Load the EXACT base model weights used during training
    # (critical: LoRA weights are trained on specific base weights)
    base_path = os.path.join(args.model_dir, "base_model.pt")
    if os.path.exists(base_path):
        model.load_state_dict(torch.load(base_path, map_location=device, weights_only=True))
        print(f"Loaded base model from {base_path}")
    else:
        print(f"WARNING: No base model found at {base_path}!")
        print("  The model will have random base weights — LoRA weights will not work correctly.")
        print("  Please retrain with the updated train_latent.py to save base_model.pt")

    # Inject LoRA structure
    model, _ = inject_lora(model, rank=args.lora_rank, alpha=args.lora_alpha)

    # Load LoRA weights — try final first, then fall back to latest checkpoint
    lora_path = os.path.join(args.model_dir, "lora_weights_final.pt")
    lora_state = None

    if os.path.exists(lora_path):
        lora_state = torch.load(lora_path, map_location=device, weights_only=True)
        print(f"Loaded LoRA weights from {lora_path}")
    else:
        # Fall back to latest resume checkpoint
        ckpts = [f for f in os.listdir(args.model_dir)
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
                ckpt_path = os.path.join(args.model_dir, latest_ckpt)
                state = torch.load(ckpt_path, map_location=device, weights_only=True)
                lora_state = state["lora_state"]
                print(f"Loaded LoRA weights from checkpoint {ckpt_path} (step {latest_step})")

    if lora_state is not None:
        model_state = model.state_dict()
        loaded = 0
        for key, val in lora_state.items():
            if key in model_state:
                model_state[key] = val
                loaded += 1
        model.load_state_dict(model_state)
        print(f"  Loaded {loaded} LoRA parameter tensors")
    else:
        print(f"WARNING: No LoRA weights found, using random init")

    model.eval()
    return model


def load_pixel_model(args, device):
    """Load pixel-space UNet checkpoint."""
    model = UNet(
        in_channels=3, out_channels=3,
        base_channels=128, channel_mult=(1, 2, 3, 4),
        num_res_blocks=2, attention_resolutions=(32, 16, 8),
        head_channels=64, num_classes=10, input_size=256,
    ).to(device)

    ckpt_path = os.path.join(args.model_dir, "checkpoint_final.pt")
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        print(f"Loaded pixel checkpoint from {ckpt_path}")
    else:
        print(f"WARNING: No checkpoint found at {ckpt_path}, using random init")

    model.eval()
    return model


def load_classifier(args, device):
    """Load trained auxiliary classifier."""
    classifier = NoisyClassifier(
        in_channels=4 if args.mode == "latent" else 3,
        hidden_channels=64, num_classes=10,
    ).to(device)

    clf_path = os.path.join(args.classifier_dir, "classifier.pt")
    if os.path.exists(clf_path):
        classifier.load_state_dict(torch.load(clf_path, map_location=device, weights_only=True))
        print(f"Loaded classifier from {clf_path}")
    else:
        print(f"WARNING: No classifier found at {clf_path}")
        return None

    classifier.eval()
    return classifier


def decode_latents(latents, vae_model_id, device):
    """Decode VAE latents back to pixel images."""
    print("Loading VAE decoder...")
    vae = AutoencoderKL.from_pretrained(vae_model_id)
    vae = vae.to(device, dtype=torch.float16)
    vae.eval()

    scale_factor = 0.18215

    with torch.no_grad():
        # Undo the scaling
        latents = latents.to(device, dtype=torch.float16) / scale_factor
        images = vae.decode(latents).sample

    # Post-process: [-1, 1] → [0, 1]
    images = (images / 2 + 0.5).clamp(0, 1)

    del vae
    torch.cuda.empty_cache()

    return images.float().cpu()


def main():
    parser = argparse.ArgumentParser(description="Generate images with classifier guidance")
    parser.add_argument("--mode", type=str, choices=["latent", "pixel"], default="latent")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples per class")
    parser.add_argument("--classes", type=str, default="0,1,2,3,4,5,6,7,8,9",
                        help="Comma-separated class indices")
    parser.add_argument("--guidance_scale", type=float, default=2.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule", type=str, default="linear")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Directory with model weights (default: runs/<mode>)")
    parser.add_argument("--classifier_dir", type=str, default="checkpoints")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    args = parser.parse_args()

    if args.model_dir is None:
        args.model_dir = f"runs/{args.mode}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")
    print(f"Guidance scale: {args.guidance_scale}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Parse class list
    target_classes = [int(c) for c in args.classes.split(",")]
    print(f"Target classes: {target_classes}")

    # CIFAR-10 class names
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

    # Load model
    if args.mode == "latent":
        model = load_latent_model(args, device)
        shape_per_sample = (4, 32, 32)
    else:
        model = load_pixel_model(args, device)
        shape_per_sample = (3, 256, 256)

    # Load classifier
    classifier = None
    if args.guidance_scale > 0:
        classifier = load_classifier(args, device)

    # Diffusion process
    diffusion = GaussianDiffusion(schedule_name=args.schedule, timesteps=args.timesteps)

    # Generate samples for each class
    all_samples = []
    for cls_idx in target_classes:
        print(f"\nGenerating class {cls_idx} ({class_names[cls_idx]})...")
        class_labels = torch.full((args.num_samples,), cls_idx, device=device, dtype=torch.long)
        shape = (args.num_samples,) + shape_per_sample

        samples = diffusion.p_sample_loop(
            model, shape, class_labels, device,
            classifier=classifier,
            guidance_scale=args.guidance_scale,
            verbose=True,
        )
        all_samples.append(samples)

    all_samples = torch.cat(all_samples, dim=0)  # (num_classes * num_samples, C, H, W)

    # Decode latents if in latent mode
    if args.mode == "latent":
        print("\nDecoding latents → images...")
        all_images = decode_latents(all_samples, args.vae_model, device)
    else:
        all_images = (all_samples.clamp(-1, 1) + 1) / 2  # [-1,1] → [0,1]

    # Save as image grid
    grid = vutils.make_grid(all_images, nrow=args.num_samples, padding=2, normalize=False)
    save_path = os.path.join(args.output_dir, f"samples_{args.mode}_guided.png")

    from torchvision.utils import save_image
    save_image(grid, save_path)
    print(f"\nSaved sample grid to {save_path}")
    print(f"Grid shape: {grid.shape}")


if __name__ == "__main__":
    main()
