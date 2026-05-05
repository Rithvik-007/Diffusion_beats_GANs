# Latent Diffusion Model — "Diffusion Models Beat GANs"

A faithful implementation of the **ADM (Ablated Diffusion Model)** architecture from [Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233), with a dual-training comparison:

- **Baseline**: Full pixel-space UNet training (slow, expensive)
- **Evolution**: Latent-space UNet + LoRA fine-tuning (fast, efficient)
- **Steering**: Classifier Guidance via a separate auxiliary classifier trained on noisy images

**Dataset**: CIFAR-10 (10,000 images)  
**Hardware Target**: RTX 3060 6GB VRAM (or similar)

---

## Architecture

```
CIFAR-10 (32×32) → Resize 256×256 → SD-VAE Encoder → Latents (4×32×32)
                                                          ↓
                                                  ADM UNet + LoRA
                                                  (predicts noise ε)
                                                          ↓
                                              Classifier-Guided Sampling
                                                          ↓
                                                  SD-VAE Decoder → Output Image (256×256)
```

### Key Specs

| Component | Specification |
|-----------|--------------|
| **ResBlocks** | BigGAN-style with AdaGN conditioning |
| **Attention** | Multi-head, 64 channels/head, at resolutions 32×32, 16×16, 8×8 |
| **Conditioning** | Sinusoidal timestep embedding + class embedding → AdaGN |
| **Guidance** | External gradient-based classifier guidance |
| **LoRA** | Rank 16, injected into attention Q/K/V/O projections |

### Pixel vs Latent Comparison

| | Pixel Baseline | Latent + LoRA |
|---|---|---|
| Input | 256×256×3 RGB | 32×32×4 latent |
| Trainable params | ~115M (all) | ~557K (LoRA only) |
| Precision | FP32 | FP16 AMP |
| Optimizer | Adam | 8-bit Adam |
| Speed | ~4 sec/iter | ~0.3 sec/iter |

---

## Project Structure

```
├── model/
│   ├── __init__.py           # Package exports
│   ├── unet.py               # Full ADM-style UNet
│   ├── blocks.py             # BigGAN ResBlock, AdaGN, Up/Downsample
│   ├── attention.py          # Multi-head self-attention (64ch heads)
│   ├── classifier.py         # Auxiliary noisy-image classifier
│   └── lora.py               # LoRA injection wrapper
├── diffusion/
│   ├── __init__.py           # Package exports
│   ├── gaussian_diffusion.py # DDPM forward/reverse + classifier guidance
│   └── schedule.py           # Linear/cosine noise schedules
├── config.py                 # All hyperparameters
├── encode_data.py            # CIFAR-10 → VAE latents + resized pixels
├── train_classifier.py       # Train noisy-image classifier for guidance
├── train_pixel.py            # Baseline: pixel-space full training
├── train_latent.py           # Evolution: latent-space LoRA training
├── sample.py                 # Generate images with classifier guidance
├── compare.py                # Speed/quality comparison plots
└── requirements.txt          # Dependencies
```

---

## Setup

### 1. Create Virtual Environment

```bash
python -m venv myenv

# Windows
.\myenv\Scripts\activate

# Linux/Mac
source myenv/bin/activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note**: `bitsandbytes` may need a Windows-specific build. If it fails, the training will automatically fall back to standard AdamW.

### 3. Verify GPU

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Execution Workflow

Run these scripts **in order**. Each step depends on the previous one.

### Step 1: Encode Data (~10 min)

Pre-encodes CIFAR-10 images into VAE latents and resized pixel tensors.

```bash
python encode_data.py
```

**Output:**
```
data/latents/train_latents.pt    # (10000, 4, 32, 32)
data/latents/train_labels.pt     # (10000,)
data/pixels/train_pixels.pt      # (10000, 3, 256, 256)
data/pixels/train_labels.pt      # (10000,)
```

### Step 2: Train Classifier (~15 min)

Trains an auxiliary classifier on noisy latents for classifier guidance.

```bash
python train_classifier.py
```

**Output:** `checkpoints/classifier.pt`

### Step 3: Train Pixel Baseline (~30 min)

The intentionally slow baseline with all ~115M params trainable in FP32.

```bash
python train_pixel.py --epochs 3 --batch_size 1
```

Supports resume if interrupted:
```bash
python train_pixel.py --epochs 3 --batch_size 1 --resume
```

**Output:** `runs/pixel/speed_log.json` + checkpoints

### Step 4: Train Latent + LoRA (~30 min)

The fast, optimized version with only ~557K LoRA params trainable.

```bash
python train_latent.py --epochs 50 --batch_size 4
```

Supports resume:
```bash
python train_latent.py --epochs 50 --batch_size 4 --resume
```

> **Important**: The first run saves `runs/latent/base_model.pt` — the frozen base weights that LoRA adapts. This file is required for sampling.

**Output:** `runs/latent/base_model.pt`, `lora_ckpt_*.pt`, `speed_log.json`

### Step 5: Generate Samples (~5 min)

Generate images with classifier guidance:

```bash
python sample.py --mode latent --guidance_scale 2.0 --num_samples 5
```

**Output:** `results/samples_latent_guided.png`

### Step 6: Compare Results (instant)

Generate presentation-ready comparison charts:

```bash
python compare.py
```

**Output:**
- `results/speed_comparison.png`
- `results/loss_curves.png`  
- `results/comparison_table.png`

---

## Classifier Guidance

During sampling, the noise prediction is steered using gradients from the trained classifier:

```
ε̂(x_t, t, y) = ε_θ(x_t, t) - s · σ_t · ∇_{x_t} log p_φ(y | x_t)
```

Where `s` is the guidance scale (default: 2.0). Higher values = stronger class conditioning.

---

## Key Command-Line Arguments

### `train_latent.py`
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--lora_rank` | 16 | LoRA rank |
| `--save_interval` | 500 | Save checkpoint every N steps |
| `--resume` | False | Resume from latest checkpoint |

### `sample.py`
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | latent | `latent` or `pixel` |
| `--guidance_scale` | 2.0 | Classifier guidance strength |
| `--num_samples` | 10 | Samples per class |
| `--classes` | 0-9 | Comma-separated class indices |

---

## Tips

- **If your PC crashes/overheats**: Use `--resume` to continue training from the last checkpoint
- **Monitor GPU temp**: Keep it below 85°C. Close other GPU-heavy apps.
- **VRAM usage**: Latent training uses ~2.5 GB, pixel training uses ~5+ GB
- **Training is working if**: Loss drops below 1.0 within the first few hundred steps (latent mode)

---

## References

- Dhariwal & Nichol, ["Diffusion Models Beat GANs on Image Synthesis"](https://arxiv.org/abs/2105.05233), 2021
- Ho et al., ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239), 2020
- Hu et al., ["LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685), 2021
