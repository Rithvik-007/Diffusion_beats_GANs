# Latent Diffusion Model — "Diffusion Models Beat GANs"

A faithful implementation of the **ADM (Ablated Diffusion Model)** architecture from [Dhariwal & Nichol, 2021](https://arxiv.org/abs/2105.05233), with a dual-training comparison:

- **Baseline**: Full pixel-space UNet training (slow, expensive)
- **Evolution**: Latent-space full UNet training on VAE latents (fast, efficient)
- **Steering**: Classifier-Free Guidance (CFG) via unconditional dropout during training
- **Optional**: Classifier Guidance via a separate auxiliary classifier

**Dataset**: CIFAR-10 (10,000 images)  
**Hardware Target**: RTX 3060 6GB VRAM (or similar)

---

## Architecture

```
CIFAR-10 (32x32) -> Resize 256x256 -> SD-VAE Encoder -> Latents (4x32x32)
                                                            |
                                                    ADM UNet (full)
                                                    (predicts noise)
                                                            |
                                                    CFG-Guided Sampling
                                                            |
                                                    SD-VAE Decoder -> Output Image (256x256)
```

### Key Specs

| Component | Specification |
|-----------|--------------|
| **ResBlocks** | BigGAN-style with AdaGN conditioning |
| **Attention** | Multi-head, 64 channels/head, at resolutions 32x32, 16x16, 8x8 |
| **Conditioning** | Sinusoidal timestep embedding + class embedding -> AdaGN |
| **Classes** | 11 (10 CIFAR-10 + 1 null class for CFG dropout) |
| **Timesteps** | 200 (linear schedule) |
| **CFG** | 10% unconditional dropout during training |

### Pixel vs Latent Comparison

| | Pixel Baseline | Latent Full UNet |
|---|---|---|
| Input | 256x256x3 RGB | 32x32x4 latent |
| Trainable params | ~115M | ~115M |
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
│   └── lora.py               # LoRA injection wrapper (optional)
├── diffusion/
│   ├── __init__.py           # Package exports
│   ├── gaussian_diffusion.py # DDPM forward/reverse + classifier guidance
│   └── schedule.py           # Linear/cosine noise schedules
├── config.py                 # All hyperparameters
├── sanity_check.py           # VAE encode/decode sanity check (run first!)
├── encode_data.py            # CIFAR-10 -> VAE latents + resized pixels
├── train_classifier.py       # Train noisy-image classifier for guidance
├── train_pixel.py            # Baseline: pixel-space full training
├── train_latent.py           # Evolution: latent-space full UNet training
├── sample.py                 # Generate images with optional guidance
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

Run these steps **in order**. Each step depends on the previous one.

### Step 0: VAE Sanity Check (~2 min)

**Do this first!** Verifies the VAE encode/decode pipeline works.

```bash
python sanity_check.py
```

Open `sanity_check.png` — it should show a blurry version of a CIFAR image.  
**If it's black, STOP. The VAE setup is broken.**

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

### Step 2: Train Latent UNet (~30 min)

Trains the full UNet on latent space with CFG dropout.

```bash
python train_latent.py --epochs 50 --batch_size 4
```

Supports resume if interrupted:
```bash
python train_latent.py --epochs 50 --batch_size 4 --resume
```

**Check progress**: Look at `runs/latent/samples/step_500.png` — you should see colored blobs by step 1000.

**Output:** `runs/latent/checkpoint_*.pt`, `speed_log.json`, sample PNGs

### Step 3: Generate Samples (~5 min)

First verify base generation works (no guidance):

```bash
python sample.py --mode latent --guidance_scale 0.0 --num_samples 3
```

Verify the images are NOT black. If they look reasonable, try with CFG:

```bash
python sample.py --mode latent --guidance_scale 2.0 --num_samples 5
```

**Output:** `results/samples_latent_guided.png`

### Step 4: Train Pixel Baseline (optional, ~30 min)

Only needed for the speed comparison. Intentionally slow.

```bash
python train_pixel.py --epochs 3 --batch_size 1
```

### Step 5: Compare Results (instant)

Generate presentation-ready comparison charts:

```bash
python compare.py
```

**Output:**
- `results/speed_comparison.png`
- `results/loss_curves.png`
- `results/comparison_table.png`

---

## Key Command-Line Arguments

### `train_latent.py`
| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch_size` | 4 | Batch size |
| `--lr` | 1e-4 | Learning rate |
| `--timesteps` | 200 | Diffusion timesteps |
| `--p_uncond` | 0.1 | CFG unconditional dropout probability |
| `--save_interval` | 500 | Save checkpoint every N steps |
| `--sample_interval` | 500 | Generate sample images every N steps |
| `--resume` | False | Resume from latest checkpoint |

### `sample.py`
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | latent | `latent` or `pixel` |
| `--guidance_scale` | 0.0 | Guidance strength (0=off, try 1.5-3.0) |
| `--num_samples` | 10 | Samples per class |
| `--timesteps` | 200 | Must match training timesteps |
| `--classes` | 0-9 | Comma-separated class indices |

---

## Classifier-Free Guidance (CFG)

During training, class labels are randomly dropped (replaced with null class 10) with probability `p_uncond=0.1`. This lets the model learn both conditional and unconditional generation.

During sampling, CFG steers generation using the difference between conditional and unconditional predictions:
```
output = unconditional + guidance_scale * (conditional - unconditional)
```

> **Note**: CFG in the sampler is not yet implemented in `gaussian_diffusion.py`. The current `--guidance_scale` flag uses classifier guidance (external classifier). CFG sampling can be added later.

---

## Tips

- **Always run `sanity_check.py` first** — saves you from training 30 min with a broken pipeline
- **If your PC crashes/overheats**: Use `--resume` to continue from the last checkpoint
- **Monitor GPU temp**: Keep it below 85°C. Close other GPU-heavy apps.
- **VRAM usage**: Latent training uses ~3-4 GB with AMP + gradient checkpointing
- **Training is working if**: Loss drops below 0.5 within the first epoch

---

## References

- Dhariwal & Nichol, ["Diffusion Models Beat GANs on Image Synthesis"](https://arxiv.org/abs/2105.05233), 2021
- Ho et al., ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239), 2020
- Ho & Salimans, ["Classifier-Free Diffusion Guidance"](https://arxiv.org/abs/2207.12598), 2022
