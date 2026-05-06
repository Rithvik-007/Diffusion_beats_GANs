# Changelog: Initial Plan vs Final Implementation

This document tracks every change made from the original `implementation_plan.md` to the final working codebase, including the reasoning behind each decision.

---

## 1. LoRA Removed — Full UNet Training Instead

### Initial Plan
- Freeze all ~115M UNet parameters
- Inject LoRA adapters into attention Q/K/V/O projections (rank=16)
- Train only ~557K LoRA parameters (~0.48% of model)
- Save lightweight LoRA weight files (~2MB)

### What Changed
- **Removed LoRA entirely** — all ~115M parameters are now trainable
- Optimizer receives `model.parameters()` instead of just `lora_params`
- Checkpoints save the full model `state_dict` (~693MB each)

### Why
Three critical bugs were caused by LoRA + zero-initialization interaction:

1. **`output_conv` zero-init (unet.py)**: The final output convolution was zero-initialized. With LoRA, this layer was frozen at zeros — the UNet permanently output `0.0` for everything.

2. **`out_proj` zero-init (attention.py)**: Every attention module's output projection was zero-initialized. Since LoRA injected into these layers couldn't override the frozen zero base weights, all 16 attention modules were permanent no-ops.

3. **`conv2` zero-init (blocks.py)**: Every ResBlock's second convolution was zero-initialized and frozen. The main path of every ResBlock was permanently dead, reducing the UNet to a chain of skip connections.

4. **Base weight mismatch**: LoRA weights were trained on specific random base weights, but `sample.py` created a new UNet with *different* random weights. The trained LoRA adapters were useless on the wrong base → garbage output → black images.

> **Note**: Zero-init is a valid technique for pretrained models (like Stable Diffusion) where it makes new components start as identity. But when training from scratch with LoRA where these layers are frozen, zero-init = permanently dead layers.

All zero-inits were removed, but training full UNet was chosen as the simpler and more robust approach.

---

## 2. Classifier-Free Guidance (CFG) Added

### Initial Plan
- Only **classifier guidance** (external gradient-based classifier)
- Required a separately trained `NoisyClassifier` network
- `num_classes = 10` (CIFAR-10 only)

### What Changed
- Added **CFG dropout** during training: 10% of the time, class labels are replaced with a null class (index 10)
- `num_classes = 11` (10 CIFAR-10 + 1 null class)
- Added `--p_uncond 0.1` argument to `train_latent.py`
- Classifier guidance still available as optional

### Why
CFG is simpler (no separate classifier needed), produces better results in modern diffusion models, and doesn't require storing gradients through a classifier during sampling — saving VRAM.

---

## 3. Timesteps Reduced: 1000 → 200

### Initial Plan
- 1000 diffusion timesteps (standard DDPM)

### What Changed
- Default timesteps reduced to **200** in both training and sampling
- Configurable via `--timesteps` argument

### Why
200 steps is sufficient for the 32×32 latent space. Training runs ~5x faster per epoch, and sampling completes in seconds instead of minutes. The model still learns effectively with the shorter schedule.

---

## 4. Default Guidance Scale: 2.0 → 0.0

### Initial Plan
- `guidance_scale = 2.0` (classifier guidance always on)

### What Changed
- Default `guidance_scale = 0.0` (guidance off)
- Added comment: "set to 0.0 first to verify base generation, then increase to 1.5-3.0"
- `sample.py` skips loading the classifier entirely when scale is 0.0

### Why
Debugging lesson learned: when samples come out black, you need to isolate whether the problem is the UNet, the VAE decoding, or the guidance. Starting with guidance=0.0 tests the base model in isolation first.

---

## 5. Sample Generation Fixed (PNG Instead of .pt)

### Initial Plan
- `_generate_samples()` saved raw latent tensors as `.pt` files
- Required manual VAE decoding to see actual images

### What Changed
- `_generate_samples()` now decodes latents through the SD-VAE and saves as **PNG grids**
- VAE is loaded, used, and immediately deleted + `torch.cuda.empty_cache()` (VRAM safety)
- Falls back to saving raw `.pt` if VAE decode fails
- Save path: `runs/latent/samples/step_{N}.png`

### Why
You couldn't tell if training was working by looking at `.pt` files. PNG samples let you visually check progress during training — colored blobs by step 1000 means it's working.

---

## 6. Checkpoint Format Changed

### Initial Plan
- Save only LoRA parameters (`lora_state` dict, ~2MB)
- Separate `base_model.pt` for the frozen base weights
- Checkpoint keys: `lora_state`, `optimizer`, `global_step`, `epoch`

### What Changed
- Save **full model `state_dict`** (~693MB each)
- Checkpoint keys: `model`, `optimizer`, `global_step`, `epoch`, `scaler`, `speed_log`
- Resume checkpoint naming: `checkpoint_{step}.pt` instead of `lora_ckpt_{step}.pt`
- Final checkpoint: `checkpoint_final.pt` instead of `lora_weights_final.pt`
- No more `base_model.pt` (not needed — full model is saved)

### Why
LoRA was removed, so there's no "LoRA-only" state to save. Full model checkpoints are larger but avoid the base-weight-mismatch bug entirely.

---

## 7. VAE Sanity Check Added (New File)

### Initial Plan
- No pre-training verification step

### What Changed
- Added `sanity_check.py` — a 2-minute script that encodes a CIFAR image through the VAE and immediately decodes it
- If the output is black, it tells you to stop before wasting 30 minutes training

### Why
Multiple training runs produced black images, and the root cause wasn't discovered until after the full training completed. This 2-minute check catches broken VAE pipelines immediately.

---

## 8. `sample.py` Simplified

### Initial Plan
- Load base model → inject LoRA → load LoRA weights → load classifier → sample
- Required `--lora_rank` and `--lora_alpha` arguments

### What Changed
- Load full model checkpoint directly (`checkpoint_final.pt` or latest `checkpoint_*.pt`)
- Removed all LoRA-related imports and logic
- Removed `--lora_rank` and `--lora_alpha` arguments
- Classifier loading skipped when `guidance_scale == 0.0`

### Why
Simpler, fewer failure modes. No more base-weight mismatch bugs.

---

## 9. `compare.py` Labels Updated

### Initial Plan
- Labels: "Pixel Baseline" vs "Latent + LoRA"
- Trainable params: "~70M (all)" vs "~1.5M (LoRA)"

### What Changed
- Labels: "Pixel Baseline" vs **"Latent Full UNet"**
- Trainable params: "~115M (all)" vs **"~115M (all)"**
- Title: "Latent is 8.8x faster" (not "Latent-LoRA")

### Why
Reflects the actual training setup — full UNet on both sides, the speedup comes from smaller input (32x32 vs 256x256) + FP16 + 8-bit Adam, not from LoRA.

---

## 10. `config.py` Updated

### Initial Plan
```python
num_classes: int = 10
timesteps: int = 1000
guidance_scale: float = 2.0
```

### What Changed
```python
num_classes: int = 11      # +1 null class for CFG
timesteps: int = 200       # faster training/sampling
guidance_scale: float = 0.0  # verify base first
```

---

## Summary Table

| Aspect | Initial Plan | Final Implementation |
|--------|-------------|---------------------|
| Training strategy | LoRA (557K params) | Full UNet (~115M params) |
| Guidance | Classifier guidance only | CFG dropout + optional classifier |
| Timesteps | 1000 | 200 |
| Default guidance | 2.0 (always on) | 0.0 (off, verify first) |
| num_classes | 10 | 11 (+ null class) |
| Samples during training | Raw `.pt` tensors | Decoded PNG images |
| Checkpoint size | ~2MB (LoRA only) | ~693MB (full model) |
| Checkpoint naming | `lora_ckpt_*.pt` | `checkpoint_*.pt` |
| Sanity check | None | `sanity_check.py` |
| Zero-init (conv2, out_proj) | Yes (standard) | Removed (fatal with LoRA) |
| Actual speedup achieved | — | **8.8x** (4.054s vs 0.458s/iter) |
