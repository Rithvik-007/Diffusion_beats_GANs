"""
Gaussian Diffusion (DDPM) with classifier guidance support.

Implements:
  - Forward process (q_sample): adding noise at timestep t
  - Training loss: MSE between predicted and actual noise
  - Reverse process (p_sample): single denoising step with optional classifier guidance
  - Full sampling loop (p_sample_loop): iterative denoising T → 0

Reference:
  - Ho et al., "Denoising Diffusion Probabilistic Models" (2020)
  - Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .schedule import get_schedule


class GaussianDiffusion:
    """
    DDPM forward and reverse diffusion processes with classifier guidance.
    """
    def __init__(self, schedule_name: str = "linear", timesteps: int = 1000):
        self.timesteps = timesteps
        schedule = get_schedule(schedule_name, timesteps)

        # Store all schedule tensors (will be moved to device on first use)
        self.betas = schedule["betas"]
        self.alphas = schedule["alphas"]
        self.alpha_bars = schedule["alpha_bars"]
        self.sqrt_alpha_bars = schedule["sqrt_alpha_bars"]
        self.sqrt_one_minus_alpha_bars = schedule["sqrt_one_minus_alpha_bars"]
        self.sqrt_recip_alphas = schedule["sqrt_recip_alphas"]
        self.posterior_variance = schedule["posterior_variance"]
        self.posterior_log_variance = schedule["posterior_log_variance"]
        self.posterior_mean_coeff1 = schedule["posterior_mean_coeff1"]
        self.posterior_mean_coeff2 = schedule["posterior_mean_coeff2"]

        self._device = None

    def _to_device(self, device):
        """Move schedule tensors to the specified device (lazy, once)."""
        if self._device != device:
            self._device = device
            self.betas = self.betas.to(device)
            self.alphas = self.alphas.to(device)
            self.alpha_bars = self.alpha_bars.to(device)
            self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
            self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
            self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
            self.posterior_variance = self.posterior_variance.to(device)
            self.posterior_log_variance = self.posterior_log_variance.to(device)
            self.posterior_mean_coeff1 = self.posterior_mean_coeff1.to(device)
            self.posterior_mean_coeff2 = self.posterior_mean_coeff2.to(device)

    def _extract(self, schedule_tensor: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """
        Extract values from a schedule tensor at timestep t and reshape for broadcasting.

        Args:
            schedule_tensor: (T,) schedule values
            t: (B,) timestep indices
            x_shape: Shape of x for broadcasting

        Returns:
            (B, 1, 1, 1) extracted values
        """
        batch_size = t.shape[0]
        vals = schedule_tensor.gather(0, t)
        return vals.reshape(batch_size, *([1] * (len(x_shape) - 1)))

    # ==================== Forward Process ====================

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise at timestep t.

        q(x_t | x_0) = √ᾱ_t · x_0 + √(1 - ᾱ_t) · ε

        Args:
            x_0: Clean data (B, C, H, W)
            t: (B,) timestep indices
            noise: Optional pre-sampled noise

        Returns:
            x_t: Noisy data at timestep t
        """
        self._to_device(x_0.device)

        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self._extract(self.sqrt_alpha_bars, t, x_0.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alpha_bars, t, x_0.shape)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    # ==================== Training ====================

    def training_loss(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the training loss: MSE between predicted and actual noise.

        Args:
            model: UNet noise predictor
            x_0: Clean data (B, C, H, W)
            t: (B,) timestep indices
            class_labels: (B,) class labels

        Returns:
            Scalar MSE loss
        """
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # Predict noise
        noise_pred = model(x_t, t, class_labels)

        # Simple MSE loss
        loss = F.mse_loss(noise_pred, noise)
        return loss

    # ==================== Reverse Process ====================

    def p_sample(
        self,
        model: nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
        class_labels: torch.Tensor,
        classifier: nn.Module = None,
        guidance_scale: float = 0.0,
    ) -> torch.Tensor:
        """
        Single reverse diffusion step with optional classifier guidance.

        For classifier guidance:
            ε̂ = ε_θ(x_t, t) - s · σ_t · ∇_{x_t} log p_φ(y | x_t)

        Args:
            model: UNet noise predictor
            x_t: Noisy data at timestep t (B, C, H, W)
            t: (B,) timestep indices (same value for all in batch)
            class_labels: (B,) target class labels
            classifier: Optional auxiliary classifier for guidance
            guidance_scale: Classifier guidance strength (s)

        Returns:
            x_{t-1}: Denoised data at timestep t-1
        """
        self._to_device(x_t.device)

        # Predict noise with the UNet
        with torch.no_grad():
            noise_pred = model(x_t, t, class_labels)

        # Classifier guidance
        if classifier is not None and guidance_scale > 0:
            with torch.enable_grad():
                x_in = x_t.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[torch.arange(len(class_labels)), class_labels]
                grad = torch.autograd.grad(selected.sum(), x_in)[0]

            # Steer noise prediction
            sigma_t = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)
            noise_pred = noise_pred - guidance_scale * sigma_t * grad

        # Compute x_{t-1} using DDPM posterior
        sqrt_recip_alpha = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        beta = self._extract(self.betas, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self._extract(self.sqrt_one_minus_alpha_bars, t, x_t.shape)

        # Predicted mean
        pred_mean = sqrt_recip_alpha * (x_t - beta / sqrt_one_minus_alpha_bar * noise_pred)

        if t[0] > 0:
            # Add noise for all steps except the last
            posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
            noise = torch.randn_like(x_t)
            return pred_mean + torch.sqrt(posterior_var) * noise
        else:
            return pred_mean

    def p_sample_loop(
        self,
        model: nn.Module,
        shape: tuple,
        class_labels: torch.Tensor,
        device: torch.device,
        classifier: nn.Module = None,
        guidance_scale: float = 0.0,
        verbose: bool = True,
    ) -> torch.Tensor:
        """
        Full reverse diffusion loop: T → 0.

        Starts from pure Gaussian noise and iteratively denoises.

        Args:
            model: UNet noise predictor
            shape: (B, C, H, W) shape for the generated samples
            class_labels: (B,) target class labels
            device: CUDA/CPU device
            classifier: Optional auxiliary classifier for guidance
            guidance_scale: Classifier guidance strength
            verbose: Whether to show progress bar

        Returns:
            x_0: Denoised output (B, C, H, W)
        """
        model.eval()
        if classifier is not None:
            classifier.eval()

        # Start from pure noise
        with torch.no_grad():
            x_t = torch.randn(shape, device=device)

        timestep_range = reversed(range(self.timesteps))
        if verbose:
            timestep_range = tqdm(timestep_range, desc="Sampling", total=self.timesteps)

        for t_val in timestep_range:
            t = torch.full((shape[0],), t_val, device=device, dtype=torch.long)
            x_t = self.p_sample(model, x_t, t, class_labels, classifier, guidance_scale)

        return x_t
