"""
LoRA (Low-Rank Adaptation) wrapper for ADM UNet attention layers.

Injects trainable low-rank matrices into the Q, K, V, and output projections
of every SelfAttention module while freezing all base model parameters.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021)
"""

import torch
import torch.nn as nn

from .attention import SelfAttention


class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear with low-rank adaptation.

    The original linear layer is frozen and a low-rank decomposition is added:
        output = original(x) + (x @ A @ B) * (alpha / rank)

    Initialization:
        - A: Kaiming-like (small random values)
        - B: Zeros (so initial LoRA output = 0, model starts as pre-trained)

    Only lora_A and lora_B have requires_grad=True.
    """
    def __init__(self, original: nn.Linear, rank: int = 16, alpha: int = 16):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

        # Low-rank adaptation matrices (on same device as original weights)
        device = original.weight.device
        # A: Kaiming-like initialization
        self.lora_A = nn.Parameter(
            torch.randn(in_features, rank, device=device) * (2.0 / in_features) ** 0.5
        )
        # B: Zero initialization (LoRA output starts at zero)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original frozen forward pass
        base_out = self.original(x)
        # LoRA adaptation
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return base_out + lora_out


def inject_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: int = 16,
) -> tuple:
    """
    Inject LoRA adapters into all SelfAttention modules of the UNet.

    Steps:
        1. Freeze ALL model parameters
        2. Find all SelfAttention modules
        3. Replace Q, K, V, and output projections with LoRALinear wrappers
        4. Only LoRA A/B matrices have requires_grad=True

    Args:
        model: The UNet model
        rank: LoRA rank (default: 16)
        alpha: LoRA scaling alpha (default: 16)

    Returns:
        (model, lora_params) where lora_params is a list of trainable parameters
    """
    # Step 1: Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Step 2 & 3: Find SelfAttention modules and inject LoRA
    lora_params = []
    injection_count = 0

    for name, module in model.named_modules():
        if isinstance(module, SelfAttention):
            # Inject into Q, K, V, and output projections
            for proj_name in ['q_proj', 'k_proj', 'v_proj', 'out_proj']:
                original_linear = getattr(module, proj_name)
                lora_layer = LoRALinear(original_linear, rank=rank, alpha=alpha)
                setattr(module, proj_name, lora_layer)

                # Collect trainable LoRA parameters
                lora_params.append(lora_layer.lora_A)
                lora_params.append(lora_layer.lora_B)
                injection_count += 1

    # Print summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"LoRA Injection Summary:")
    print(f"  Injected into {injection_count} linear layers "
          f"({injection_count // 4} attention modules)")
    print(f"  Rank: {rank}, Alpha: {alpha}, Scaling: {alpha/rank:.1f}")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    print(f"  Frozen params:    {total_params - trainable_params:,}")

    return model, lora_params
