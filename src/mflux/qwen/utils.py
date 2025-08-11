"""
Utility functions for Qwen image generation.
"""

import mlx.core as mx
import torch


def load_pt_tensor(path: str) -> torch.Tensor:
    """Load a PyTorch tensor from file."""
    t = torch.load(path, map_location="cpu")
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"Expected a single Tensor in {path}, got {type(t)}")
    return t


def torch_to_mx(x: torch.Tensor, dtype: mx.Dtype | None = None) -> mx.array:
    """Convert PyTorch tensor to MLX array."""
    if x.dtype in (torch.bfloat16, torch.float16):
        x = x.to(torch.float32)
    out = mx.array(x.detach().cpu().numpy())
    return out.astype(dtype) if dtype is not None else out
