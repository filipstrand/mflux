#!/usr/bin/env python3
"""
Tutorial: A/B Attention Comparison - PyTorch Side

This script demonstrates how to use the dedicated A/B checkpoint helpers:

    debug_checkpoint_pytorch_A(...)
    debug_checkpoint_pytorch_B(...)

to bracket the *attention core* in a tiny, self-contained PyTorch example.

The MLX companion script is `tutorial_ab_mlx.py`, which performs the same
computation with MLX using:

    debug_checkpoint_mlx_A(...)
    debug_checkpoint_mlx_B(...)

You can then use the debugger to:
  - Verify that tensors *match* at checkpoint A (inputs to attention), and
  - Inspect any differences at checkpoint B (outputs of attention).

This pairs with:
  - `mflux-debug-pytorch start src/mflux_debugger/examples/tutorial_ab_pytorch.py`
  - `mflux-debug-mlx start src/mflux_debugger/examples/tutorial_ab_mlx.py`
"""

import torch

from mflux_debugger.semantic_checkpoint import (
    debug_checkpoint_pytorch_A,
    debug_checkpoint_pytorch_B,
)


def make_qkv():
    """
    Create tiny, deterministic Q/K/V tensors for attention.

    Shapes:
        q, k, v: [B, S, H, D]
        with B=1, S=4, H=2, D=2
    """
    B, S, H, D = 1, 4, 2, 2
    base = torch.arange(B * S * H * D, dtype=torch.float32).reshape(B, S, H, D) / 10.0
    q = base
    k = base + 0.1
    v = base - 0.2
    return q, k, v


def attention_core(q, k, v):
    """
    Simple scaled dot-product attention, mirroring the layout used in the real code.

    Expects:
        q, k, v: [B, S, H, D]
    Returns:
        attn_output: [B, S, H, D]
    """
    B, S, H, D = q.shape

    # [B, S, H, D] -> [B, H, S, D]
    q_bhsd = q.permute(0, 2, 1, 3)
    k_bhsd = k.permute(0, 2, 1, 3)
    v_bhsd = v.permute(0, 2, 1, 3)

    scale = 1.0 / (D**0.5)

    # [B, H, S, D] x [B, H, D, S] -> [B, H, S, S]
    scores = torch.matmul(q_bhsd, k_bhsd.transpose(-1, -2)) * scale
    weights = torch.softmax(scores, dim=-1)

    # [B, H, S, S] x [B, H, S, D] -> [B, H, S, D]
    out_bhsd = torch.matmul(weights, v_bhsd)

    # Back to [B, S, H, D]
    out = out_bhsd.permute(0, 2, 1, 3).contiguous()
    return out


def main():
    print("🎓 A/B Attention Tutorial (PyTorch)")
    print("=" * 60)

    # 1. Construct Q/K/V
    q, k, v = make_qkv()
    print(f"Q/K/V shapes: {q.shape}, {k.shape}, {v.shape}")

    # 2. Checkpoint A: inputs to attention core
    print("\n📍 PyTorch A: Before attention core (Q/K/V)")
    debug_checkpoint_pytorch_A(
        skip=False,
        metadata={"stage": "A", "framework": "pytorch", "description": "before attention"},
        query=q,
        key=k,
        value=v,
    )

    # 3. Run the attention math
    attn_output = attention_core(q, k, v)
    print(f"Attention output shape: {attn_output.shape}")

    # 4. Checkpoint B: outputs of attention core
    print("\n📍 PyTorch B: After attention core (attn_output)")
    debug_checkpoint_pytorch_B(
        skip=False,
        metadata={"stage": "B", "framework": "pytorch", "description": "after attention"},
        attn_output=attn_output,
    )

    print("\n✅ PyTorch A/B checkpoints hit. Use the debugger to inspect:")
    print("   - At A: query/key/value tensors")
    print("   - At B: attn_output tensor")


if __name__ == "__main__":
    main()
