#!/usr/bin/env python3
"""
Tutorial: A/B Attention Comparison - MLX Side

This script mirrors `tutorial_ab_pytorch.py` but uses MLX arrays and the
MLX-specific A/B checkpoint helpers:

    debug_checkpoint_mlx_A(...)
    debug_checkpoint_mlx_B(...)

Both scripts construct the *same* Q/K/V tensors and apply the same
attention math. With the debugger, you can:
  - Verify that MLX and PyTorch match at A (inputs to attention), and
  - Inspect any differences at B (outputs of attention).

Pair this with:
  - `mflux-debug-pytorch start src/mflux_debugger/examples/tutorial_ab_pytorch.py`
  - `mflux-debug-mlx start src/mflux_debugger/examples/tutorial_ab_mlx.py`
"""

import mlx.core as mx

from mflux_debugger.semantic_checkpoint import (
    debug_checkpoint_mlx_A,
    debug_checkpoint_mlx_B,
)


def make_qkv():
    """
    Create tiny, deterministic Q/K/V tensors for attention.

    Shapes:
        q, k, v: [B, S, H, D]
        with B=1, S=4, H=2, D=2

    Uses the same numeric pattern as the PyTorch tutorial so that
    values match exactly across frameworks.
    """
    B, S, H, D = 1, 4, 2, 2
    base = mx.arange(B * S * H * D, dtype=mx.float32).reshape(B, S, H, D) / 10.0
    q = base
    k = base + 0.1
    v = base - 0.2
    return q, k, v


def attention_core(q: mx.array, k: mx.array, v: mx.array) -> mx.array:
    """
    Simple scaled dot-product attention, mirroring the layout used in the real code.

    Expects:
        q, k, v: [B, S, H, D]
    Returns:
        attn_output: [B, S, H, D]
    """
    B, S, H, D = q.shape

    # [B, S, H, D] -> [B, H, S, D]
    q_bhsd = mx.transpose(q, (0, 2, 1, 3))
    k_bhsd = mx.transpose(k, (0, 2, 1, 3))
    v_bhsd = mx.transpose(v, (0, 2, 1, 3))

    scale = 1.0 / mx.sqrt(mx.array(D, dtype=q.dtype))

    # [B, H, S, D] x [B, H, D, S] -> [B, H, S, S]
    scores = mx.matmul(q_bhsd, mx.transpose(k_bhsd, (0, 1, 3, 2))) * scale
    weights = mx.softmax(scores, axis=-1)

    # [B, H, S, S] x [B, H, S, D] -> [B, H, S, D]
    out_bhsd = mx.matmul(weights, v_bhsd)

    # Back to [B, S, H, D]
    out = mx.transpose(out_bhsd, (0, 2, 1, 3))
    return out


def main():
    print("🎓 A/B Attention Tutorial (MLX)")
    print("=" * 60)

    # 1. Construct Q/K/V
    q, k, v = make_qkv()
    print(f"Q/K/V shapes: {q.shape}, {k.shape}, {v.shape}")

    # 2. Checkpoint A: inputs to attention core
    print("\n📍 MLX A: Before attention core (Q/K/V)")
    debug_checkpoint_mlx_A(
        skip=False,
        metadata={"stage": "A", "framework": "mlx", "description": "before attention"},
        query=q,
        key=k,
        value=v,
    )

    # 3. Run the attention math
    attn_output = attention_core(q, k, v)
    print(f"Attention output shape: {attn_output.shape}")

    # 4. Checkpoint B: outputs of attention core
    print("\n📍 MLX B: After attention core (attn_output)")
    debug_checkpoint_mlx_B(
        skip=False,
        metadata={"stage": "B", "framework": "mlx", "description": "after attention"},
        attn_output=attn_output,
    )

    print("\n✅ MLX A/B checkpoints hit. Use the debugger to inspect:")
    print("   - At A: query/key/value tensors")
    print("   - At B: attn_output tensor")


if __name__ == "__main__":
    main()
