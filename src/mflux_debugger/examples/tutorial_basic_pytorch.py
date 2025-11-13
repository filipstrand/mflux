#!/usr/bin/env python3
"""
Tutorial: Basic Breakpoints with PyTorch - Saving Tensors

This script demonstrates using debug_checkpoint() for semantic breakpoints
and debug_save() to save PyTorch tensors for comparison with MLX.

Run this first, then run tutorial_basic_mlx.py to load and compare.

To start the tutorial:
    mflux-debug-pytorch tutorial

IMPORTANT: debug_save() and debug_load() work from ANYWHERE!
- You can call them from mflux code, diffusers code, transformers code, or any script
- They automatically find the correct mflux_debugger directory
- No path manipulation or configuration needed - just import and use!
"""

# Import at the top of file (best practice)
import torch

from mflux_debugger.semantic_checkpoint import debug_checkpoint
from mflux_debugger.tensor_debug import debug_save


def create_input_tensor():
    """Step 1: Create an input tensor."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    # Save tensor for comparison with MLX
    debug_save(x, "input_tensor")

    return x


def scale_tensor(x):
    """Step 2: Scale the tensor by 2."""
    scaled = x * 2.0

    # Save intermediate result
    debug_save(scaled, "scaled_tensor")

    return scaled


def compute_sum(x):
    """Step 3: Compute sum along axis."""
    result = torch.sum(x, dim=1)

    # Save final result
    debug_save(result, "sum_result")

    return result


def main():
    print("🎓 Tutorial: PyTorch with Checkpoints & Saving")
    print("=" * 50)

    # Step 1: Create input
    print("\n📍 Checkpoint 1: Creating input tensor...")
    x = create_input_tensor()
    debug_checkpoint(
        "after_input_creation",
        metadata={"step": 1, "operation": "create_input", "tensor_name": "x"},
    )  # Semantic breakpoint with metadata
    print(f"PyTorch input shape: {x.shape}")
    print("✅ Saved 'input_tensor' for MLX comparison")

    # Step 2: Scale (skipped checkpoint - won't pause)
    print("\n⏭️  Checkpoint 2: Scaling tensor (skipped)...")
    scaled = scale_tensor(x)
    debug_checkpoint(
        "after_scaling",
        skip=True,  # This checkpoint is SKIPPED - execution continues
        metadata={"step": 2, "operation": "scale", "tensor_name": "scaled"},
    )
    print(f"PyTorch scaled shape: {scaled.shape}")
    print("✅ Saved 'scaled_tensor' for MLX comparison")

    # Step 2.5: Loop with conditional breakpoints (demonstrates skip)
    print("\n🔁 Loop Example: Processing 5 iterations...")
    print("   (Will pause only on iteration 3, skip others)")
    for i in range(5):
        # Process the data
        temp = scaled * (i + 1)

        # Checkpoint with conditional skip - only break on iteration 3
        debug_checkpoint(
            "loop_iteration",
            skip=(i != 3),  # Skip all except iteration 3
            metadata={"iteration": i, "multiplier": i + 1},
        )

        if i == 3:
            print(f"   📍 Paused at iteration {i} - temp shape: {temp.shape}")
        else:
            print(f"   ⏭️  Skipped iteration {i}")

    print("✅ Loop complete!")

    # Step 3: Compute sum
    print("\n📍 Checkpoint 3: Computing sum...")
    result = compute_sum(scaled)
    debug_checkpoint(
        "after_sum",
        metadata={"step": 3, "operation": "sum", "tensor_name": "result", "final": True},
    )  # Semantic breakpoint
    print(f"PyTorch result shape: {result.shape}")
    print(f"PyTorch final result: {result}")
    print("✅ Saved 'sum_result' for MLX comparison")

    print("\n✅ Tutorial complete!")
    print("\n💡 Next: Run 'mflux-debug-mlx tutorial' to load these tensors in MLX")


if __name__ == "__main__":
    main()
