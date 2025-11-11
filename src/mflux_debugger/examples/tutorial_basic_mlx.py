#!/usr/bin/env python3
"""
Tutorial: Basic Breakpoints with MLX - Loading and Comparing Tensors

This script demonstrates using debug_checkpoint() for semantic breakpoints
and debug_load() to load tensors saved from PyTorch for comparison.

Workflow:
    1. First run: mflux-debug-pytorch tutorial (saves PyTorch tensors)
    2. Then run: mflux-debug-mlx tutorial (loads and compares with MLX)

To start the tutorial:
    mflux-debug-mlx tutorial
"""

# Import at the top of file (best practice)
import mlx.core as mx

from mflux_debugger.semantic_checkpoint import debug_checkpoint
from mflux_debugger.tensor_debug import debug_load


def create_input_tensor():
    """Step 1: Create an input tensor."""
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    return x


def scale_tensor(x):
    """Step 2: Scale the tensor by 2."""
    scaled = x * 2.0
    return scaled


def compute_sum(x):
    """Step 3: Compute sum along axis."""
    result = mx.sum(x, axis=1)
    return result


def main():
    print("🎓 Tutorial: MLX with Checkpoint Debugging")
    print("=" * 50)

    # Step 1: Create input
    print("\n📍 Checkpoint 1: Creating input tensor...")
    x = create_input_tensor()

    # Load PyTorch tensor for comparison - BEFORE checkpoint
    # Note: debug_load() will raise RuntimeError if tensor not found
    # Run 'mflux-debug-pytorch tutorial' first to save the tensor
    try:
        pytorch_x = debug_load("input_tensor")
    except RuntimeError:
        pytorch_x = None

    debug_checkpoint(
        "after_input_creation",
        metadata={"step": 1, "operation": "create_input", "tensor_name": "x"},
    )  # Semantic breakpoint with metadata - more maintainable!
    print(f"MLX input shape: {x.shape}")
    if pytorch_x is not None:
        print("✅ Loaded PyTorch 'input_tensor' for comparison")
        print(f"   Match: {mx.allclose(x, pytorch_x, atol=1e-5)}")

    # Step 2: Scale (skipped checkpoint - won't pause)
    print("\n⏭️  Checkpoint 2: Scaling tensor (skipped)...")
    scaled = scale_tensor(x)

    # Load PyTorch tensor for comparison - BEFORE checkpoint
    try:
        pytorch_scaled = debug_load("scaled_tensor")
    except RuntimeError:
        pytorch_scaled = None

    debug_checkpoint(
        "after_scaling",
        skip=True,  # This checkpoint is SKIPPED - execution continues
        metadata={"step": 2, "operation": "scale", "tensor_name": "scaled"},
    )
    print(f"MLX scaled shape: {scaled.shape}")
    if pytorch_scaled is not None:
        print("✅ Loaded PyTorch 'scaled_tensor' for comparison")
        print(f"   Match: {mx.allclose(scaled, pytorch_scaled, atol=1e-5)}")

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

    # Load PyTorch tensor for comparison - BEFORE checkpoint
    try:
        pytorch_result = debug_load("sum_result")
    except RuntimeError:
        pytorch_result = None

    debug_checkpoint(
        "after_sum",
        metadata={"step": 3, "operation": "sum", "tensor_name": "result", "final": True},
    )  # Semantic breakpoint
    print(f"MLX result shape: {result.shape}")
    print(f"MLX final result: {result}")
    if pytorch_result is not None:
        print("✅ Loaded PyTorch 'sum_result' for comparison")
        print(f"   PyTorch result: {pytorch_result}")
        print(f"   Match: {mx.allclose(result, pytorch_result, atol=1e-5)}")

    print("\n✅ Tutorial complete!")
    print("\n💡 TIP: Run 'mflux-debug-pytorch tutorial' first to save PyTorch tensors,")
    print("   then run this script again to see the comparison!")


if __name__ == "__main__":
    main()
