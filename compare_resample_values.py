#!/usr/bin/env python3
"""
Systematic comparison of MLX vs PyTorch resample conv2d:
1. Input values (before conv2d)
2. Weight values
3. Bias values (if exists)
4. Output values (after conv2d)

Working backwards from divergence point to find where they first differ.
"""

import sys
from pathlib import Path

import numpy as np

from mflux_debugger.tensor_debug import debug_load

# Add mflux to path
mflux_src = Path(__file__).parent / "src"
if str(mflux_src) not in sys.path:
    sys.path.insert(0, str(mflux_src))

print("=" * 80)
print("SYSTEMATIC COMPARISON: MLX vs PyTorch Resample Conv2d")
print("=" * 80)

tensors_dir = Path("mflux_debugger/tensors/latest")

# ============================================================================
# Step 1: Load MLX values we already have
# ============================================================================
print("\n" + "=" * 80)
print("STEP 1: Load MLX Values")
print("=" * 80)

mlx_input = None
mlx_output = None

try:
    mlx_input = debug_load("mlx_resample_after_repeat")
    print(f"✅ MLX input (after repeat, before conv2d) shape: {mlx_input.shape}")
    print("   Format: (batch*t, height, width, channels) = channels-last")
    print(f"   Value at (0, 20, 42, 13): {mlx_input[0, 20, 42, 13]:.10f}")
except RuntimeError as e:
    print(f"❌ Could not load MLX input: {e}")

try:
    mlx_output = debug_load("mlx_resample_after_conv2d")
    print(f"\n✅ MLX output (after conv2d) shape: {mlx_output.shape}")
    print("   Format: (batch*t, height, width, channels) = channels-last")
    print(f"   Value at (0, 20, 42, 13): {mlx_output[0, 20, 42, 13]:.10f}")
except RuntimeError as e:
    print(f"❌ Could not load MLX output: {e}")

# ============================================================================
# Step 2: Load PyTorch values
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Load PyTorch Values")
print("=" * 80)

# Check what PyTorch files we have
pytorch_files = sorted([f.name for f in tensors_dir.glob("pytorch_*.npy")])
print(f"\nAvailable PyTorch checkpoints ({len(pytorch_files)}):")
for f in pytorch_files[:10]:  # Show first 10
    print(f"  - {f}")
if len(pytorch_files) > 10:
    print(f"  ... and {len(pytorch_files) - 10} more")

# Try to load PyTorch resample values
pytorch_input = None
pytorch_output = None

# Check for resample-specific checkpoints
if "pytorch_resample_after_upsample.npy" in pytorch_files:
    pytorch_input = np.load(tensors_dir / "pytorch_resample_after_upsample.npy")
    print(f"\n✅ PyTorch input (after upsample, before conv2d) shape: {pytorch_input.shape}")
    print("   Format: (batch*t, channels, height, width) = channels-first")
    # Convert to channels-last for comparison
    pytorch_input_cl = pytorch_input.transpose(0, 2, 3, 1)
    print(f"   After transpose to channels-last: {pytorch_input_cl.shape}")
    print(f"   Value at (0, 20, 42, 13): {pytorch_input_cl[0, 20, 42, 13]:.10f}")
else:
    print("\n⚠️  PyTorch resample_after_upsample not found")
    print("   This checkpoint should be saved by PyTorch debug_save calls")

if "pytorch_resample_after_conv2d.npy" in pytorch_files:
    pytorch_output = np.load(tensors_dir / "pytorch_resample_after_conv2d.npy")
    print(f"\n✅ PyTorch output (after conv2d) shape: {pytorch_output.shape}")
    print("   Format: (batch*t, channels, height, width) = channels-first")
    # Convert to channels-last for comparison
    pytorch_output_cl = pytorch_output.transpose(0, 2, 3, 1)
    print(f"   After transpose to channels-last: {pytorch_output_cl.shape}")
    print(f"   Value at (0, 20, 42, 13): {pytorch_output_cl[0, 20, 42, 13]:.10f}")
else:
    print("\n⚠️  PyTorch resample_after_conv2d not found")
    print("   This checkpoint should be saved by PyTorch debug_save calls")

# ============================================================================
# Step 3: Compare Inputs
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Compare Inputs (Before Conv2d)")
print("=" * 80)

if mlx_input is not None and pytorch_input is not None:
    pytorch_input_cl = pytorch_input.transpose(0, 2, 3, 1)

    # Compare shapes
    print("\nShapes:")
    print(f"  MLX: {mlx_input.shape}")
    print(f"  PyTorch (transposed): {pytorch_input_cl.shape}")
    print(f"  ✅ Shapes match: {mlx_input.shape == pytorch_input_cl.shape}")

    if mlx_input.shape == pytorch_input_cl.shape:
        # Compare actual values
        diff = np.abs(mlx_input - pytorch_input_cl)
        max_diff = np.max(diff)
        max_diff_loc = np.unravel_index(np.argmax(diff), diff.shape)

        print("\nValue comparison:")
        print(f"  MLX at (0, 20, 42, 13): {mlx_input[0, 20, 42, 13]:.10f}")
        print(f"  PyTorch at (0, 20, 42, 13): {pytorch_input_cl[0, 20, 42, 13]:.10f}")
        print(f"  Difference: {abs(mlx_input[0, 20, 42, 13] - pytorch_input_cl[0, 20, 42, 13]):.10f}")
        print(f"\nMax difference: {max_diff:.10f} at location {max_diff_loc}")
        print(f"  MLX value: {mlx_input[max_diff_loc]:.10f}")
        print(f"  PyTorch value: {pytorch_input_cl[max_diff_loc]:.10f}")

        if max_diff < 1e-5:
            print("\n✅ INPUTS MATCH - Values are identical!")
        else:
            print("\n❌ INPUTS DON'T MATCH - Divergence starts BEFORE conv2d!")
            print("   Need to go back further to find where divergence starts")
else:
    print("\n⚠️  Cannot compare inputs - missing data")

# ============================================================================
# Step 4: Compare Outputs
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Compare Outputs (After Conv2d)")
print("=" * 80)

if mlx_output is not None and pytorch_output is not None:
    pytorch_output_cl = pytorch_output.transpose(0, 2, 3, 1)

    # Compare shapes
    print("\nShapes:")
    print(f"  MLX: {mlx_output.shape}")
    print(f"  PyTorch (transposed): {pytorch_output_cl.shape}")
    print(f"  ✅ Shapes match: {mlx_output.shape == pytorch_output_cl.shape}")

    if mlx_output.shape == pytorch_output_cl.shape:
        # Compare actual values
        diff = np.abs(mlx_output - pytorch_output_cl)
        max_diff = np.max(diff)
        max_diff_loc = np.unravel_index(np.argmax(diff), diff.shape)

        print("\nValue comparison:")
        print(f"  MLX at (0, 20, 42, 13): {mlx_output[0, 20, 42, 13]:.10f}")
        print(f"  PyTorch at (0, 20, 42, 13): {pytorch_output_cl[0, 20, 42, 13]:.10f}")
        print(f"  Difference: {abs(mlx_output[0, 20, 42, 13] - pytorch_output_cl[0, 20, 42, 13]):.10f}")
        print(f"\nMax difference: {max_diff:.10f} at location {max_diff_loc}")
        print(f"  MLX value: {mlx_output[max_diff_loc]:.10f}")
        print(f"  PyTorch value: {pytorch_output_cl[max_diff_loc]:.10f}")

        print("\n❌ OUTPUTS DON'T MATCH - This is the divergence point!")
else:
    print("\n⚠️  Cannot compare outputs - missing data")
    if mlx_output is not None:
        print(f"   MLX output at (0, 20, 42, 13): {mlx_output[0, 20, 42, 13]:.10f}")
    # We know PyTorch final output from earlier debugging
    print("   PyTorch final output at (0, 13, 0, 20, 42): -4.687500 (from earlier)")

# ============================================================================
# Step 5: Load and Compare Weights
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Load and Compare Weights")
print("=" * 80)

print("\n⚠️  Need to load weights from actual model files")
print("   This requires accessing the loaded model weights")
print("   Will need to use debugger to inspect self.resample_conv.weight")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print("""
Next steps:
1. If inputs DON'T match: Go back further to find where divergence starts
2. If inputs DO match: Check weights and bias
3. If weights/bias match: Then conv2d computation is wrong
4. If weights/bias don't match: Fix weight loading/transpose
""")
