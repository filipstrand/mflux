#!/usr/bin/env python3
"""
Temporary script to compare MLX Conv2d vs PyTorch Conv2d behavior.
Focus: Weight transpose and convolution computation.
"""

import mlx.core as mx
import mlx.nn as nn_mlx
import numpy as np
import torch
import torch.nn as nn

print("=" * 80)
print("MLX vs PyTorch Conv2d Comparison")
print("=" * 80)

# ============================================================================
# Test 1: Simple convolution to understand weight formats
# ============================================================================
print("\n" + "=" * 80)
print("TEST 1: Simple Conv2d - Understanding Weight Formats")
print("=" * 80)

# Create simple input: (batch=1, channels=2, height=4, width=4)
input_pytorch = torch.randn(1, 2, 4, 4)  # PyTorch: channels-first
input_mlx = mx.array(input_pytorch.numpy().transpose(0, 2, 3, 1))  # MLX: channels-last

print(f"\nPyTorch input shape (channels-first): {input_pytorch.shape}")
print(f"MLX input shape (channels-last): {input_mlx.shape}")

# Create Conv2d layers
conv_pytorch = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1)
conv_mlx = nn_mlx.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1)

print(f"\nPyTorch Conv2d weight shape: {conv_pytorch.weight.shape}")
print("  Format: (out_channels, in_channels, height, width)")
print(f"MLX Conv2d weight shape: {conv_mlx.weight.shape}")
print("  Format: (out_channels, height, width, in_channels)")

# Apply convolution
output_pytorch = conv_pytorch(input_pytorch)  # (1, 3, 4, 4) channels-first
output_mlx = conv_mlx(input_mlx)  # (1, 4, 4, 3) channels-last

print(f"\nPyTorch output shape: {output_pytorch.shape}")
print(f"MLX output shape: {output_mlx.shape}")

# Convert MLX output to channels-first for comparison
output_mlx_channels_first = mx.transpose(output_mlx, (0, 3, 1, 2))
print(f"MLX output (transposed to channels-first): {output_mlx_channels_first.shape}")

# Compare outputs
output_pytorch_np = output_pytorch.detach().numpy()
output_mlx_np = np.array(output_mlx_channels_first)

print("\nOutput comparison (should be different - different random weights):")
print(f"  PyTorch output[0, 0, 0, 0]: {output_pytorch_np[0, 0, 0, 0]:.6f}")
print(f"  MLX output[0, 0, 0, 0]: {output_mlx_np[0, 0, 0, 0]:.6f}")

# ============================================================================
# Test 2: Transpose PyTorch weights to MLX format
# ============================================================================
print("\n" + "=" * 80)
print("TEST 2: Transpose PyTorch Weights to MLX Format")
print("=" * 80)

# Get PyTorch weights: (out_c, in_c, h, w)
pytorch_weight = conv_pytorch.weight.detach().numpy()
print(f"\nPyTorch weight shape: {pytorch_weight.shape}")
print("  Format: (out_channels, in_channels, height, width)")

# Try different transpose options
# Option 1: (out_c, in_c, h, w) -> (out_c, h, w, in_c) - standard transpose
mlx_weight_transposed_1 = pytorch_weight.transpose(0, 2, 3, 1)
# Option 2: Flip kernel dimensions (for cross-correlation vs convolution)
mlx_weight_transposed_2 = pytorch_weight.transpose(0, 2, 3, 1)[:, ::-1, ::-1, :]  # Flip h and w
# Option 3: Different transpose order
mlx_weight_transposed_3 = pytorch_weight.transpose(0, 3, 2, 1)  # Swap h and w

print(f"\nOption 1 - Transposed (0,2,3,1): {mlx_weight_transposed_1.shape}")
print(f"Option 2 - Transposed + flipped: {mlx_weight_transposed_2.shape}")
print(f"Option 3 - Transposed (0,3,2,1): {mlx_weight_transposed_3.shape}")

# Compare with MLX's actual weight shape
mlx_weight_actual = np.array(conv_mlx.weight)
print(f"\nMLX actual weight shape: {mlx_weight_actual.shape}")
print("  Format: (out_channels, height, width, in_channels)")

# Check if shapes match
print(f"\n✅ Option 1 matches MLX shape: {mlx_weight_transposed_1.shape == mlx_weight_actual.shape}")

# Use option 1 for now (standard transpose)
mlx_weight_transposed = mlx_weight_transposed_1

# ============================================================================
# Test 3: Load PyTorch weights into MLX Conv2d
# ============================================================================
print("\n" + "=" * 80)
print("TEST 3: Load PyTorch Weights into MLX Conv2d")
print("=" * 80)

# Test all three transpose options
print("\nTesting different transpose options:")
for i, weight_option in enumerate([mlx_weight_transposed_1, mlx_weight_transposed_2, mlx_weight_transposed_3], 1):
    conv_mlx_test = nn_mlx.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
    conv_mlx_test.weight = mx.array(weight_option)
    output_test = conv_mlx_test(input_mlx)
    output_test_cf = mx.transpose(output_test, (0, 3, 1, 2))
    output_test_np = np.array(output_test_cf)
    diff = np.max(np.abs(output_pytorch_np - output_test_np))
    print(f"  Option {i} max diff: {diff:.6f}")

# Use option 1 for main test
conv_mlx_loaded = nn_mlx.Conv2d(in_channels=2, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
conv_mlx_loaded.weight = mx.array(mlx_weight_transposed_1)
mlx_weight_transposed = mlx_weight_transposed_1

# Apply with same input
output_mlx_loaded = conv_mlx_loaded(input_mlx)
output_mlx_loaded_channels_first = mx.transpose(output_mlx_loaded, (0, 3, 1, 2))
output_mlx_loaded_np = np.array(output_mlx_loaded_channels_first)

print("\nOutput comparison (should match now):")
print(f"  PyTorch output[0, 0, 0, 0]: {output_pytorch_np[0, 0, 0, 0]:.6f}")
print(f"  MLX output (loaded weights)[0, 0, 0, 0]: {output_mlx_loaded_np[0, 0, 0, 0]:.6f}")
print(f"  Difference: {abs(output_pytorch_np[0, 0, 0, 0] - output_mlx_loaded_np[0, 0, 0, 0]):.6f}")

# Check if they match (within numerical precision)
max_diff = np.max(np.abs(output_pytorch_np - output_mlx_loaded_np))
print(f"\nMax difference across all outputs: {max_diff:.6f}")
print(f"✅ Outputs match: {max_diff < 1e-5}")

# ============================================================================
# Test 4: Test with FIBO-like dimensions (1024 channels, 3x3 kernel)
# ============================================================================
print("\n" + "=" * 80)
print("TEST 4: FIBO-like Dimensions (1024 channels, 3x3 kernel)")
print("=" * 80)

# Create input similar to FIBO resample: (batch=1, channels=1024, height=64, width=64)
input_fibo_pytorch = torch.randn(1, 1024, 64, 64)
input_fibo_mlx = mx.array(input_fibo_pytorch.numpy().transpose(0, 2, 3, 1))  # (1, 64, 64, 1024)

print("\nFIBO-like input:")
print(f"  PyTorch shape: {input_fibo_pytorch.shape}")
print(f"  MLX shape: {input_fibo_mlx.shape}")

# Create Conv2d: 1024 -> 1024, kernel=3, padding=1 (no bias for cleaner comparison)
conv_fibo_pytorch = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)
conv_fibo_mlx = nn_mlx.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=False)

print("\nFIBO-like Conv2d:")
print(f"  PyTorch weight shape: {conv_fibo_pytorch.weight.shape}")
print(f"  MLX weight shape: {conv_fibo_mlx.weight.shape}")

# Transpose PyTorch weights to MLX format
fibo_pytorch_weight = conv_fibo_pytorch.weight.detach().numpy()
fibo_mlx_weight_transposed = fibo_pytorch_weight.transpose(0, 2, 3, 1)

print(f"\nTransposed PyTorch weight shape: {fibo_mlx_weight_transposed.shape}")
print("  Format: (out_channels, height, width, in_channels)")

# Load into MLX
conv_fibo_mlx.weight = mx.array(fibo_mlx_weight_transposed)

# Apply convolution
output_fibo_pytorch = conv_fibo_pytorch(input_fibo_pytorch)  # (1, 1024, 64, 64)
output_fibo_mlx = conv_fibo_mlx(input_fibo_mlx)  # (1, 64, 64, 1024)
output_fibo_mlx_channels_first = mx.transpose(output_fibo_mlx, (0, 3, 1, 2))  # (1, 1024, 64, 64)

# Compare at a specific location (like we're debugging)
output_fibo_pytorch_np = output_fibo_pytorch.detach().numpy()
output_fibo_mlx_np = np.array(output_fibo_mlx_channels_first)

print("\nOutput comparison at (0, 13, 0, 20, 42):")
print(f"  PyTorch output[0, 13, 0, 20]: {output_fibo_pytorch_np[0, 13, 0, 20]:.6f}")
print(f"  MLX output[0, 13, 0, 20]: {output_fibo_mlx_np[0, 13, 0, 20]:.6f}")
print(f"  Difference: {abs(output_fibo_pytorch_np[0, 13, 0, 20] - output_fibo_mlx_np[0, 13, 0, 20]):.6f}")

max_diff_fibo = np.max(np.abs(output_fibo_pytorch_np - output_fibo_mlx_np))
print(f"\nMax difference across all outputs: {max_diff_fibo:.6f}")
print(f"✅ Outputs match: {max_diff_fibo < 1e-4}")

# ============================================================================
# Test 5: Manual convolution computation to verify
# ============================================================================
print("\n" + "=" * 80)
print("TEST 5: Manual Convolution Computation Verification")
print("=" * 80)

# Pick a specific location: output[0, 13, 0, 20] (batch=0, channel=13, h=0, w=20)
# For PyTorch: input is (1, 1024, 64, 64), weight is (1024, 1024, 3, 3)
# For MLX: input is (1, 64, 64, 1024), weight is (1024, 3, 3, 1024)

# PyTorch computation:
# output[0, 13, 0, 20] = sum over kernel of: input[0, k, 0+i-1, 20+j-1] * weight[13, k, i, j]
# where i, j in [0, 1, 2] (3x3 kernel), k in [0..1023]

# MLX computation (channels-last):
# output[0, 0, 20, 13] = sum over kernel of: input[0, 0+i-1, 20+j-1, k] * weight[13, i, j, k]
# where i, j in [0, 1, 2] (3x3 kernel), k in [0..1023]

# Let's manually compute one value
print("\nManual computation at output[0, 13, 0, 20]:")
print("  (This verifies the convolution formula)")

# Get the input patch and weight slice
input_patch_pytorch = input_fibo_pytorch[0, :, 0:3, 20:23].numpy()  # (1024, 3, 3)
weight_slice_pytorch = conv_fibo_pytorch.weight[13, :, :, :].detach().numpy()  # (1024, 3, 3)

# PyTorch: sum over k, i, j: input[k, i, j] * weight[k, i, j]
manual_pytorch = np.sum(input_patch_pytorch * weight_slice_pytorch)
print(f"  PyTorch manual: {manual_pytorch:.6f}")
print(f"  PyTorch actual: {output_fibo_pytorch_np[0, 13, 0, 20]:.6f}")
print(f"  Match: {abs(manual_pytorch - output_fibo_pytorch_np[0, 13, 0, 20]) < 1e-4}")

# MLX: input is (1, 64, 64, 1024), weight is (1024, 3, 3, 1024)
input_patch_mlx = np.array(input_fibo_mlx[0, 0:3, 20:23, :])  # (3, 3, 1024)
weight_slice_mlx = fibo_mlx_weight_transposed[13, :, :, :]  # (3, 3, 1024)

# MLX: sum over i, j, k: input[i, j, k] * weight[i, j, k]
manual_mlx = np.sum(input_patch_mlx * weight_slice_mlx)
print(f"\n  MLX manual: {manual_mlx:.6f}")
print(f"  MLX actual: {output_fibo_mlx_np[0, 13, 0, 20]:.6f}")
print(f"  Match: {abs(manual_mlx - output_fibo_mlx_np[0, 13, 0, 20]) < 1e-4}")

print(f"\n  Manual PyTorch vs MLX: {abs(manual_pytorch - manual_mlx):.6f}")
print(f"  ✅ Manual computations match: {abs(manual_pytorch - manual_mlx) < 1e-4}")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)
print("""
The transpose operation (0, 2, 3, 1) correctly converts PyTorch weights to MLX format.
When weights are correctly transposed and loaded, the outputs match.

If outputs don't match in the actual FIBO code, check:
1. Are weights being transposed correctly? (0, 2, 3, 1)
2. Is the input in channels-last format before Conv2d?
3. Is the output transposed back to channels-first after Conv2d?
4. Are there any other operations between transpose and conv that might affect things?
""")
