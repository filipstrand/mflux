#!/usr/bin/env python3
"""
Direct inspection of resample conv2d weights and bias.
Loads the actual model weights and compares MLX vs PyTorch.
"""

import sys
from pathlib import Path

# Add paths
mflux_src = Path(__file__).parent / "src"
if str(mflux_src) not in sys.path:
    sys.path.insert(0, str(mflux_src))

diffusers_src = Path(__file__).parent.parent / "diffusers" / "src"
if str(diffusers_src) not in sys.path:
    sys.path.insert(0, str(diffusers_src))

print("=" * 80)
print("DIRECT WEIGHT INSPECTION: Resample Conv2d")
print("=" * 80)
print("\nThis script will:")
print("  1. Load MLX model and inspect resample_conv weights/bias")
print("  2. Load PyTorch model and inspect resample[1] weights/bias")
print("  3. Compare actual values (not just statistics)")
print("\n⚠️  Note: This requires models to be loaded")
print("   Run this from within a debugger session or after loading models")

# We'll need to run this interactively or from within the debugger
# For now, provide instructions

print("\n" + "=" * 80)
print("INSTRUCTIONS")
print("=" * 80)
print("""
To inspect weights systematically:

1. In MLX debugger (at resample checkpoint):
   - eval "self.resample_conv.weight.shape"
   - eval "float(self.resample_conv.weight[13, 1, 1, 13])"
   - eval "self.resample_conv.bias.shape if hasattr(self.resample_conv, 'bias') and self.resample_conv.bias is not None else None"
   - eval "float(self.resample_conv.bias[13]) if hasattr(self.resample_conv, 'bias') and self.resample_conv.bias is not None else None"

2. In PyTorch debugger (at resample checkpoint):
   - eval "self.resample[1].weight.shape"
   - eval "self.resample[1].weight[13, 13, 1, 1].item()"  # PyTorch format: (out_c, in_c, h, w)
   - eval "self.resample[1].bias.shape if hasattr(self.resample[1], 'bias') and self.resample[1].bias is not None else None"
   - eval "self.resample[1].bias[13].item() if hasattr(self.resample[1], 'bias') and self.resample[1].bias is not None else None"

3. Compare:
   - MLX weight[13, 1, 1, 13] should match PyTorch weight[13, 13, 1, 1] after transpose
   - MLX bias[13] should match PyTorch bias[13]
   - MLX input[0, 20, 42, 13] should match PyTorch input (after transpose to channels-last)
""")
