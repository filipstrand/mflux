# FIBO VAE Decoder Debugging - Current State

## Overview
Debugging divergence between MLX and PyTorch implementations of the FIBO VAE decoder, specifically focusing on the `up_block_0` resample operation.

## Key Findings

### Divergence Point
- **Location**: `up_block_0` resample conv2d operation
- **Checkpoint**: `pytorch_decoder_after_up_block_0` vs `mlx_decoder_after_up_block_0`
- **Max Difference**: 4.379446 at location `(0, 13, 0, 20, 42)`
- **Values**:
  - PyTorch: `-4.687500`
  - MLX: `-0.308054`

### Verified Correct
1. **Channel Count**: Fixed - MLX now uses `upsample_out_dim=out_dim` (1024) instead of defaulting to `dim // 2` (512)
2. **Weights**: Verified correct - shape `[1024, 3, 3, 1024]` matches PyTorch after transpose
3. **Input Format**: Correct - MLX uses channels-last `(b*t, h, w, c)` for Conv2d
4. **Weight Transpose**: Correct - `(0, 2, 3, 1)` maps `(out_c, in_c, h, w) -> (out_c, h, w, in_c)`

### Divergence Analysis

#### Before Resample
- `pytorch_decoder_after_mid_block` vs `mlx_decoder_after_mid_block`: Max diff `0.035532` ✅ (acceptable)
- Input to resample matches well

#### During Resample
- **After Upsampling**:
  - PyTorch: Uses `nn.Upsample` with `mode='nearest-exact'`
  - MLX: Uses `mx.repeat(x, 2, axis=1)` and `mx.repeat(x, 2, axis=2)`
  - **Status**: Should be equivalent for 2x upsampling, but not verified

- **After Conv2d**:
  - **Major divergence occurs here**
  - Input to conv2d: `-1.041270` (both match)
  - MLX output: `-0.308054`
  - PyTorch output: `-4.687500`
  - **Difference**: `4.379446` - This is where the problem is

#### After Resample
- Divergence compounds through subsequent up_blocks
- `up_block_1`: Max diff `52.356827`
- `up_block_2`: Max diff `42.447765`

## Files Modified

### MLX Implementation
1. **`wan_up_block.py`**: Fixed to pass `upsample_out_dim=out_dim` to `WanResample`
2. **`wan_resample.py`**: Added extensive debug checkpoints for `block_idx == 0`

### PyTorch Implementation
1. **`autoencoder_kl_wan.py`**: Added debug checkpoints in `WanResample.forward()` (but not saving - needs investigation)
2. **`pipeline_bria_fibo.py`**: Temporary modifications to skip denoising loop and load saved tensor (NOT COMMITTED)

## Debug Checkpoints

### MLX Checkpoints (Working)
- `mlx_resample_before_upsample_conv`: Input before upsampling
- `mlx_resample_after_repeat`: After `mx.repeat` upsampling
- `mlx_resample_conv2d_weights`: Weight metadata
- `mlx_resample_before_conv2d_compute`: Input to conv2d
- `mlx_resample_after_conv2d_compute`: Output from conv2d
- `mlx_resample_after_conv2d`: Final resample output

### PyTorch Checkpoints (Not Saving)
- `pytorch_resample_after_upsample`: Should exist but not being saved
- `pytorch_resample_after_conv2d`: Should exist but not being saved
- **Issue**: `debug_save` calls in `WanResample.forward()` not executing
- **Possible causes**:
  - `block_idx` not being passed correctly
  - `isinstance(self.resample, nn.Sequential)` check failing
  - `debug_save` import failing silently

## Key Code Locations

### MLX Resample Conv2d
**File**: `src/mflux/models/fibo/model/fibo_vae/decoder/wan_resample.py`
- Line 84-85: `mx.repeat` upsampling
- Line 127: `self.resample_conv(x)` - **Divergence occurs here**

### PyTorch Resample Conv2d
**File**: `diffusers/src/diffusers/models/autoencoders/autoencoder_kl_wan.py`
- Line 435: `WanUpsample` (uses `nn.Upsample` with `mode='nearest-exact'`)
- Line 438: `self.resample[1](x)` - Conv2d operation

## Weight Mapping

**File**: `src/mflux/models/fibo/weights/fibo_weight_mapping.py`
- Line 167-170: Resample conv2d weight mapping
- Pattern: `decoder.up_blocks.{block}.upsampler.resample.1.weight` -> `decoder.up_blocks.{block}.upsampler.resample_conv.weight`
- Transform: `transpose_conv2d_weight` (line 30-37)

## Next Steps

1. **Investigate MLX Conv2d computation**:
   - Compare actual convolution computation with PyTorch
   - Check if padding behavior differs
   - Verify weight application matches PyTorch

2. **Fix PyTorch debug checkpoints**:
   - Ensure `block_idx=0` is passed to resample
   - Verify `debug_save` imports work correctly
   - Get intermediate PyTorch checkpoints for comparison

3. **Compare upsampling methods**:
   - Verify `mx.repeat` produces same result as `nn.Upsample` with `mode='nearest-exact'`
   - Check if there's a difference in how values are duplicated

4. **Weight verification**:
   - Double-check weight transpose is correct for MLX Conv2d
   - Verify weights are being applied in the correct order

## Debugger Usage

### MLX Debugger
```bash
mflux-debug-mlx start src/mflux_debugger/_scripts/debug_mflux_txt2img.py
mflux-debug-mlx continue  # Continue to checkpoints
mflux-debug-mlx checkpoint-break mlx_resample_after_repeat  # Set breakpoint
```

### PyTorch Debugger
```bash
mflux-debug-pytorch start src/mflux_debugger/_scripts/debug_diffusers_txt2img.py
mflux-debug-pytorch continue  # Continue to checkpoints
mflux-debug-pytorch checkpoint-break pytorch_resample_after_upsample  # Set breakpoint
```

## Saved Tensors Location
- **Latest**: `mflux_debugger/tensors/latest/`
- **Archived**: `mflux_debugger_deleted/mflux_debugger_YYYYMMDD_HHMMSS/tensors/latest/`

## Important Notes

1. **Temporary PyTorch modifications**: `pipeline_bria_fibo.py` has commented-out denoising loop - DO NOT COMMIT
2. **Channel count fix**: `wan_up_block.py` fix is committed and correct
3. **Debug checkpoints**: Many checkpoints added for `block_idx == 0` - can set `skip=True` on early ones once verified

## Current Hypothesis

The divergence is likely due to:
1. **Convolution computation difference**: MLX Conv2d might compute differently than PyTorch Conv2d
2. **Weight application order**: The way weights are applied in channels-last format might differ
3. **Padding behavior**: MLX and PyTorch might handle padding differently

The weights are correct, so the issue is in the computation, not the data.

