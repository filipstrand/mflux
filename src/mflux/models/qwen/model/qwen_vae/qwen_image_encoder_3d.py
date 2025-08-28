import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen.model.qwen_vae.qwen_image_down_block_3d import QwenImageDownBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_mid_block_3d import QwenImageMidBlock3D
from mflux.models.qwen.model.qwen_vae.qwen_image_rms_norm import QwenImageRMSNorm


class QwenImageEncoder3D(nn.Module):

    def __init__(self, dim=96, z_dim=32, dim_mult=[1, 2, 4, 4], num_res_blocks=2,
                 attn_scales=[], temporal_downsample=[False, True, True], dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temporal_downsample = temporal_downsample
        self.dropout = dropout

        dims = [dim * u for u in [1] + dim_mult]
        self.conv_in = QwenImageCausalConv3D(3, dims[0], 3, 1, 1)

        down_blocks = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i < len(self.temporal_downsample):
                downsample_mode = "downsample3d" if self.temporal_downsample[i] else "downsample2d"
            else:
                downsample_mode = "downsample2d"
            if i == 0 and in_dim == out_dim:
                downsample_mode = None
            down_block = QwenImageDownBlock3D(
                in_dim, out_dim,
                num_res_blocks=num_res_blocks,
                downsample_mode=downsample_mode
            )
            down_blocks.append(down_block)
        self.down_blocks = down_blocks

        self.mid_block = QwenImageMidBlock3D(dims[-1], num_layers=1)
        self.norm_out = QwenImageRMSNorm(dims[-1])
        self.conv_out = QwenImageCausalConv3D(dims[-1], 32, 3, 1, 1) # Changed z_dim to 32

    def __call__(self, x: mx.array) -> mx.array:
        import numpy as np
        from pathlib import Path
        
        # Create debug directory
        debug_dir = Path("debug_tensors_mlx")
        debug_dir.mkdir(exist_ok=True)
        
        # Save encoder input
        np.save(debug_dir / "encoder_00_input.npy", np.array(x))
        
        # CRITICAL FIX: Apply conv_in first!
        x = self.conv_in(x)         # 3 -> 96 channels
        np.save(debug_dir / "encoder_01_conv_in.npy", np.array(x))
        
        # Process through down blocks
        for i, down_block in enumerate(self.down_blocks):
            x = down_block(x)
            np.save(debug_dir / f"encoder_02_down_block_{i}.npy", np.array(x))
        
        # Final processing
        x = self.mid_block(x)       # Process through mid block
        np.save(debug_dir / "encoder_03_mid_block.npy", np.array(x))
        
        x = self.norm_out(x)        # Normalize
        np.save(debug_dir / "encoder_04_norm_out.npy", np.array(x))
        
        x = nn.silu(x)              # Activation
        np.save(debug_dir / "encoder_05_silu.npy", np.array(x))
        
        x = self.conv_out(x)        # Final convolution to 32 channels
        np.save(debug_dir / "encoder_06_conv_out.npy", np.array(x))
        
        return x