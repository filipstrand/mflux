"""Wan Resample module for FIBO VAE decoder.

Simplified version based on QwenImageResample3D.
"""

import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_causal_conv_3d import WanCausalConv3d


class WanResample(nn.Module):
    """Resampling module for 2D and 3D upsampling/downsampling.

    Simplified version - supports upsample2d and upsample3d for decoder.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None):
        """Initialize resample module.

        Args:
            dim: Number of channels
            mode: Resampling mode ('upsample2d', 'upsample3d', or None)
            upsample_out_dim: Output channels after upsampling (defaults to dim // 2)
        """
        super().__init__()
        self.dim = dim
        self.mode = mode

        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        if mode == "upsample3d":
            # Temporal upsampling: duplicate frames, then spatial upsample
            self.time_conv = WanCausalConv3d(dim, dim * 2, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
            self.resample_conv = nn.Conv2d(dim, upsample_out_dim, kernel_size=3, stride=1, padding=1)
        elif mode == "upsample2d":
            # Spatial upsampling only
            self.resample_conv = nn.Conv2d(dim, upsample_out_dim, kernel_size=3, stride=1, padding=1)
            self.time_conv = None
        else:
            raise ValueError(f"Unsupported resample mode: {mode}")

    def __call__(self, x: mx.array, block_idx: int | None = None) -> mx.array:
        """Apply resampling.

        Args:
            x: Input tensor of shape (batch, channels, time, height, width)
            block_idx: Optional block index for debugging

        Returns:
            Resampled tensor
        """
        # DEBUG: Always checkpoint at entry to verify resample is called
        from mflux_debugger.semantic_checkpoint import debug_checkpoint

        debug_checkpoint(
            "mlx_resample_entry",
            x=x,
            metadata={"block_idx": block_idx, "input_shape": list(x.shape), "mode": self.mode},
            skip=True,  # Log but don't pause - just verify it's called
        )
        # Debug: input (skip - we know input matches)
        # if block_idx == 0:
        #     from mflux_debugger.tensor_debug import debug_save
        #     debug_save(x, "mlx_resample_input")

        b, c, t, h, w = x.shape

        # Handle temporal upsampling for 3D mode
        if self.mode == "upsample3d" and self.time_conv is not None:
            # Apply temporal convolution to double temporal dimension
            x = self.time_conv(x)
            # Reshape: (b, c*2, t, h, w) -> (b, 2, c, t, h, w) -> interleave -> (b, c, t*2, h, w)
            x = mx.reshape(x, (b, 2, c, t, h, w))
            x = mx.transpose(x, (0, 2, 3, 1, 4, 5))  # (b, c, t, 2, h, w)
            x = mx.reshape(x, (b, c, t * 2, h, w))
            t = t * 2

        # Reshape for 2D convolution: (b, c, t, h, w) -> (b*t, c, h, w)
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # (b, t, c, h, w)
        x = mx.reshape(x, (b * t, c, h, w))
        x = mx.transpose(x, (0, 2, 3, 1))  # (b*t, h, w, c) for Conv2d

        # Debug: before upsampling (after reshape) - skip, we know this matches
        # if block_idx == 0:
        #     from mflux_debugger.tensor_debug import debug_save
        #     debug_save(x, "mlx_resample_before_upsample_conv")

        # Spatial upsampling: nearest neighbor 2x
        x = mx.repeat(x, 2, axis=1)  # Repeat height
        x = mx.repeat(x, 2, axis=2)  # Repeat width

        # Debug: after repeat (before conv) - skip, we'll check at before_conv2d_compute
        # if block_idx == 0:
        #     from mflux_debugger.semantic_checkpoint import debug_checkpoint
        #     from mflux_debugger.tensor_debug import debug_save
        #     debug_save(x, "mlx_resample_after_repeat")
        #     # Debug: inspect conv2d weights
        #     if hasattr(self.resample_conv, 'weight'):
        #         weight_shape = self.resample_conv.weight.shape
        #         weight_mean = float(mx.mean(self.resample_conv.weight))
        #         weight_min = float(mx.min(self.resample_conv.weight))
        #         weight_max = float(mx.max(self.resample_conv.weight))
        #         debug_checkpoint(
        #             "mlx_resample_conv2d_weights",
        #             metadata={
        #                 "weight_shape": list(weight_shape),
        #                 "weight_mean": weight_mean,
        #                 "weight_min": weight_min,
        #                 "weight_max": weight_max,
        #                 "in_channels": self.dim,
        #                 "out_channels": self.resample_conv.out_channels if hasattr(self.resample_conv, 'out_channels') else 'unknown',
        #             },
        #         )

        # Apply 2D convolution
        # MLX Conv2d expects channels-last input (B, H, W, C), which we already have
        # DEBUG: Inspect before conv2d - THIS IS WHERE WE NEED TO BE FOR VERIFICATION
        # CRITICAL: This checkpoint MUST pause when block_idx == 0
        if block_idx == 0:
            import sys

            print(f"DEBUG: About to call checkpoint, block_idx={block_idx}", file=sys.stderr, flush=True)
            from mflux_debugger.semantic_checkpoint import debug_checkpoint

            print("DEBUG: Calling debug_checkpoint now", file=sys.stderr, flush=True)
            debug_checkpoint(
                "mlx_resample_before_conv2d_compute",
                x=x,  # Pass tensor directly for inspection
                metadata={"block_idx": block_idx, "input_shape": list(x.shape)},
                skip=False,  # EXPLICIT: Must pause here!
            )
            print("DEBUG: After debug_checkpoint call", file=sys.stderr, flush=True)
        x = self.resample_conv(x)  # (b*t, h*2, w*2, out_c)
        # DEBUG: Inspect after conv2d - CRITICAL CHECKPOINT for comparison
        if block_idx == 0:
            from mflux_debugger.semantic_checkpoint import debug_checkpoint

            debug_checkpoint(
                "mlx_resample_after_conv2d_compute",
                metadata={
                    "output_shape": list(x.shape),
                    "output_mean": float(mx.mean(x)),
                    "output_std": float(mx.std(x)),
                    "output_at_20_42_13": float(x[0, 20, 42, 13]),
                },
            )

        # Debug: after conv2d - keep this for final comparison
        if block_idx == 0:
            from mflux_debugger.tensor_debug import debug_save

            debug_save(x, "mlx_resample_after_conv2d")

        x = mx.transpose(x, (0, 3, 1, 2))  # (b*t, out_c, h*2, w*2)

        # Reshape back: (b*t, out_c, h*2, w*2) -> (b, out_c, t, h*2, w*2)
        new_c = x.shape[1]
        new_h, new_w = x.shape[2], x.shape[3]
        x = mx.reshape(x, (b, t, new_c, new_h, new_w))
        x = mx.transpose(x, (0, 2, 1, 3, 4))  # (b, out_c, t, new_h, new_w)

        # Debug: output (skip - same as after_conv2d, just reshaped)
        # if block_idx == 0:
        #     from mflux_debugger.tensor_debug import debug_save
        #     debug_save(x, "mlx_resample_output")

        return x
