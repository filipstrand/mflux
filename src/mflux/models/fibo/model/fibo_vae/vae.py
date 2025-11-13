"""FIBO VAE implementation.

The FIBO model uses AutoencoderKLWan (WanDecoder3d) from diffusers.
This implements the decoder part for now.
"""

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.fibo.model.fibo_vae.decoder.wan_causal_conv_3d import WanCausalConv3d
from mflux.models.fibo.model.fibo_vae.decoder.wan_decoder_3d import WanDecoder3d
from mflux_debugger.semantic_checkpoint import debug_checkpoint
from mflux_debugger.tensor_debug import debug_save


class VAE(nn.Module):
    """FIBO VAE - decoder implementation.

    For now, we only implement the decoder part.
    The decoder uses WanDecoder3d structure (3D causal convolutions).

    Config values from FIBO model (captured from debug checkpoint):
    - z_dim: 48 (latent channels)
    - decoder_base_dim: 256
    - dim_mult: [1, 2, 4, 4]
    - num_res_blocks: 2
    - temporal_upsample: None (no temporal upsampling)
    - out_channels: 12
    - latents_mean: array of 48 values
    - latents_std: array of 48 values
    """

    # Config values from FIBO model (captured from debug checkpoint)
    Z_DIM = 48
    DECODER_BASE_DIM = 256
    DIM_MULT = [1, 2, 4, 4]
    NUM_RES_BLOCKS = 2
    TEMPORAL_UPSAMPLE = None  # No temporal upsampling
    OUT_CHANNELS = 12
    DROPOUT = 0.0

    # Latents normalization values (48 channels)
    LATENTS_MEAN = np.array(
        [
            -0.2289,
            -0.0052,
            -0.1323,
            -0.2339,
            -0.2799,
            0.0174,
            0.1838,
            0.1557,
            -0.1382,
            0.0542,
            0.2813,
            0.0891,
            0.157,
            -0.0098,
            0.0375,
            -0.1825,
            -0.2246,
            -0.1207,
            -0.0698,
            0.5109,
            0.2665,
            -0.2108,
            -0.2158,
            0.2502,
            -0.2055,
            -0.0322,
            0.1109,
            0.1567,
            -0.0729,
            0.0899,
            -0.2799,
            -0.123,
            -0.0313,
            -0.1649,
            0.0117,
            0.0723,
            -0.2839,
            -0.2083,
            -0.052,
            0.3748,
            0.0152,
            0.1957,
            0.1433,
            -0.2944,
            0.3573,
            -0.0548,
            -0.1681,
            -0.0667,
        ],
        dtype=np.float32,
    )

    LATENTS_STD = np.array(
        [
            0.4765,
            1.0364,
            0.4514,
            1.1677,
            0.5313,
            0.499,
            0.4818,
            0.5013,
            0.8158,
            1.0344,
            0.5894,
            1.0901,
            0.6885,
            0.6165,
            0.8454,
            0.4978,
            0.5759,
            0.3523,
            0.7135,
            0.6804,
            0.5833,
            1.4146,
            0.8986,
            0.5659,
            0.7069,
            0.5338,
            0.4889,
            0.4917,
            0.4069,
            0.4999,
            0.6866,
            0.4093,
            0.5709,
            0.6065,
            0.6415,
            0.4944,
            0.5726,
            1.2042,
            0.5458,
            1.6887,
            0.3971,
            1.06,
            0.3943,
            0.5537,
            0.5444,
            0.4089,
            0.7468,
            0.7744,
        ],
        dtype=np.float32,
    )

    def __init__(self):
        super().__init__()
        # Initialize decoder with actual FIBO config values
        self.decoder = WanDecoder3d(
            dim=self.DECODER_BASE_DIM,
            z_dim=self.Z_DIM,
            dim_mult=self.DIM_MULT,
            num_res_blocks=self.NUM_RES_BLOCKS,
            temporal_upsample=self.TEMPORAL_UPSAMPLE,  # None = no temporal upsampling
            dropout=self.DROPOUT,
            out_channels=self.OUT_CHANNELS,
        )
        # Post-quantization convolution (applied before decoder)
        self.post_quant_conv = WanCausalConv3d(self.Z_DIM, self.Z_DIM, 1, padding=0, name="post_quant_conv")

    def decode(self, latents: mx.array) -> mx.array:
        """Decode latents to image.

        Args:
            latents: Latent tensor from the denoising process.
                    Shape should be (batch, channels, height, width) or (batch, channels, 1, height, width)
                    Expected shape: (batch, 48, height, width) or (batch, 48, 1, height, width)
                    Latents should already be scaled: (latent / latents_std) + latents_mean

        Returns:
            Decoded image tensor of shape (batch, 12, height, width)
            Note: FIBO outputs 12 channels, not 3 (likely for post-processing/decoding)
        """
        # Debug checkpoint: VAE decode input
        debug_checkpoint(
            "mlx_vae_decode_input",
            metadata={
                "shape": list(latents.shape),
                "dtype": str(latents.dtype),
                "min": float(latents.min()),
                "max": float(latents.max()),
                "mean": float(latents.mean()),
            },
            skip=True,  # Verified correct - skip to speed up debugging
        )
        debug_save(latents, "mlx_vae_decode_input")

        # Ensure latents are 5D: (batch, channels, 1, height, width)
        if latents.ndim == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])

        # Apply post-quantization convolution
        latents = self.post_quant_conv(latents)
        debug_checkpoint(
            "mlx_vae_after_post_quant_conv",
            metadata={"shape": list(latents.shape), "dtype": str(latents.dtype)},
            skip=True,  # Verified correct - matches PyTorch (max diff 0.003895)
        )
        debug_save(latents, "mlx_vae_after_post_quant_conv")

        # Decode
        decoded = self.decoder(latents)
        debug_checkpoint(
            "mlx_vae_after_decoder",
            metadata={"shape": list(decoded.shape), "dtype": str(decoded.dtype)},
            skip=True,  # After decoder - skip to focus on resample issue
        )
        debug_save(decoded, "mlx_vae_after_decoder")

        # Apply unpatchify if patch_size is set (FIBO uses patch_size=2)
        # unpatchify upscales spatial dimensions and reduces channels
        # Input: (batch, channels * patch_size^2, frames, height, width)
        # Output: (batch, channels, frames, height * patch_size, width * patch_size)
        patch_size = 2  # FIBO VAE uses patch_size=2
        decoded = self._unpatchify(decoded, patch_size=patch_size)

        debug_checkpoint(
            "mlx_vae_after_unpatchify",
            metadata={"shape": list(decoded.shape), "dtype": str(decoded.dtype), "patch_size": patch_size},
        )
        debug_save(decoded, "mlx_vae_after_unpatchify")

        # Remove temporal dimension: (batch, channels, 1, height, width) -> (batch, channels, height, width)
        if decoded.shape[2] == 1:
            decoded = decoded[:, :, 0, :, :]

        debug_checkpoint("mlx_vae_decode_output", metadata={"shape": list(decoded.shape), "dtype": str(decoded.dtype)})
        debug_save(decoded, "mlx_vae_decode_output")

        return decoded

    @staticmethod
    def _unpatchify(x: mx.array, patch_size: int) -> mx.array:
        """Unpatchify tensor - reverse of patchify.

        Args:
            x: Input tensor of shape (batch, channels * patch_size^2, frames, height, width)
            patch_size: Patch size (typically 2)

        Returns:
            Unpatchified tensor of shape (batch, channels, frames, height * patch_size, width * patch_size)
        """
        if patch_size == 1:
            return x

        if x.ndim != 5:
            raise ValueError(f"Invalid input shape: {x.shape}")

        batch_size, c_patches, frames, height, width = x.shape
        channels = c_patches // (patch_size * patch_size)

        if c_patches % (patch_size * patch_size) != 0:
            raise ValueError(
                f"Input channels ({c_patches}) must be divisible by patch_size^2 ({patch_size * patch_size})"
            )

        # Reshape to [b, c, patch_size, patch_size, f, h, w]
        x = mx.reshape(x, (batch_size, channels, patch_size, patch_size, frames, height, width))

        # Rearrange to [b, c, f, h * patch_size, w * patch_size]
        # Permute: (0, 1, 4, 5, 3, 6, 2) -> (b, c, f, h, patch_size, w, patch_size)
        # Then reshape to (b, c, f, h * patch_size, w * patch_size)
        x = mx.transpose(x, (0, 1, 4, 5, 3, 6, 2))  # (b, c, f, h, patch_size, w, patch_size)
        x = mx.reshape(x, (batch_size, channels, frames, height * patch_size, width * patch_size))

        return x
