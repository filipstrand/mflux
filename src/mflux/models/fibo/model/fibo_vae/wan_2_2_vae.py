import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.fibo.model.fibo_vae.common.wan_2_2_causal_conv_3d import Wan2_2_CausalConv3d
from mflux.models.fibo.model.fibo_vae.decoder.wan_2_2_decoder_3d import Wan2_2_Decoder3d
from mflux.models.fibo.model.fibo_vae.encoder.wan_2_2_encoder_3d import Wan2_2_Encoder3d


class Wan2_2_VAE(nn.Module):
    Z_DIM = 48
    ENCODER_BASE_DIM = 160
    DECODER_BASE_DIM = 256
    DIM_MULT = [1, 2, 4, 4]
    NUM_RES_BLOCKS = 2
    OUT_CHANNELS = 12
    spatial_scale = 16
    latent_channels = 48

    LATENTS_MEAN = np.array([-0.2289, -0.0052, -0.1323, -0.2339, -0.2799, 0.0174, 0.1838, 0.1557, -0.1382, 0.0542, 0.2813, 0.0891, 0.157, -0.0098, 0.0375, -0.1825, -0.2246, -0.1207, -0.0698, 0.5109, 0.2665, -0.2108, -0.2158, 0.2502, -0.2055, -0.0322, 0.1109, 0.1567, -0.0729, 0.0899, -0.2799, -0.123, -0.0313, -0.1649, 0.0117, 0.0723, -0.2839, -0.2083, -0.052, 0.3748, 0.0152, 0.1957, 0.1433, -0.2944, 0.3573, -0.0548, -0.1681, -0.0667], dtype=np.float32)  # fmt: off
    LATENTS_STD = np.array([0.4765, 1.0364, 0.4514, 1.1677, 0.5313, 0.499, 0.4818, 0.5013, 0.8158, 1.0344, 0.5894, 1.0901, 0.6885, 0.6165, 0.8454, 0.4978, 0.5759, 0.3523, 0.7135, 0.6804, 0.5833, 1.4146, 0.8986, 0.5659, 0.7069, 0.5338, 0.4889, 0.4917, 0.4069, 0.4999, 0.6866, 0.4093, 0.5709, 0.6065, 0.6415, 0.4944, 0.5726, 1.2042, 0.5458, 1.6887, 0.3971, 1.06, 0.3943, 0.5537, 0.5444, 0.4089, 0.7468, 0.7744], dtype=np.float32)  # fmt: off

    def __init__(self):
        super().__init__()

        self.encoder = Wan2_2_Encoder3d(
            in_channels=12,
            dim=self.ENCODER_BASE_DIM,
            z_dim=self.Z_DIM * 2,
            dim_mult=self.DIM_MULT,
            num_res_blocks=self.NUM_RES_BLOCKS,
            attn_scales=[],
            temporal_downsample=[False, True, True],
        )
        self.quant_conv = Wan2_2_CausalConv3d(self.Z_DIM * 2, self.Z_DIM * 2, 1, padding=0, name="quant_conv")

        self.decoder = Wan2_2_Decoder3d(
            dim=self.DECODER_BASE_DIM,
            z_dim=self.Z_DIM,
            dim_mult=self.DIM_MULT,
            num_res_blocks=self.NUM_RES_BLOCKS,
            temporal_upsample=None,
            out_channels=self.OUT_CHANNELS,
        )
        self.post_quant_conv = Wan2_2_CausalConv3d(self.Z_DIM, self.Z_DIM, 1, padding=0, name="post_quant_conv")

    def encode(self, images: mx.array) -> mx.array:
        if images.ndim == 4:
            x = images.reshape(images.shape[0], images.shape[1], 1, images.shape[2], images.shape[3])
        elif images.ndim == 5:
            x = images
        else:
            raise ValueError(f"Expected 4D or 5D input for VAE.encode, got shape {images.shape}")

        patch_size = 2
        x = self._patchify(x, patch_size=patch_size)
        h = self.encoder(x)
        h = self.quant_conv(h)
        mean = h[:, : self.Z_DIM, :, :, :]
        latents_mean = mx.array(self.LATENTS_MEAN).reshape(1, self.Z_DIM, 1, 1, 1)
        latents_std = mx.array(self.LATENTS_STD).reshape(1, self.Z_DIM, 1, 1, 1)
        encoded = (mean - latents_mean) / latents_std
        return encoded

    def decode(self, latents: mx.array) -> mx.array:
        if latents.ndim == 4:
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])

        latents_mean = mx.array(self.LATENTS_MEAN).reshape(1, self.Z_DIM, 1, 1, 1)
        latents_std = mx.array(self.LATENTS_STD).reshape(1, self.Z_DIM, 1, 1, 1)
        latents = latents * latents_std + latents_mean
        latents = self.post_quant_conv(latents)
        decoded = self.decoder(latents)
        patch_size = 2
        decoded = self._unpatchify(decoded, patch_size=patch_size)
        return decoded

    @staticmethod
    def _patchify(x: mx.array, patch_size: int) -> mx.array:
        if patch_size == 1:
            return x
        batch_size, channels, frames, height, width = x.shape
        x = mx.reshape(
            x,
            (
                batch_size,
                channels,
                frames,
                height // patch_size,
                patch_size,
                width // patch_size,
                patch_size,
            ),
        )
        x = mx.transpose(x, (0, 1, 6, 4, 2, 3, 5))
        x = mx.reshape(
            x,
            (batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size),
        )
        return x

    @staticmethod
    def _unpatchify(x: mx.array, patch_size: int) -> mx.array:
        if patch_size == 1:
            return x
        batch_size, c_patches, frames, height, width = x.shape
        channels = c_patches // (patch_size * patch_size)
        x = mx.reshape(x, (batch_size, channels, patch_size, patch_size, frames, height, width))
        x = mx.transpose(x, (0, 1, 4, 5, 3, 6, 2))
        x = mx.reshape(x, (batch_size, channels, frames, height * patch_size, width * patch_size))
        return x
