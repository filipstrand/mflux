import mlx.core as mx

from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE


class Flux3LatentCreator:
    DOWNSAMPLING_RATIO = 16
    IMAGE_ROPE_SCALE = 0.5
    LATENT_CHANNELS = 128

    @staticmethod
    def latent_shape(height: int, width: int) -> tuple[int, int]:
        ratio = Flux3LatentCreator.DOWNSAMPLING_RATIO
        if height % ratio or width % ratio:
            raise ValueError(f"width and height must be multiples of {ratio}")
        return height // ratio, width // ratio

    @staticmethod
    def create_noise(seed: int, height: int, width: int) -> mx.array:
        h, w = Flux3LatentCreator.latent_shape(height, width)
        noise = mx.random.normal(shape=(1, h * w, Flux3LatentCreator.LATENT_CHANNELS), key=mx.random.key(seed))
        # The reference rounds the noise to bfloat16 and keeps the Euler state in float32.
        return noise.astype(mx.bfloat16).astype(mx.float32)

    @staticmethod
    def prepare_ids(h: int, w: int, layer: int = 0) -> mx.array:
        # (t, row, col, l) per token; rows and columns are scaled by the image RoPE scale.
        scale = Flux3LatentCreator.IMAGE_ROPE_SCALE
        rows = mx.broadcast_to(mx.arange(h, dtype=mx.float32)[:, None], (h, w)).reshape(-1) * scale
        cols = mx.broadcast_to(mx.arange(w, dtype=mx.float32)[None, :], (h, w)).reshape(-1) * scale
        zeros = mx.zeros_like(rows)
        return mx.stack([zeros, rows, cols, zeros + layer], axis=1)

    @staticmethod
    def pack_latents(latents: mx.array) -> mx.array:
        return Flux2LatentCreator.pack_latents(latents)

    @staticmethod
    def unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        # (B, h*w, C) tokens -> (B, C, h, w); height and width are pixels (callback convention).
        if latents.ndim == 4:
            return latents
        h, w = Flux3LatentCreator.latent_shape(height, width)
        batch, seq_len, channels = latents.shape
        if seq_len != h * w:
            raise ValueError(f"Token count {seq_len} does not match a {h}x{w} latent grid")
        return latents.reshape(batch, h, w, channels).transpose(0, 3, 1, 2)

    @staticmethod
    def encode_tokens(vae: Flux2VAE, image: mx.array, tiling_config: TilingConfig | None = None) -> mx.array:
        # image (B, 3, H, W) in [-1, 1] -> normalized tokens (B, h*w, 128); channel index is c*4 + dy*2 + dx
        # VAEUtil.encode squeezes the 5D (B, C, 1, H, W) output of the tiled path.
        latents = Flux2LatentCreator.patchify_latents(VAEUtil.encode(vae=vae, image=image, tiling_config=tiling_config))
        bn_mean = vae.bn.running_mean.reshape(1, -1, 1, 1).astype(latents.dtype)
        bn_std = mx.sqrt(vae.bn.running_var.reshape(1, -1, 1, 1) + vae.bn.eps).astype(latents.dtype)
        return Flux3LatentCreator.pack_latents((latents - bn_mean) / bn_std)

    @staticmethod
    def decode_tokens(
        vae: Flux2VAE, tokens: mx.array, h: int, w: int, tiling_config: TilingConfig | None = None
    ) -> mx.array:
        unpacked = Flux3LatentCreator.unpack_latents(tokens, h * 16, w * 16)
        return vae.decode_packed_latents(unpacked, tiling_config=tiling_config)
