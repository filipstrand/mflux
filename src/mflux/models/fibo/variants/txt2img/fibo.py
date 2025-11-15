import mlx.core as mx
from mlx import nn

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.fibo.fibo_initializer import FIBOInitializer

# from mflux.models.fibo.model.fibo_transformer import FiboTransformer  # Commented out - not needed when loading final latents directly
from mflux.models.fibo.model.fibo_vae.vae import VAE
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux_debugger.tensor_debug import debug_load


class FIBO(nn.Module):
    vae: VAE
    # transformer: FiboTransformer  # Commented out - not needed when loading final latents directly

    def __init__(
        self,
        quantize: int | None = None,
        local_path: str | None = None,
    ):
        super().__init__()
        self.bits = quantize
        self.local_path = local_path

        # Load weights and initialize model components
        # NOTE: For testing, we only initialize VAE (transformer not needed)
        FIBOInitializer.init(
            fibo_model=self,
            quantize=quantize,
            local_path=local_path,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        config: RuntimeConfig,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        """
        Minimal FIBO txt2img pipeline in MLX.

        NOTE: For now we rely on PyTorch to provide the text-conditioning tensors
        via debug_save() (pt_encoder_hidden_states, pt_prompt_layers, etc.).
        The denoising loop and VAE decoding run fully in MLX.
        """
        # ---------------------------------------------------------------------
        # 1. Load final latents directly from PyTorch
        # ---------------------------------------------------------------------
        # For testing reshape/unpack logic, we skip:
        # - Text encoder loading (saves time)
        # - Transformer/denoising loop (saves time)
        # - Only load final latents and run VAE decode
        latents = debug_load("fibo_final_latents")

        # Get dimensions from the loaded latents
        batch_size = latents.shape[0]
        channels = latents.shape[2]
        height = config.height
        width = config.width
        vae_scale_factor = 16
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        latents_unpacked = mx.reshape(latents, (batch_size, latent_height, latent_width, channels))
        latents_unpacked = mx.transpose(latents_unpacked, (0, 3, 1, 2))  # (B, C, H', W')
        latents_unpacked = mx.expand_dims(latents_unpacked, axis=2)
        latents_mean = mx.array(self.vae.LATENTS_MEAN).reshape(1, self.vae.Z_DIM, 1, 1, 1)
        latents_std = mx.array(self.vae.LATENTS_STD).reshape(1, self.vae.Z_DIM, 1, 1, 1)
        latents_scaled = latents_unpacked * latents_std + latents_mean
        latents_for_vae = latents_scaled
        decoded = self.vae.decode(latents_for_vae)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=None,
            lora_scales=None,
            image_path=None,
            image_strength=None,
            generation_time=0.0,
        )
