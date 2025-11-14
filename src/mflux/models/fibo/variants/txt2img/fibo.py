from mlx import nn

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.fibo.fibo_initializer import FIBOInitializer
from mflux.models.fibo.model.fibo_vae.vae import VAE
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux_debugger.tensor_debug import debug_load


class FIBO(nn.Module):
    vae: VAE

    def __init__(
        self,
        quantize: int | None = None,
        local_path: str | None = None,
    ):
        super().__init__()
        self.bits = quantize
        self.local_path = local_path

        # Load weights and initialize model components
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
        # TODO: Text encoding and diffusion loop

        latents_for_vae = debug_load("vae_input_latents")
        if latents_for_vae.ndim == 4:
            latents_for_vae = latents_for_vae.reshape(
                latents_for_vae.shape[0],
                latents_for_vae.shape[1],
                1,
                latents_for_vae.shape[2],
                latents_for_vae.shape[3],
            )

        # Decode using VAE with the exact tensor from diffusers
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
