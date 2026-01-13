"""FLUX.2 text-to-image generation model.

FLUX.2 is a 32B parameter flow-matching transformer model with:
- 8 joint transformer blocks + 48 single transformer blocks
- Mistral3 multimodal text encoder
- 32-channel VAE
- Global modulation layers
"""

from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder import Flux2PromptEncoder, Mistral3TextEncoder
from mflux.models.flux2.model.flux2_transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae import Flux2VAE
from mflux.models.flux2.weights import Flux2WeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux2(nn.Module):
    """FLUX.2 text-to-image generation model.

    Attributes:
        vae: FLUX.2 32-channel VAE for encoding/decoding images
        transformer: FLUX.2 transformer (8 joint + 48 single blocks)
        text_encoder: Mistral3 text encoder
    """

    vae: Flux2VAE
    transformer: Flux2Transformer
    text_encoder: Mistral3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ):
        """Initialize FLUX.2 model.

        Args:
            quantize: Quantization bits (4 or 8) or None for full precision
            model_path: Optional custom path to model weights
            lora_paths: Optional list of LoRA adapter paths
            lora_scales: Optional list of LoRA scales (one per path)
            model_config: Optional model configuration (defaults to FLUX.2 config)
        """
        super().__init__()

        # Use default FLUX.2 config if not provided
        if model_config is None:
            model_config = ModelConfig.flux2()

        Flux2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 3.5,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        """Generate an image from a text prompt.

        Args:
            seed: Random seed for reproducibility
            prompt: Text description of the image to generate
            num_inference_steps: Number of denoising steps (default 30)
            height: Output image height in pixels (default 1024)
            width: Output image width in pixels (default 1024)
            guidance: Guidance scale (default 3.5)
            image_path: Optional input image for img2img
            image_strength: Strength for img2img (0.0-1.0)
            scheduler: Noise scheduler type (default "linear")
            negative_prompt: Optional negative prompt (not used in FLUX.2)

        Returns:
            GeneratedImage containing the generated image and metadata
        """
        # 0. Create config based on input parameters
        config = Config(
            width=width,
            height=height,
            guidance=guidance,
            scheduler=scheduler,
            image_path=image_path,
            image_strength=image_strength,
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
        )

        # 1. Create the initial latents
        latents = LatentCreator.create_for_txt2img_or_img2img(
            seed=seed,
            width=config.width,
            height=config.height,
            img2img=Img2Img(
                vae=self.vae,
                latent_creator=Flux2LatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
            ),
        )

        # 2. Encode the prompt using Mistral3
        prompt_embeds, pooled_prompt_embeds = Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            prompt_cache=self.prompt_cache,
            mistral3_tokenizer=self.tokenizers["mistral3"],
            text_encoder=self.text_encoder,
        )

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 4.t Predict the noise
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )

                # 5.t Take one denoise step
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)

                # 6.t Call subscribers in-loop
                ctx.in_loop(t, latents)

                # (Optional) Evaluate to enable progress tracking
                mx.eval(latents)

            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # 7. Call subscribers after loop
        ctx.after_loop(latents)

        # 8. Decode the latent array and return the image
        latents = Flux2LatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = VAEUtil.decode(vae=self.vae, latent=latents, tiling_config=self.tiling_config)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_path=config.image_path,
            image_strength=config.image_strength,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    @staticmethod
    def from_name(model_name: str, quantize: int | None = None) -> "Flux2":
        """Create a FLUX.2 model from a model name.

        Args:
            model_name: Name or alias of the model
            quantize: Quantization bits (4 or 8) or None

        Returns:
            Initialized Flux2 model
        """
        return Flux2(
            model_config=ModelConfig.from_name(model_name=model_name, base_model=None),
            quantize=quantize,
        )

    def save_model(self, base_path: str) -> None:
        """Save the model weights to disk.

        Args:
            base_path: Base directory for saving weights
        """
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Flux2WeightDefinition,
        )

    def freeze(self, **kwargs):
        """Freeze model parameters for inference."""
        self.vae.freeze()
        self.transformer.freeze()
        self.text_encoder.freeze()
