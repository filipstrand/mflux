"""
Chroma Text-to-Image Model.

Main model class for Chroma1-HD, a modified FLUX.1-schnell model with:
- DistilledGuidanceLayer (pre-computed modulations)
- T5-only text encoding (no CLIP)
- Support for negative prompts
- Recommended 40 inference steps
"""

from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.chroma.chroma_initializer import ChromaInitializer
from mflux.models.chroma.weights.chroma_weight_definition import ChromaWeightDefinition
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.flux.model.flux_transformer.transformer import Transformer
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Chroma(nn.Module):
    """
    Chroma Text-to-Image Model.

    A modified FLUX.1-schnell with DistilledGuidanceLayer for efficient
    inference with pre-computed modulations. Uses T5-only text encoding.

    Attributes:
        vae: VAE encoder/decoder
        transformer: ChromaTransformer with DistilledGuidanceLayer
        t5_text_encoder: T5 text encoder (no CLIP)
    """

    vae: VAE
    transformer: Transformer  # Actually ChromaTransformer, but same base type
    t5_text_encoder: T5Encoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = None,
    ):
        """
        Initialize Chroma model.

        Args:
            quantize: Quantization bit width (4 or 8) or None for full precision
            model_path: Path to model weights (local or HuggingFace repo ID)
            model_config: Model configuration (defaults to Chroma config)
        """
        super().__init__()
        if model_config is None:
            model_config = ModelConfig.chroma()

        ChromaInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 40,  # Chroma recommends 40 steps
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
        negative_prompt: str | None = None,  # Chroma supports negative prompts
    ) -> GeneratedImage:
        """
        Generate an image from a text prompt.

        Args:
            seed: Random seed for reproducibility
            prompt: Text description of the image to generate
            num_inference_steps: Number of denoising steps (default: 40)
            height: Image height in pixels (default: 1024)
            width: Image width in pixels (default: 1024)
            guidance: Guidance scale (default: 4.0)
            image_path: Optional input image for img2img
            image_strength: Strength for img2img (0.0-1.0)
            scheduler: Scheduler type ("linear" or "cosine")
            negative_prompt: Optional negative prompt (not yet implemented)

        Returns:
            GeneratedImage object containing the generated image and metadata
        """
        # 0. Create a new config based on the model type and input parameters
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
                latent_creator=FluxLatentCreator,
                image_path=config.image_path,
                sigmas=config.scheduler.sigmas,
                init_time_step=config.init_time_step,
            ),
        )

        # 2. Encode the prompt (T5 only, no CLIP)
        prompt_embeds = self._encode_prompt(prompt)

        # TODO: Implement negative prompt support
        # if negative_prompt:
        #     negative_prompt_embeds = self._encode_prompt(negative_prompt)

        # 3. Create callback context and call before_loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)

        for t in config.time_steps:
            try:
                # Scale model input if needed by the scheduler
                latents = config.scheduler.scale_model_input(latents, t)

                # 4.t Predict the noise
                # Note: Chroma doesn't need pooled_prompt_embeds (no CLIP)
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
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
        latents = FluxLatentCreator.unpack_latents(latents=latents, height=config.height, width=config.width)
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

    def _encode_prompt(self, prompt: str) -> mx.array:
        """
        Encode a text prompt using T5.

        Args:
            prompt: Text to encode

        Returns:
            T5 embeddings [batch, seq_len, 4096]
        """
        # Check cache first
        if prompt in self.prompt_cache:
            return self.prompt_cache[prompt]

        # Tokenize and encode with T5
        t5_tokenizer = self.tokenizers["t5"]
        t5_output = t5_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder(t5_output.input_ids)

        # Cache and return
        self.prompt_cache[prompt] = prompt_embeds
        return prompt_embeds

    @staticmethod
    def from_name(model_name: str, quantize: int | None = None) -> "Chroma":
        """
        Create a Chroma model from a model name.

        Args:
            model_name: Model name or alias (e.g., "chroma", "chroma-hd")
            quantize: Quantization bit width (4 or 8) or None

        Returns:
            Initialized Chroma model
        """
        return Chroma(
            model_config=ModelConfig.from_name(model_name=model_name, base_model=None),
            quantize=quantize,
        )

    def save_model(self, base_path: str) -> None:
        """
        Save the model to disk.

        Args:
            base_path: Directory path to save the model
        """
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=ChromaWeightDefinition,
        )

    def freeze(self, **kwargs):
        """Freeze all model parameters."""
        self.vae.freeze()
        self.transformer.freeze()
        self.t5_text_encoder.freeze()
