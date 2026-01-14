"""
LongCat Text-to-Image Model.

Main model class for LongCat-Image, a 6B parameter flow match model with:
- Qwen2.5-VL text encoder (3584 hidden, 28 layers)
- Flow Match transformer (10 joint + 20 single blocks)
- Standard 16-channel VAE
- Character-level text encoding support
"""

from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import Img2Img, LatentCreator
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
from mflux.models.flux.model.flux_vae.vae import VAE
from mflux.models.longcat.longcat_initializer import LongCatInitializer
from mflux.models.longcat.model.longcat_text_encoder.longcat_text_encoder import LongCatTextEncoder
from mflux.models.longcat.model.longcat_transformer.longcat_transformer import LongCatTransformer
from mflux.models.longcat.weights.longcat_weight_definition import LongCatWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class LongCat(nn.Module):
    """
    LongCat Text-to-Image Model.

    A 6B parameter flow match model using Qwen2.5-VL for text encoding.
    Supports character-level text rendering (text in quotes).

    Architecture:
    - Text Encoder: Qwen2.5-VL (3584 hidden, 28 layers)
    - Transformer: 10 joint + 20 single blocks (3072 hidden)
    - VAE: Standard 16-channel (same as FLUX)

    Attributes:
        vae: VAE encoder/decoder
        transformer: LongCatTransformer
        text_encoder: Qwen2.5-VL text encoder
    """

    vae: VAE
    transformer: LongCatTransformer
    text_encoder: LongCatTextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        """
        Initialize LongCat model.

        Args:
            quantize: Quantization bit width (4 or 8) or None for full precision
            model_path: Path to model weights (local or HuggingFace repo ID)
            model_config: Model configuration (defaults to LongCat config)
            lora_paths: List of LoRA paths (local files or HuggingFace repos)
            lora_scales: List of LoRA scale factors (default 1.0)
        """
        super().__init__()
        if model_config is None:
            model_config = ModelConfig.longcat()

        LongCatInitializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 30,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 4.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "linear",
    ) -> GeneratedImage:
        """
        Generate an image from a text prompt.

        Args:
            seed: Random seed for reproducibility
            prompt: Text description of the image to generate
            num_inference_steps: Number of denoising steps (default: 30)
            height: Image height in pixels (default: 1024)
            width: Image width in pixels (default: 1024)
            guidance: Guidance scale (default: 4.0)
            image_path: Optional input image for img2img
            image_strength: Strength for img2img (0.0-1.0)
            scheduler: Scheduler type ("linear" or "cosine")

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

        # 2. Encode the prompt
        prompt_embeds, pooled_prompt_embeds = self._encode_prompt(prompt)

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

    def _encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        """
        Encode a text prompt using Qwen2.5-VL.

        Args:
            prompt: Text to encode

        Returns:
            Tuple of (prompt_embeds, pooled_prompt_embeds)
            - prompt_embeds: Sequence embeddings [batch, seq_len, 3584]
            - pooled_prompt_embeds: Pooled embeddings [batch, 3584]
        """
        # Check cache first
        if prompt in self.prompt_cache:
            return self.prompt_cache[prompt]

        # Tokenize with Qwen tokenizer
        qwen_tokenizer = self.tokenizers["qwen"]
        tokenized = qwen_tokenizer.tokenize(prompt)

        # Encode with Qwen2.5-VL text encoder
        prompt_embeds, pooled_prompt_embeds = self.text_encoder(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
        )

        # Cache and return
        result = (prompt_embeds, pooled_prompt_embeds)
        self.prompt_cache[prompt] = result
        return result

    @staticmethod
    def from_name(model_name: str, quantize: int | None = None) -> "LongCat":
        """
        Create a LongCat model from a model name.

        Args:
            model_name: Model name or alias (e.g., "longcat", "longcat-image")
            quantize: Quantization bit width (4 or 8) or None

        Returns:
            Initialized LongCat model
        """
        return LongCat(
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
            weight_definition=LongCatWeightDefinition,
        )

    def freeze(self, **kwargs):
        """Freeze all model parameters."""
        self.vae.freeze()
        self.transformer.freeze()
        self.text_encoder.freeze()
