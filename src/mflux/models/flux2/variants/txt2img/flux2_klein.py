from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux2Klein(nn.Module):
    vae: Flux2VAE
    transformer: Flux2Transformer
    text_encoder: Qwen3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        Flux2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config or ModelConfig.flux2_klein_4b(),
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 4,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 1.0,
        image_path: Path | str | None = None,
        image_strength: float | None = None,
        scheduler: str = "flow_match_euler_discrete",
    ) -> GeneratedImage:
        if guidance != 1.0:
            raise ValueError("FLUX.2 does not support negative prompts / CFG. Use guidance=1.0.")

        # 0. Create a new config based on the model type and input parameters
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=1.0,
            image_path=image_path,
            image_strength=image_strength,
            scheduler=scheduler,
        )
        # 1. Encode prompt
        prompt_embeds, text_ids = Flux2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["qwen3"],
            text_encoder=self.text_encoder,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )

        # 2. Prepare latents
        latents, latent_ids, latent_height, latent_width = Flux2LatentCreator.prepare_packed_latents(
            seed=seed,
            height=config.height,
            width=config.width,
            batch_size=1,
        )

        # 4. Denoising loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                # 5.t Predict the noise
                noise = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=config.scheduler.timesteps[t],
                    img_ids=latent_ids,
                    txt_ids=text_ids,
                    guidance=None,
                )

                # 7.t Take one denoise step
                latents = config.scheduler.step(
                    noise=noise, timestep=t, latents=latents, sigmas=config.scheduler.sigmas
                )

                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        ctx.after_loop(latents)

        # 8. Decode latents
        packed_latents = latents.reshape(latents.shape[0], latent_height, latent_width, latents.shape[-1]).transpose(0, 3, 1, 2)  # fmt: off
        decoded = self.vae.decode_packed_latents(packed_latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            negative_prompt=None,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Flux2KleinWeightDefinition,
        )
