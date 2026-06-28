import math

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.krea2.krea2_initializer import Krea2Initializer
from mflux.models.krea2.latent_creator.krea2_latent_creator import (
    PATCH_SIZE,
    VAE_SCALE_FACTOR,
    Krea2LatentCreator,
)
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

NUM_TRAIN_TIMESTEPS = 1000
DISTILLED_MU = 1.15


class Krea2(nn.Module):
    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        model_config: ModelConfig | None = None,
    ):
        super().__init__()
        Krea2Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            model_config=model_config or ModelConfig.krea2_turbo(),
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int = 8,
        height: int = 1024,
        width: int = 1024,
        guidance: float = 0.0,
        negative_prompt: str | None = None,
    ) -> GeneratedImage:
        multiple = VAE_SCALE_FACTOR * PATCH_SIZE
        height = (height // multiple) * multiple
        width = (width // multiple) * multiple

        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
        )

        # 1. Encode prompt -> (B, text_seq, 12, 2560), mask (B, text_seq)
        prompt_embeds, prompt_mask = Krea2PromptEncoder.encode_prompt(
            prompt=prompt,
            tokenizer=self.tokenizers["qwen"],
            text_encoder=self.text_encoder,
            max_sequence_length=512,
        )
        prompt_embeds = prompt_embeds.astype(ModelConfig.precision)

        do_cfg = guidance is not None and guidance > 0.0
        negative_prompt_embeds = None
        negative_prompt_mask = None
        if do_cfg:
            neg = negative_prompt if negative_prompt is not None else ""
            negative_prompt_embeds, negative_prompt_mask = Krea2PromptEncoder.encode_prompt(
                prompt=neg,
                tokenizer=self.tokenizers["qwen"],
                text_encoder=self.text_encoder,
                max_sequence_length=512,
            )
            negative_prompt_embeds = negative_prompt_embeds.astype(ModelConfig.precision)

        # 2. Latents + position ids + sigma schedule
        latents = Krea2LatentCreator.create_packed_noise(seed=seed, height=height, width=width, batch_size=1)
        latents = latents.astype(ModelConfig.precision)
        grid_h = height // (VAE_SCALE_FACTOR * PATCH_SIZE)
        grid_w = width // (VAE_SCALE_FACTOR * PATCH_SIZE)
        position_ids = Krea2LatentCreator.prepare_position_ids(prompt_embeds.shape[1], grid_h, grid_w)
        rotary_cos_sin = self.transformer.rotary_emb.compute(position_ids, dtype=ModelConfig.precision)

        sigmas = Krea2._make_sigmas(num_inference_steps)

        # 3. Denoising loop
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                timestep = mx.array([sigmas[t]], dtype=ModelConfig.precision)
                velocity = self.transformer(
                    hidden_states=latents,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    position_ids=position_ids,
                    encoder_attention_mask=prompt_mask,
                    rotary_cos_sin=rotary_cos_sin,
                )
                if do_cfg:
                    neg_velocity = self.transformer(
                        hidden_states=latents,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep=timestep,
                        position_ids=position_ids,
                        encoder_attention_mask=negative_prompt_mask,
                        rotary_cos_sin=rotary_cos_sin,
                    )
                    velocity = velocity + guidance * (velocity - neg_velocity)

                dt = sigmas[t + 1] - sigmas[t]
                latents = latents + dt * velocity.astype(latents.dtype)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(latents)

        # 4. Decode
        unpacked = Krea2LatentCreator.unpack_latents(latents, height=height, width=width)
        decoded = self.vae.decode(unpacked.astype(ModelConfig.precision))
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            generation_time=config.time_steps.format_dict["elapsed"],
            negative_prompt=negative_prompt,
        )

    @staticmethod
    def _make_sigmas(num_inference_steps: int) -> list[float]:
        # Turbo: fixed mu = 1.15, exponential time shift over linspace(1, 1/steps, steps), trailing 0.
        base = [1.0 - i * (1.0 - 1.0 / num_inference_steps) / (num_inference_steps - 1) for i in range(num_inference_steps)]
        shifted = [math.exp(DISTILLED_MU) / (math.exp(DISTILLED_MU) + (1.0 / t - 1.0)) for t in base]
        return shifted + [0.0]
