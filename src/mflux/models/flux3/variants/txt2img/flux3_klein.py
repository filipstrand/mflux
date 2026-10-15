from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux3.flux3_initializer import Flux3Initializer
from mflux.models.flux3.latent_creator.flux3_latent_creator import Flux3LatentCreator
from mflux.models.flux3.model.flux3_scheduler.flux3_flow_map_scheduler import Flux3FlowMapScheduler
from mflux.models.flux3.model.flux3_text_encoder.prompt_encoder import Flux3PromptEncoder
from mflux.models.flux3.model.flux3_transformer.flux3_transformer import Flux3Transformer
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Flux3Klein(nn.Module):
    """FLUX 3 Image Klein. The distilled checkpoint is a flow map (16 steps, no CFG); the Base checkpoint is
    undistilled (50 steps) and runs classifier-free guidance against the empty prompt."""

    vae: Flux2VAE
    transformer: Flux3Transformer
    text_encoder: Qwen3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        compute_precision: mx.Dtype | None = None,
    ):
        super().__init__()
        Flux3Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config or ModelConfig.flux3_image_klein(),
            compute_precision=compute_precision,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
    ) -> GeneratedImage:
        return self._generate(
            seed=seed,
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
        )

    def _resolve_sampling(self, num_inference_steps: int | None, guidance: float | None) -> tuple[int, float | None]:
        # Steps default per checkpoint (16 distilled, 50 Base); guidance only exists for Base (default 3.5).
        if num_inference_steps is None:
            num_inference_steps = 50 if self.model_config.supports_guidance else 16
        if not self.model_config.supports_guidance:
            return num_inference_steps, None
        return num_inference_steps, ui_defaults.GUIDANCE_SCALE if guidance is None else float(guidance)

    def _generate(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None,
        height: int,
        width: int,
        guidance: float | None = None,
        cond_tokens: mx.array | None = None,
        cond_ids: mx.array | None = None,
        image_paths: list[Path | str] | None = None,
        crop_box: tuple[int, int, int, int] | None = None,
    ) -> GeneratedImage:
        # Shared by the txt2img and edit variants: cond_* are the encoded reference images (None for txt2img).
        num_inference_steps, guidance = self._resolve_sampling(num_inference_steps, guidance)
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
        )
        text_tokens, text_ids = self._encode_prompt(prompt)
        # CFG's unconditional branch is the empty prompt; encoded here because --low-ram frees the text encoder
        # before the loop.
        uncond = None if self.transformer.flow_map else self._encode_prompt("")
        h, w = Flux3LatentCreator.latent_shape(config.height, config.width)
        latents = self._noise_override
        if latents is None:
            latents = Flux3LatentCreator.create_noise(seed, config.height, config.width)
        latent_ids = Flux3LatentCreator.prepare_ids(h, w)

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        latents = self.denoise(
            latents=latents,
            latent_ids=latent_ids,
            text_tokens=text_tokens,
            text_ids=text_ids,
            config=config,
            cond_tokens=cond_tokens,
            cond_ids=cond_ids,
            ctx=ctx,
            uncond=uncond,
        )
        ctx.after_loop(latents)

        decoded = self.decode_latents(latents, h, w)
        generated = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            image_paths=image_paths,
            generation_time=config.time_steps.format_dict["elapsed"],
        )
        if crop_box is not None:
            generated.image = generated.image.crop(crop_box)
            generated.width, generated.height = generated.image.size
        return generated

    def denoise(
        self,
        latents: mx.array,
        latent_ids: mx.array,
        text_tokens: mx.array,
        text_ids: mx.array,
        config: Config,
        cond_tokens: mx.array | None = None,
        cond_ids: mx.array | None = None,
        ctx=None,
        uncond: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        # Same schedule and Euler update for both checkpoints; the flow-map scheduler is just that.
        scheduler = Flux3FlowMapScheduler(num_inference_steps=config.num_inference_steps)
        for step in config.time_steps:
            try:
                timestep = scheduler.timesteps[step : step + 1]
                target = scheduler.timesteps[step + 1 : step + 2]
                velocity = self._velocity(latents, latent_ids, text_tokens, text_ids, timestep, target, cond_tokens, cond_ids)  # fmt: skip
                if uncond is not None:
                    # CFG: v = u + g (c - u), combined in float32 like the float32 latent state; one forward per branch.
                    unconditional = self._velocity(latents, latent_ids, *uncond, timestep, target, cond_tokens, cond_ids)  # fmt: skip
                    velocity = unconditional.astype(mx.float32) + config.guidance * (
                        velocity.astype(mx.float32) - unconditional.astype(mx.float32)
                    )
                latents = scheduler.step(latents=latents, velocity=velocity, step=step)
                if ctx is not None:
                    ctx.in_loop(step, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                if ctx is not None:
                    ctx.interruption(step, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {step + 1}/{config.num_inference_steps}"
                )
        return latents

    def _velocity(self, latents, latent_ids, text_tokens, text_ids, timestep, target, cond_tokens, cond_ids):
        return self.transformer(
            hidden_states=latents,
            img_ids=latent_ids,
            encoder_hidden_states=text_tokens,
            txt_ids=text_ids,
            timestep=timestep,
            target_timestep=target if self.transformer.flow_map else None,
            cond_hidden_states=cond_tokens,
            cond_ids=cond_ids,
        )

    def decode_latents(self, latents: mx.array, h: int, w: int) -> mx.array:
        return Flux3LatentCreator.decode_tokens(self.vae, latents, h, w)

    def _encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        if prompt not in self.prompt_cache:
            self.prompt_cache[prompt] = Flux3PromptEncoder.encode_prompt(
                prompt=prompt,
                tokenizer=self.tokenizers["qwen3_vl"],
                text_encoder=self.text_encoder,
            )
        return self.prompt_cache[prompt]
