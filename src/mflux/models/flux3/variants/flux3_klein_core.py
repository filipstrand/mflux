from pathlib import Path

import mlx.core as mx
from mlx import nn

from mflux.cli.defaults import defaults as ui_defaults
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
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


class _Flux3KleinCore(nn.Module):
    vae: Flux2VAE
    transformer: Flux3Transformer
    text_encoder: Qwen3TextEncoder

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        compute_precision: mx.Dtype | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ):
        super().__init__()
        Flux3Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            model_config=model_config or ModelConfig.flux3_image_klein(),
            compute_precision=compute_precision,
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Flux3Initializer.weight_definition(self.model_config),
        )

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
        flow_map = self.transformer.flow_map

        # 1. Encode the prompt; CFG's unconditional branch is the empty prompt, encoded here because --low-ram
        # frees the text encoder before the loop.
        text_tokens, text_ids = self._encode_prompt(prompt)
        uncond_tokens, uncond_ids = (None, None) if flow_map else self._encode_prompt("")

        # 2. Prepare latents
        h, w = Flux3LatentCreator.latent_shape(config.height, config.width)
        latents = self._noise_override
        if latents is None:
            latents = Flux3LatentCreator.create_noise(seed, config.height, config.width)
        latent_ids = Flux3LatentCreator.prepare_ids(h, w)

        # 3. Denoising loop: one Euler update per step; the flow map and Base share the schedule.
        scheduler = Flux3FlowMapScheduler(num_inference_steps=config.num_inference_steps)
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        predict = self._predict(self.transformer)
        for t in config.time_steps:
            try:
                velocity = predict(
                    latents=latents,
                    latent_ids=latent_ids,
                    text_tokens=text_tokens,
                    text_ids=text_ids,
                    uncond_tokens=uncond_tokens,
                    uncond_ids=uncond_ids,
                    cond_tokens=cond_tokens,
                    cond_ids=cond_ids,
                    guidance=guidance,
                    timestep=scheduler.timesteps[t : t + 1],
                    target_timestep=scheduler.timesteps[t + 1 : t + 2] if flow_map else None,
                )
                latents = scheduler.step(latents=latents, velocity=velocity, step=t)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {t + 1}/{config.num_inference_steps}"
                )

        # The predict closure holds the transformer. Delete it so that --low-ram can free the transformer before decode.
        del predict
        ctx.after_loop(latents)

        # 4. Decode latents
        decoded = self._decode_latents(latents, h, w)
        generated = ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_paths=image_paths,
            generation_time=config.time_steps.format_dict["elapsed"],
            generation_parameters=self.compute_precision.generation_parameters(),
        )
        if crop_box is not None:
            generated.image = generated.image.crop(crop_box)
            generated.width, generated.height = generated.image.size
        return generated

    def _resolve_sampling(self, num_inference_steps: int | None, guidance: float | None) -> tuple[int, float | None]:
        # Steps default per checkpoint (16 distilled, 50 Base); guidance only exists for Base (default 3.5).
        if num_inference_steps is None:
            num_inference_steps = 50 if self.model_config.supports_guidance else 16
        if not self.model_config.supports_guidance:
            return num_inference_steps, None
        return num_inference_steps, ui_defaults.GUIDANCE_SCALE if guidance is None else float(guidance)

    def _encode_prompt(self, prompt: str) -> tuple[mx.array, mx.array]:
        if prompt not in self.prompt_cache:
            self.prompt_cache[prompt] = Flux3PromptEncoder.encode_prompt(
                prompt=prompt,
                tokenizer=self.tokenizers["qwen3_vl"],
                text_encoder=self.text_encoder,
            )
        return self.prompt_cache[prompt]

    def _decode_latents(self, latents: mx.array, h: int, w: int) -> mx.array:
        return Flux3LatentCreator.decode_tokens(self.vae, latents, h, w, tiling_config=self.tiling_config)

    @staticmethod
    def _predict(transformer):
        def predict(
            latents: mx.array,
            latent_ids: mx.array,
            text_tokens: mx.array,
            text_ids: mx.array,
            uncond_tokens: mx.array | None,
            uncond_ids: mx.array | None,
            cond_tokens: mx.array | None,
            cond_ids: mx.array | None,
            guidance: float | None,
            timestep: mx.array,
            target_timestep: mx.array | None,
        ):
            def forward(tokens: mx.array, ids: mx.array) -> mx.array:
                return transformer(
                    hidden_states=latents,
                    img_ids=latent_ids,
                    encoder_hidden_states=tokens,
                    txt_ids=ids,
                    timestep=timestep,
                    target_timestep=target_timestep,
                    cond_hidden_states=cond_tokens,
                    cond_ids=cond_ids,
                )

            velocity = forward(text_tokens, text_ids)
            if uncond_tokens is None:
                return velocity
            # CFG: v = u + g (c - u), combined in float32 like the float32 latent state.
            unconditional = forward(uncond_tokens, uncond_ids).astype(mx.float32)
            return unconditional + guidance * (velocity.astype(mx.float32) - unconditional)

        # Not mx.compile'd: the RoPE angles are built on the host in float64 (Flux3PosEmbed), which cannot be traced.
        return predict
