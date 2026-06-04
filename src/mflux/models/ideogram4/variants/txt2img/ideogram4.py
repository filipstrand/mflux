from typing import Any

import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.common.config import Config, ModelConfig
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.ideogram4.caption import Ideogram4Caption
from mflux.models.ideogram4.config import validate_dimensions
from mflux.models.ideogram4.constants import (
    IMAGE_POSITION_OFFSET,
    IMAGE_TOKEN_STRIDE,
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    SEQUENCE_PADDING_INDICATOR,
)
from mflux.models.ideogram4.ideogram4_initializer import Ideogram4Initializer
from mflux.models.ideogram4.latent_creator import Ideogram4LatentCreator
from mflux.models.ideogram4.model import Ideogram4Transformer, Qwen3TextEncoder
from mflux.models.ideogram4.scheduler import Ideogram4Scheduler
from mflux.models.ideogram4.weights import Ideogram4WeightDefinition
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil


class Ideogram4(nn.Module):
    vae: Flux2VAE
    text_encoder: Qwen3TextEncoder | None
    conditional_transformer: Ideogram4Transformer
    unconditional_transformer: Ideogram4Transformer

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig = ModelConfig.ideogram4_fp8(),
    ):
        super().__init__()
        Ideogram4Initializer.init(
            model=self,
            quantize=quantize,
            model_path=model_path,
            model_config=model_config,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str | dict[str, Any],
        num_inference_steps: int | None = None,
        height: int = 1024,
        width: int = 1024,
        guidance: float | None = None,
        preset: str | None = None,
        use_preset_steps: bool = False,
        strict_caption_validation: bool = False,
        warn_on_caption_issues: bool = True,
    ) -> GeneratedImage:
        prepared_prompt = Ideogram4Caption.prepare(prompt)
        if strict_caption_validation:
            Ideogram4Caption.raise_for_warnings(prepared_prompt.warnings)
        elif warn_on_caption_issues:
            Ideogram4Caption.emit_warnings(prepared_prompt.warnings, stacklevel=2)
        prompt = prepared_prompt.prompt
        if not prompt:
            raise ValueError("prompt must not be empty")
        validate_dimensions(width=width, height=height)

        sampler = Ideogram4Scheduler.get_preset(preset)
        use_sampler_schedule = use_preset_steps or num_inference_steps is None
        actual_steps = sampler.num_steps if use_sampler_schedule else num_inference_steps
        if actual_steps < 1:
            raise ValueError(f"num_inference_steps must be >= 1, got {actual_steps}")
        guidance_values = (
            sampler.guidance_schedule
            if use_sampler_schedule
            else (float(guidance if guidance is not None else 7.0),) * actual_steps
        )
        config = Config(
            width=width,
            height=height,
            guidance=guidance if guidance is not None else guidance_values[-1],
            scheduler="linear",
            model_config=self.model_config,
            num_inference_steps=actual_steps,
        )

        inputs = self._build_inputs([prompt], height=config.height, width=config.width)
        llm_features = self._encode_prompt(
            prompt=prompt,
            width=config.width,
            height=config.height,
            inputs=inputs,
        )
        z = Ideogram4LatentCreator.create_noise(
            seed=seed,
            width=config.width,
            height=config.height,
            latent_dim=self.conditional_transformer.config.in_channels,
        )
        text_z_padding = mx.zeros(
            (1, int(inputs["max_text_tokens"]), self.conditional_transformer.config.in_channels),
            dtype=mx.float32,
        )
        negative_inputs = self._negative_inputs(inputs, llm_features)
        t_values, s_values = Ideogram4Scheduler.make_timesteps(
            num_steps=actual_steps,
            height=config.height,
            width=config.width,
            mu=sampler.mu,
            std=sampler.std,
        )

        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(z)
        time_steps = config.time_steps
        for step_index in time_steps:
            try:
                schedule_index = actual_steps - 1 - step_index
                z = self._denoise_step(
                    z=z,
                    t_value=float(t_values[schedule_index]),
                    s_value=float(s_values[schedule_index]),
                    guidance_value=float(guidance_values[schedule_index]),
                    text_z_padding=text_z_padding,
                    llm_features=llm_features,
                    inputs=inputs,
                    negative_inputs=negative_inputs,
                )
                ctx.in_loop(step_index, z, time_steps=time_steps)
                mx.eval(z)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(step_index, z, time_steps=time_steps)
                raise StopImageGenerationException(
                    f"Stopping image generation at step {step_index + 1}/{config.num_inference_steps}"
                )
        ctx.after_loop(z)

        decoded = self.vae.decode(Ideogram4LatentCreator.unpack_latents(z, config.height, config.width))
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            generation_time=time_steps.format_dict["elapsed"],
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(
            model=self,
            bits=self.bits,
            base_path=base_path,
            weight_definition=Ideogram4WeightDefinition,
        )

    def _build_inputs(self, prompts: list[str], *, height: int, width: int) -> dict[str, Any]:
        tokenizer = self.tokenizers["ideogram4"]
        tokenized = [(tokenizer.tokenize_one(prompt), 0) for prompt in prompts]
        tokenized = [(tokens, int(tokens.shape[0])) for tokens, _ in tokenized]
        batch_size = len(prompts)
        grid_h = height // IMAGE_TOKEN_STRIDE
        grid_w = width // IMAGE_TOKEN_STRIDE
        num_image_tokens = grid_h * grid_w
        max_text_tokens = max(num_text for _, num_text in tokenized)
        total_seq_len = max_text_tokens + num_image_tokens

        h_idx = np.repeat(np.arange(grid_h, dtype=np.int64), grid_w)
        w_idx = np.tile(np.arange(grid_w, dtype=np.int64), grid_h)
        t_idx = np.zeros_like(h_idx)
        image_pos = np.stack([t_idx, h_idx, w_idx], axis=1) + IMAGE_POSITION_OFFSET

        token_ids = np.zeros((batch_size, total_seq_len), dtype=np.int64)
        text_position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int64)
        position_ids = np.zeros((batch_size, total_seq_len, 3), dtype=np.int64)
        segment_ids = np.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=np.int64)
        indicator = np.zeros((batch_size, total_seq_len), dtype=np.int64)

        for batch_idx, (tokens, num_text) in enumerate(tokenized):
            pad_len = max_text_tokens - num_text
            total_unpadded = num_text + num_image_tokens
            offset = pad_len
            token_ids[batch_idx, offset : offset + num_text] = tokens

            text_pos = np.arange(num_text, dtype=np.int64)
            text_pos_3d = np.stack([text_pos, text_pos, text_pos], axis=1)
            text_position_ids[batch_idx, offset : offset + num_text] = text_pos_3d
            position_ids[batch_idx, offset : offset + num_text] = text_pos_3d
            position_ids[batch_idx, offset + num_text :] = image_pos

            indicator[batch_idx, offset : offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[batch_idx, offset + num_text :] = OUTPUT_IMAGE_INDICATOR
            segment_ids[batch_idx, offset : offset + total_unpadded] = 1

        return {
            "token_ids": mx.array(token_ids, dtype=mx.int32),
            "text_position_ids": mx.array(text_position_ids, dtype=mx.int32),
            "position_ids": mx.array(position_ids, dtype=mx.int32),
            "segment_ids": mx.array(segment_ids, dtype=mx.int32),
            "indicator": mx.array(indicator, dtype=mx.int32),
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    def _encode_prompt(self, prompt: str, width: int, height: int, inputs: dict[str, Any]) -> mx.array:
        cache_key = (prompt, width, height)
        if cache_key in self.prompt_cache:
            return self.prompt_cache[cache_key]
        if self.text_encoder is None:
            raise RuntimeError("Text encoder has been evicted and prompt features are not cached.")
        attention_mask = (inputs["indicator"] == LLM_TOKEN_INDICATOR).astype(mx.int32)
        pos_2d = inputs["text_position_ids"][:, :, 0]
        embeds = self.text_encoder.get_prompt_embeds(
            inputs["token_ids"],
            attention_mask,
            pos_2d,
        )
        embeds = embeds * attention_mask[..., None].astype(embeds.dtype)
        embeds = embeds.astype(mx.float32)
        mx.eval(embeds)
        self.prompt_cache[cache_key] = embeds
        return embeds

    def _negative_inputs(self, inputs: dict[str, Any], llm_features: mx.array) -> dict[str, mx.array]:
        max_text_tokens = int(inputs["max_text_tokens"])
        num_image_tokens = int(inputs["num_image_tokens"])
        return {
            "position_ids": inputs["position_ids"][:, max_text_tokens:, :],
            "segment_ids": inputs["segment_ids"][:, max_text_tokens:],
            "indicator": inputs["indicator"][:, max_text_tokens:],
            "llm_features": mx.zeros((1, num_image_tokens, llm_features.shape[-1]), dtype=llm_features.dtype),
        }

    def _denoise_step(
        self,
        *,
        z: mx.array,
        t_value: float,
        s_value: float,
        guidance_value: float,
        text_z_padding: mx.array,
        llm_features: mx.array,
        inputs: dict[str, Any],
        negative_inputs: dict[str, mx.array],
    ) -> mx.array:
        t = mx.full((1,), t_value, dtype=mx.float32)
        pos_z = mx.concatenate([text_z_padding, z], axis=1)
        pos_out = self.conditional_transformer(
            llm_features=llm_features,
            x=pos_z,
            t=t,
            position_ids=inputs["position_ids"],
            segment_ids=inputs["segment_ids"],
            indicator=inputs["indicator"],
        )
        pos_v = pos_out[:, int(inputs["max_text_tokens"]) :, :]
        neg_v = self.unconditional_transformer(
            llm_features=negative_inputs["llm_features"],
            x=z,
            t=t,
            position_ids=negative_inputs["position_ids"],
            segment_ids=negative_inputs["segment_ids"],
            indicator=negative_inputs["indicator"],
        )
        v = guidance_value * pos_v + (1.0 - guidance_value) * neg_v
        return z + v * (s_value - t_value)
