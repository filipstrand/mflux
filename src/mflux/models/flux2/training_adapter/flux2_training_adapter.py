from __future__ import annotations

from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.training_adapter.flux2_base_training_adapter import Flux2BaseTrainingAdapter
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein


class Flux2TrainingAdapter(Flux2BaseTrainingAdapter):
    def __init__(self, *, model_config: ModelConfig, quantize: int | None):
        super().__init__(
            model_config=model_config,
            quantize=quantize,
            model_factory=lambda cfg, q: Flux2Klein(model_config=cfg, quantize=q),
        )

    def encode_data(
        self,
        *,
        data_id: int,
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,
    ) -> tuple[mx.array, Any]:
        clean_latents, img_ids = self._encode_output_latents(image_path=image_path, width=width, height=height)
        prompt_embeds, text_ids = self._encode_prompt(prompt=prompt)

        mx.eval(clean_latents, prompt_embeds, text_ids, img_ids)
        return clean_latents, {"prompt_embeds": prompt_embeds, "text_ids": text_ids, "img_ids": img_ids}

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:  # noqa: ARG002
        return self._flux2.transformer(
            hidden_states=latents_t,
            encoder_hidden_states=cond["prompt_embeds"],
            timestep=config.scheduler.timesteps[t],
            img_ids=cond["img_ids"],
            txt_ids=cond["text_ids"],
            guidance=None,
        )

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,
    ):
        with self._assistant_disabled():
            image = self._flux2.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=self._guidance,
            )
        self._flux2.prompt_cache = {}
        return image.image
