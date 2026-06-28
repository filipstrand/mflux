from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.adapters.base import TrainingAdapter
from mflux.models.common.training.state.training_spec import TrainingSpec
from mflux.models.common.training.utils import TrainingUtil
from mflux.models.krea2.latent_creator.krea2_latent_creator import (
    LATENT_CHANNELS,
    PATCH_SIZE,
    VAE_SCALE_FACTOR,
    Krea2LatentCreator,
)
from mflux.models.krea2.model.krea2_text_encoder.prompt_encoder import Krea2PromptEncoder
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.utils.version_util import VersionUtil


class Krea2TrainingAdapter(TrainingAdapter):
    """QLoRA training adapter for Krea 2 (flow-matching velocity) on MLX.

    Krea 2 is a single-stream MMDiT with no CFG-unconditional branch, so training is the simple
    flow-matching case: one transformer, trained directly. The trainer noises with
    z = (1 - sigma) * clean + sigma * noise, so the target velocity (noise - clean) IS the model's
    native prediction at timestep = sigma. We therefore feed t = sigma and return the velocity
    unchanged. The base is q8-quantized; the LoRA trains in float over the frozen QuantizedLinear
    base (QLoRA), which fits with gradient checkpointing. Save prefix: transformer.*
    """

    def __init__(self, *, model_config: ModelConfig, quantize: int | None, model_path: str | None = None):
        self._model_config = model_config
        self._krea2 = Krea2(quantize=quantize, model_config=model_config, model_path=model_path)
        self._guidance: float | None = None

    def model(self):
        return self._krea2

    def transformer(self):
        return self._krea2.transformer

    def create_config(self, training_spec: TrainingSpec, *, width: int, height: int) -> Config:
        self._guidance = training_spec.guidance
        return Config(
            model_config=self._model_config,
            num_inference_steps=training_spec.steps,
            width=width,
            height=height,
            guidance=training_spec.guidance,
        )

    def freeze_base(self) -> None:
        self._krea2.vae.freeze()
        self._krea2.transformer.freeze()
        if getattr(self._krea2, "text_encoder", None) is not None:
            self._krea2.text_encoder.freeze()
        # Recompute the 28-block activations in backward instead of storing them, which keeps the
        # ~12B-param activation graph from blowing out RAM. Only the LoRA matrices are trainable.
        self._krea2.transformer.gradient_checkpointing = True

    def encode_data(
        self,
        *,
        data_id: int,  # noqa: ARG002
        image_path: Path,
        prompt: str,
        width: int,
        height: int,
        input_image_path: Path | None = None,  # txt2img: unused  # noqa: ARG002
    ) -> tuple[mx.array, Any]:
        # 1) image -> VAE latents (1, 16, H/8, W/8) -> Krea2 packing (1, seq, 64)
        encoded = LatentCreator.encode_image(
            vae=self._krea2.vae,
            image_path=image_path,
            height=height,
            width=width,
        )
        latent_h = height // VAE_SCALE_FACTOR
        latent_w = width // VAE_SCALE_FACTOR
        clean_latents = Krea2LatentCreator.pack_latents(
            encoded, batch_size=1, num_channels=LATENT_CHANNELS, height=latent_h, width=latent_w
        ).astype(mx.float32)

        # 2) caption -> Qwen3 layer-tap embeddings (1, text_seq, 12, 2560) + mask (1, text_seq)
        caption = Krea2TrainingAdapter._caption_text(prompt)
        prompt_embeds, prompt_mask = Krea2PromptEncoder.encode_prompt(
            prompt=caption,
            tokenizer=self._krea2.tokenizers["qwen"],
            text_encoder=self._krea2.text_encoder,
            max_sequence_length=512,
        )
        prompt_embeds = prompt_embeds.astype(ModelConfig.precision)

        # 3) position ids over [text, image] for the 3-axis RoPE
        grid_h = height // (VAE_SCALE_FACTOR * PATCH_SIZE)
        grid_w = width // (VAE_SCALE_FACTOR * PATCH_SIZE)
        position_ids = Krea2LatentCreator.prepare_position_ids(prompt_embeds.shape[1], grid_h, grid_w)

        mx.eval(clean_latents, prompt_embeds, prompt_mask)
        cond = {
            "prompt_embeds": prompt_embeds,
            "prompt_mask": prompt_mask,
            "position_ids": position_ids,
        }
        return clean_latents, cond

    def predict_noise(self, *, t: int, latents_t: mx.array, sigmas: mx.array, cond: Any, config: Config) -> mx.array:  # noqa: ARG002
        # Flow-matching velocity. The trainer noises with z = (1-sigma)*clean + sigma*noise, so the
        # target velocity (noise - clean) is exactly Krea 2's prediction at timestep = sigma. No
        # 1-sigma remap and no negation (the generic loss expects noise - clean).
        timestep = mx.array([float(sigmas[t])], dtype=ModelConfig.precision)
        velocity = self._krea2.transformer(
            hidden_states=latents_t.astype(ModelConfig.precision),
            encoder_hidden_states=cond["prompt_embeds"],
            timestep=timestep,
            position_ids=cond["position_ids"],
            encoder_attention_mask=cond["prompt_mask"],
        )
        return velocity.astype(latents_t.dtype)

    def generate_preview_image(
        self,
        *,
        seed: int,
        prompt: str,
        width: int,
        height: int,
        steps: int,
        image_paths: list[Path | str] | None = None,  # txt2img: unused  # noqa: ARG002
    ):
        with self._assistant_disabled():
            return self._krea2.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=self._guidance or 0.0,
            )

    def save_lora_adapter(self, *, path: Path, training_spec: TrainingSpec) -> None:
        weights: dict[str, mx.array] = {}
        for target in training_spec.lora_layers.targets:
            if target.blocks is not None:
                for b in target.blocks.get_blocks():
                    self._append_train_lora_weights(weights, target.module_path.format(block=b))
            else:
                self._append_train_lora_weights(weights, target.module_path)
        mx.save_safetensors(
            str(path),
            weights,
            metadata={"mflux_version": VersionUtil.get_mflux_version(), "model": training_spec.model},
        )

    def load_lora_adapter(self, *, path: str | Path) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=self._krea2.transformer,
            lora_paths=[str(path)],
            lora_scales=[1.0],
            role="train",
        )

    def load_training_adapter(self, *, path: str | Path, scale: float = 1.0) -> None:
        LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=self._krea2.transformer,
            lora_paths=[str(path)],
            lora_scales=[float(scale)],
            role="assistant",
        )

    def _assistant_disabled(self):
        return TrainingUtil.assistant_disabled(self._krea2.transformer)

    def _append_train_lora_weights(self, weights: dict[str, mx.array], module_path: str) -> None:
        train_lora = TrainingUtil.get_train_lora(self._krea2.transformer, module_path)
        weights[f"transformer.{module_path}.lora_A.weight"] = mx.transpose(train_lora.lora_A)
        weights[f"transformer.{module_path}.lora_B.weight"] = mx.transpose(train_lora.lora_B)

    @staticmethod
    def _caption_text(prompt: str) -> str:
        # Datasets may store captions as plain text or as the Ideogram-style JSON schema
        # ({"high_level_description": ...}). Krea 2's text encoder takes a describe-the-image
        # Qwen3-VL prompt, so feed natural language: extract high_level_description from JSON,
        # else use the raw string.
        text = prompt.strip()
        if not (text.startswith("{") and text.endswith("}")):
            return text
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return text
        if isinstance(data, dict):
            desc = data.get("high_level_description")
            if isinstance(desc, str) and desc.strip():
                return desc.strip()
        return text
