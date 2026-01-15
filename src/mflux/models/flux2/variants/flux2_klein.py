from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.flux2_initializer import Flux2Initializer
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
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
            model_config=model_config or ModelConfig.from_name("flux2-klein-4b"),
        )

    def generate_image(self, **kwargs):
        raise NotImplementedError("Flux2Klein generation will be implemented after core ports.")

    def encode_prompt(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> tuple[mx.array, mx.array]:
        prompt_embeds = self._get_qwen3_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            hidden_state_layers=text_encoder_out_layers,
        )
        if num_images_per_prompt > 1:
            prompt_embeds = mx.repeat(prompt_embeds, num_images_per_prompt, axis=0)
        text_ids = self._prepare_text_ids(prompt_embeds)
        return prompt_embeds, text_ids

    def _get_qwen3_prompt_embeds(
        self,
        prompt: str | list[str],
        max_sequence_length: int,
        hidden_state_layers: tuple[int, ...],
    ) -> mx.array:
        tokenizer = self.tokenizers["qwen3"]
        tokens = tokenizer.tokenize(prompt=prompt, max_length=max_sequence_length)
        return self.text_encoder.get_prompt_embeds(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            hidden_state_layers=hidden_state_layers,
        )

    @staticmethod
    def _prepare_text_ids(x: mx.array, t_coord: mx.array | None = None) -> mx.array:
        batch_size, seq_len, _ = x.shape
        out_ids = []
        for i in range(batch_size):
            if t_coord is None:
                t = mx.zeros((seq_len,), dtype=mx.int32)
            else:
                t = t_coord[i]
                if t.ndim == 0:
                    t = mx.full((seq_len,), t, dtype=mx.int32)
                elif t.shape[0] != seq_len:
                    t = mx.broadcast_to(t, (seq_len,))
                t = t.astype(mx.int32)
            h = mx.zeros((seq_len,), dtype=mx.int32)
            w = mx.zeros((seq_len,), dtype=mx.int32)
            token_ids = mx.arange(seq_len, dtype=mx.int32)
            coords = mx.stack([t, h, w, token_ids], axis=1)
            out_ids.append(coords)
        return mx.stack(out_ids, axis=0)

    def debug_decode_packed_latents(
        self,
        latents_path: str | Path,
        output_path: str | Path | None = None,
    ) -> PIL.Image.Image:
        latents_path = Path(latents_path)
        data = mx.load(str(latents_path))
        if isinstance(data, dict):
            packed_latents = next(iter(data.values()))
        else:
            packed_latents = data

        latents = mx.array(packed_latents)
        if latents.ndim >= 4 and latents.shape[1] == self.vae.latent_channels:
            decoded = self.vae.decode(latents)
        else:
            decoded = self.vae.decode_packed_latents(latents)
        normalized = ImageUtil._denormalize(decoded)
        image = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(normalized))
        if output_path is not None:
            image.save(output_path)
        return image

    def debug_roundtrip_image(
        self,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> PIL.Image.Image:
        image = ImageUtil.load_image(image_path)
        image_array = ImageUtil.to_array(image)
        latents = self.vae.encode(image_array)
        decoded = self.vae.decode(latents)
        normalized = ImageUtil._denormalize(decoded)
        out_image = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(normalized))
        if output_path is not None:
            out_image.save(output_path)
        return out_image

    def debug_transformer_step(
        self,
        inputs_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, float]:
        data = np.load(Path(inputs_path))
        latents = mx.array(data["latents"])
        latent_ids = mx.array(data["latent_ids"])
        prompt_embeds = mx.array(data["prompt_embeds"])
        text_ids = mx.array(data["text_ids"])
        timestep = mx.array(data["timestep"])

        noise_pred = self.transformer(
            hidden_states=latents,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep / 1000,
            img_ids=latent_ids,
            txt_ids=text_ids,
            guidance=None,
        )

        if output_path is not None:
            np.save(Path(output_path), np.array(noise_pred))

        metrics: dict[str, float] = {}
        if "noise_pred" in data:
            ref = mx.array(data["noise_pred"])
            diff = mx.abs(noise_pred - ref)
            metrics["mean_abs_error"] = float(mx.mean(diff))
            metrics["max_abs_error"] = float(mx.max(diff))
        return metrics

    def debug_text_encoder(
        self,
        prompt: str | list[str],
        inputs_path: str | Path,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> dict[str, float]:
        data = np.load(Path(inputs_path))
        ref_prompt_embeds = mx.array(data["prompt_embeds"])
        ref_text_ids = mx.array(data["text_ids"])

        prompt_embeds = self._get_qwen3_prompt_embeds(
            prompt=prompt,
            max_sequence_length=max_sequence_length,
            hidden_state_layers=text_encoder_out_layers,
        )
        text_ids = self._prepare_text_ids(prompt_embeds)

        metrics: dict[str, float] = {}
        diff = mx.abs(prompt_embeds - ref_prompt_embeds)
        metrics["mean_abs_error"] = float(mx.mean(diff))
        metrics["max_abs_error"] = float(mx.max(diff))
        metrics["text_ids_match"] = float(mx.mean((text_ids == ref_text_ids).astype(mx.float32)))
        return metrics
