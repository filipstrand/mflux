from pathlib import Path

import mlx.core as mx
import numpy as np
import PIL.Image

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.flux2.model.flux2_text_encoder.prompt_encoder import Flux2PromptEncoder
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.flux2.model.flux2_transformer.transformer import Flux2Transformer
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.image_util import ImageUtil


class Flux2KleinDebug:
    @staticmethod
    def decode_packed_latents(
        vae: Flux2VAE,
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
        if latents.ndim >= 4 and latents.shape[1] == vae.latent_channels:
            decoded = vae.decode(latents)
        else:
            decoded = vae.decode_packed_latents(latents)
        normalized = ImageUtil._denormalize(decoded)
        image = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(normalized))
        if output_path is not None:
            image.save(output_path)
        return image

    @staticmethod
    def roundtrip_image(
        vae: Flux2VAE,
        image_path: str | Path,
        output_path: str | Path | None = None,
    ) -> PIL.Image.Image:
        image = ImageUtil.load_image(image_path)
        image_array = ImageUtil.to_array(image)
        latents = vae.encode(image_array)
        decoded = vae.decode(latents)
        normalized = ImageUtil._denormalize(decoded)
        out_image = ImageUtil._numpy_to_pil(ImageUtil._to_numpy(normalized))
        if output_path is not None:
            out_image.save(output_path)
        return out_image

    @staticmethod
    def transformer_step(
        transformer: Flux2Transformer,
        inputs_path: str | Path,
        output_path: str | Path | None = None,
    ) -> dict[str, float]:
        data = np.load(Path(inputs_path))
        latents = mx.array(data["latents"])
        latent_ids = mx.array(data["latent_ids"])
        prompt_embeds = mx.array(data["prompt_embeds"])
        text_ids = mx.array(data["text_ids"])
        timestep = mx.array(data["timestep"])

        noise_pred = transformer(
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

    @staticmethod
    def text_encoder(
        text_encoder: Qwen3TextEncoder,
        tokenizer: Tokenizer,
        prompt: str | list[str],
        inputs_path: str | Path,
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int, ...] = (9, 18, 27),
    ) -> dict[str, float]:
        data = np.load(Path(inputs_path))
        ref_prompt_embeds = mx.array(data["prompt_embeds"])
        ref_text_ids = mx.array(data["text_ids"])

        tokens = tokenizer.tokenize(prompt=prompt, max_length=max_sequence_length)
        prompt_embeds = text_encoder.get_prompt_embeds(
            input_ids=tokens.input_ids,
            attention_mask=tokens.attention_mask,
            hidden_state_layers=text_encoder_out_layers,
        )
        text_ids = Flux2PromptEncoder.prepare_text_ids(prompt_embeds)

        metrics: dict[str, float] = {}
        diff = mx.abs(prompt_embeds - ref_prompt_embeds)
        metrics["mean_abs_error"] = float(mx.mean(diff))
        metrics["max_abs_error"] = float(mx.max(diff))
        metrics["text_ids_match"] = float(mx.mean((text_ids == ref_text_ids).astype(mx.float32)))
        return metrics
