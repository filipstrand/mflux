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
