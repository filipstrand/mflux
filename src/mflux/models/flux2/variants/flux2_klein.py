from pathlib import Path

import mlx.core as mx
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

        decoded = self.vae.decode_packed_latents(mx.array(packed_latents))
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
