from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import mlx.core as mx
from mlx import nn

from mflux.utils.image_util import ImageUtil

if TYPE_CHECKING:
    from mflux.models.fibo.latent_creator.fibo_latent_creator import FiboLatentCreator
    from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator
    from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator
    from mflux.models.z_image.latent_creator.z_image_latent_creator import ZImageLatentCreator

    LatentCreatorType: TypeAlias = type[FiboLatentCreator | FluxLatentCreator | QwenLatentCreator | ZImageLatentCreator]


class Img2Img:
    def __init__(
        self,
        vae: nn.Module,
        latent_creator: "LatentCreatorType",
        sigmas: mx.array,
        init_time_step: int,
        image_path: str | Path | None,
    ):
        self.vae = vae
        self.sigmas = sigmas
        self.init_time_step = init_time_step
        self.image_path = image_path
        self.latent_creator = latent_creator


class LatentCreator:
    @staticmethod
    def create_for_txt2img_or_img2img(
        seed: int,
        height: int,
        width: int,
        img2img: Img2Img,
    ) -> mx.array:
        latent_creator = img2img.latent_creator

        if img2img.image_path is None:
            # txt2img: just create noise
            return latent_creator.create_noise(seed, height, width)
        else:
            # img2img: blend encoded image with noise
            pure_noise = latent_creator.create_noise(seed, height, width)
            encoded = LatentCreator.encode_image(vae=img2img.vae, image_path=img2img.image_path, height=height, width=width)  # fmt: off
            latents = latent_creator.pack_latents(encoded, height, width)
            sigma = img2img.sigmas[img2img.init_time_step]
            return LatentCreator.add_noise_by_interpolation(clean=latents, noise=pure_noise, sigma=sigma)

    @staticmethod
    def encode_image(vae: nn.Module, image_path: str | Path, height: int, width: int) -> mx.array:
        scaled_user_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(image_path).convert("RGB"),
            target_width=width,
            target_height=height,
        )
        return vae.encode(ImageUtil.to_array(scaled_user_image))

    @staticmethod
    def add_noise_by_interpolation(clean: mx.array, noise: mx.array, sigma: float) -> mx.array:
        return (1 - sigma) * clean + sigma * noise
