import mlx.core as mx
from mlx import nn

from mflux.config.runtime_config import RuntimeConfig
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.image_util import ImageUtil


class LatentCreator:
    @staticmethod
    def create(
        seed: int,
        height: int,
        width: int,
    ) -> mx.array:
        return mx.random.normal(
            shape=[1, (height // 16) * (width // 16), 64],
            key=mx.random.key(seed)
        )  # fmt: off

    @staticmethod
    def create_for_txt2img_or_img2img(
        seed: int,
        runtime_conf: RuntimeConfig,
        vae: nn.Module,
    ) -> mx.array:
        pure_noise = LatentCreator.create(
            seed=seed,
            height=runtime_conf.height,
            width=runtime_conf.width,
        )

        if runtime_conf.config.init_image_path is None:
            # Text2Image
            return pure_noise
        else:
            # Image2Image
            user_image = ImageUtil.load_image(runtime_conf.config.init_image_path).convert("RGB")
            scaled_user_image = ImageUtil.scale_to_dimensions(
                image=user_image,
                target_width=runtime_conf.width,
                target_height=runtime_conf.height,
            )
            encoded = vae.encode(ImageUtil.to_array(scaled_user_image))
            latents = ArrayUtil.pack_latents(latents=encoded, height=runtime_conf.height, width=runtime_conf.width)
            sigma = runtime_conf.sigmas[runtime_conf.init_time_step]
            return LatentCreator.add_noise_by_interpolation(
                clean=latents,
                noise=pure_noise,
                sigma=sigma
            )  # fmt: off

    @staticmethod
    def add_noise_by_interpolation(clean: mx.array, noise: mx.array, sigma: float) -> mx.array:
        return (1 - sigma) * clean + sigma * noise
