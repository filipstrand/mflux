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
    ):
        return mx.random.normal(
            shape=[1, (height // 16) * (width // 16), 64],
            key=mx.random.key(seed)
        )  # fmt: off

    @staticmethod
    def create_for_txt2img_or_img2img(
        seed: int,
        runtime_conf: RuntimeConfig,
        vae: nn.Module,
    ):
        noise = LatentCreator.create(
            seed=seed,
            height=runtime_conf.height,
            width=runtime_conf.width,
        )

        if runtime_conf.config.init_image_path is None:
            # Text2Image
            return noise
        else:
            # Image2Image
            user_image = ImageUtil.load_image(runtime_conf.config.init_image_path).convert("RGB")
            scaled_user_image = ImageUtil.scale_to_dimensions(user_image, runtime_conf.width, runtime_conf.height)
            encoded = vae.encode(ImageUtil.to_array(scaled_user_image))
            latents = ArrayUtil.pack_latents(encoded, runtime_conf.width, runtime_conf.height)
            sigmas_for_init_image_strength = runtime_conf.sigmas[runtime_conf.init_time_step]
            latents_adjusted = latents * (1.0 - sigmas_for_init_image_strength)
            noise_adjusted = noise * sigmas_for_init_image_strength
            return latents_adjusted + noise_adjusted
