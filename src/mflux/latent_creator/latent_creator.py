import mlx.core as mx

from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.image_util import ImageUtil


class Img2Img:
    def __init__(
        self,
        vae: VAE,
        sigmas: mx.array,
        init_time_step: int,
        image_path: int,
    ):
        self.vae = vae
        self.sigmas = sigmas
        self.init_time_step = init_time_step
        self.image_path = image_path


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
        height: int,
        width: int,
        img2img: Img2Img,
    ) -> mx.array:
        # 0. Determine type of image generation
        is_text2img = img2img.image_path is None

        if is_text2img:
            # 1. Create the pure noise
            return LatentCreator.create(
                seed=seed,
                height=height,
                width=width,
            )
        else:
            # 1. Create the pure noise
            pure_noise = LatentCreator.create(
                seed=seed,
                height=height,
                width=width,
            )

            # 2. Encode the image
            scaled_user_image = ImageUtil.scale_to_dimensions(
                image=ImageUtil.load_image(img2img.init_image_path).convert("RGB"),
                target_width=width,
                target_height=height,
            )
            encoded = img2img.vae.encode(ImageUtil.to_array(scaled_user_image))
            latents = ArrayUtil.pack_latents(latents=encoded, height=height, width=width)

            # 3. Find the appropriate sigma value
            sigma = img2img.sigmas[img2img.init_time_step]

            # 4. Blend the appropriate amount of noise based on linear interpolation
            return LatentCreator.add_noise_by_interpolation(
                clean=latents,
                noise=pure_noise,
                sigma=sigma
            )  # fmt: off

    @staticmethod
    def add_noise_by_interpolation(clean: mx.array, noise: mx.array, sigma: float) -> mx.array:
        return (1 - sigma) * clean + sigma * noise
