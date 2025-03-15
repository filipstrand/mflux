import mlx.core as mx

from mflux.config.runtime_config import RuntimeConfig
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil


class MaskUtil:
    @staticmethod
    def create_masked_latents(
        vae: VAE,
        config: RuntimeConfig,
        latents: mx.array,
        img_path: str,
        mask_path: str | None
    ) -> mx.array:  # fmt: off
        from mflux import ImageUtil

        # 1. Get the reference image
        scaled_image = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(config.image_path).convert("RGB"),
            target_width=config.width,
            target_height=config.height,
        )
        image = ImageUtil.to_array(scaled_image)

        # 2. Get the mask
        scaled = ImageUtil.scale_to_dimensions(
            image=ImageUtil.load_image(mask_path).convert("RGB"),
            target_width=config.width,
            target_height=config.height,
        )
        the_mask = ImageUtil.to_array(scaled, is_mask=True)

        # 3. Create and pack the masked image
        masked_image = image * (1 - the_mask)
        masked_image = vae.encode(masked_image)
        masked_image = ArrayUtil.pack_latents(latents=masked_image, height=config.height, width=config.width)

        # 4. Resize mask and pack latents
        mask = MaskUtil._reshape_mask(the_mask=the_mask, height=config.height, width=config.width)
        mask = ArrayUtil.pack_latents(latents=mask, height=config.height, width=config.width, num_channels_latents=64)

        # 5. Concat the masked_image and the mask
        masked_image_latents = mx.concatenate([masked_image, mask], axis=-1)
        return masked_image_latents

    @staticmethod
    def _reshape_mask(the_mask: mx.array, height: int, width: int):
        mask = the_mask[:, 0, :, :]
        mask = mx.reshape(mask, (1, height // 8, 8, width // 8, 8))
        mask = mx.transpose(mask, (0, 2, 4, 1, 3))
        mask = mx.reshape(mask, (1, 64, height // 8, width // 8))
        return mask
