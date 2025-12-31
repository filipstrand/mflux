import os

import mlx.core as mx

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.qwen.latent_creator.qwen_latent_creator import QwenLatentCreator


class QwenEditUtil:
    @staticmethod
    def create_image_conditioning_latents(
        vae,
        height: int,
        width: int,
        image_paths: list[str] | str,
        vl_width: int | None = None,
        vl_height: int | None = None,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[mx.array, mx.array, int, int, int]:
        if not isinstance(image_paths, list):
            image_paths = [str(image_paths)]

        if vl_width is not None and vl_height is not None:
            calc_w, calc_h = vl_width, vl_height
        else:
            from mflux.utils.image_util import ImageUtil

            pil_image = ImageUtil.load_image(image_paths[-1]).convert("RGB")
            img_w, img_h = pil_image.size
            target_area_env = os.getenv("MFLUX_QWEN_VL_TARGET_AREA")
            target_area = int(target_area_env) if target_area_env is not None else 1024 * 1024
            ratio = img_w / img_h
            calc_w = int(round(((target_area * ratio) ** 0.5) / 32) * 32)
            calc_h = int(round((calc_w / ratio) / 32) * 32)

        all_image_latents = []
        for image_path in image_paths:
            input_image = LatentCreator.encode_image(
                vae=vae,
                image_path=image_path,
                height=calc_h,
                width=calc_w,
                tiling_config=tiling_config,
            )

            image_latents = QwenLatentCreator.pack_latents(
                latents=input_image,
                height=calc_h,
                width=calc_w,
                num_channels_latents=16,
            )
            all_image_latents.append(image_latents)

        image_latents = mx.concatenate(all_image_latents, axis=1)

        all_image_ids = []
        for _ in image_paths:
            image_ids = QwenEditUtil._create_image_ids(
                height=calc_h,
                width=calc_w,
            )
            all_image_ids.append(image_ids)
        image_ids = mx.concatenate(all_image_ids, axis=1)

        cond_h_patches = calc_h // 16
        cond_w_patches = calc_w // 16
        num_images = len(image_paths)
        return image_latents, image_ids, cond_h_patches, cond_w_patches, num_images

    @staticmethod
    def _create_image_ids(
        height: int,
        width: int,
    ) -> mx.array:
        latent_height = height // 16
        latent_width = width // 16

        image_ids = mx.zeros((latent_height, latent_width, 3))

        row_coords = mx.arange(0, latent_height)[:, None]
        row_coords = mx.broadcast_to(row_coords, (latent_height, latent_width))
        image_ids = mx.concatenate(
            [
                image_ids[:, :, :1],
                row_coords[:, :, None],
                image_ids[:, :, 2:],
            ],
            axis=2,
        )

        col_coords = mx.arange(0, latent_width)[None, :]
        col_coords = mx.broadcast_to(col_coords, (latent_height, latent_width))
        image_ids = mx.concatenate(
            [
                image_ids[:, :, :2],
                col_coords[:, :, None],
            ],
            axis=2,
        )

        image_ids = mx.reshape(image_ids, (latent_height * latent_width, 3))

        first_dim = mx.ones((image_ids.shape[0], 1))
        image_ids = mx.concatenate([first_dim, image_ids[:, 1:]], axis=1)

        image_ids = mx.expand_dims(image_ids, axis=0)

        return image_ids
