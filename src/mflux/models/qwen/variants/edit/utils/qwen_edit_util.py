import os

import mlx.core as mx

from mflux.latent_creator.latent_creator import LatentCreator
from mflux.post_processing.array_util import ArrayUtil


class QwenEditUtil:
    @staticmethod
    def create_image_conditioning_latents(
        vae,
        height: int,
        width: int,
        image_paths: list[str] | str,
        vl_width: int | None = None,
        vl_height: int | None = None,
    ) -> tuple[mx.array, mx.array, int, int, int]:
        # Normalize to list format (handle str, Path, or list)
        if not isinstance(image_paths, list):
            image_paths = [str(image_paths)]  # Convert Path to string if needed

        # 0) Use VL tokenizer's calculated dimensions if available, otherwise compute from last image
        if vl_width is not None and vl_height is not None:
            calc_w, calc_h = vl_width, vl_height
        else:
            # Fallback: compute dimensions independently from last image (matching PyTorch)
            from mflux.post_processing.image_util import ImageUtil

            pil_image = ImageUtil.load_image(image_paths[-1]).convert("RGB")
            img_w, img_h = pil_image.size
            target_area_env = os.getenv("MFLUX_QWEN_VL_TARGET_AREA")
            target_area = int(target_area_env) if target_area_env is not None else 1024 * 1024
            ratio = img_w / img_h
            calc_w = int(round(((target_area * ratio) ** 0.5) / 32) * 32)
            calc_h = int(round((calc_w / ratio) / 32) * 32)

        # 1) Process each image and encode at resized conditioning resolution
        all_image_latents = []
        for image_path in image_paths:
            input_image = LatentCreator.encode_image(
                vae=vae,
                image_path=image_path,
                height=calc_h,
                width=calc_w,
            )

            # 2) Pack image latents for conditioning according to resized dims
            image_latents = ArrayUtil.pack_latents(
                latents=input_image,
                height=calc_h,
                width=calc_w,
                num_channels_latents=16,  # Qwen uses 16 channels like Flux
            )
            all_image_latents.append(image_latents)

        # 3) Concatenate all image latents along sequence dimension (matching PyTorch line 536)
        # PyTorch: image_latents = torch.cat(all_image_latents, dim=1)
        # In MLX, packed latents are [batch, seq_len, channels], so concatenate along dim=1
        image_latents = mx.concatenate(all_image_latents, axis=1)

        # 4) Create image IDs for positional embeddings (one for each image, then concatenate)
        all_image_ids = []
        for _ in image_paths:
            image_ids = QwenEditUtil._create_image_ids(
                height=calc_h,
                width=calc_w,
            )
            all_image_ids.append(image_ids)
        # Concatenate image IDs along sequence dimension (matching latents)
        image_ids = mx.concatenate(all_image_ids, axis=1)

        # 5) Return latents, ids, and the packed latent patch grid for RoPE.
        # VAE downsamples by 8 and we pack by 2, so the patch grid is H//16, W//16.
        cond_h_patches = calc_h // 16
        cond_w_patches = calc_w // 16
        num_images = len(image_paths)
        print(
            f"🔎 MLX EditUtil: conditioning {num_images} image(s) {calc_w}x{calc_h} → {cond_w_patches}x{cond_h_patches} patches"
        )
        return image_latents, image_ids, cond_h_patches, cond_w_patches, num_images

    @staticmethod
    def _create_image_ids(
        height: int,
        width: int,
    ) -> mx.array:
        # Create image IDs similar to Kontext implementation but adapted for Qwen
        latent_height = height // 16  # VAE downsampling factor (same as Flux)
        latent_width = width // 16

        # Create coordinate grid for image positioning
        image_ids = mx.zeros((latent_height, latent_width, 3))

        # Add row coordinates
        row_coords = mx.arange(0, latent_height)[:, None]
        row_coords = mx.broadcast_to(row_coords, (latent_height, latent_width))
        image_ids = mx.concatenate(
            [
                image_ids[:, :, :1],  # Keep first dimension as 0 for now
                row_coords[:, :, None],  # Set row coordinates
                image_ids[:, :, 2:],  # Keep remaining dimensions
            ],
            axis=2,
        )

        # Add column coordinates
        col_coords = mx.arange(0, latent_width)[None, :]
        col_coords = mx.broadcast_to(col_coords, (latent_height, latent_width))
        image_ids = mx.concatenate(
            [
                image_ids[:, :, :2],  # Keep first two dimensions
                col_coords[:, :, None],  # Set column coordinates
            ],
            axis=2,
        )

        # Reshape to sequence format
        image_ids = mx.reshape(image_ids, (latent_height * latent_width, 3))

        # Set the first dimension to 1 to distinguish from generation latents (which use 0)
        first_dim = mx.ones((image_ids.shape[0], 1))
        image_ids = mx.concatenate([first_dim, image_ids[:, 1:]], axis=1)

        # Add batch dimension
        image_ids = mx.expand_dims(image_ids, axis=0)

        return image_ids
