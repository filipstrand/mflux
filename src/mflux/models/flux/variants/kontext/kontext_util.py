import mlx.core as mx

from mflux.models.common.latent_creator.latent_creator import LatentCreator
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.flux.latent_creator.flux_latent_creator import FluxLatentCreator


class KontextUtil:
    @staticmethod
    def create_image_conditioning_latents(
        vae,
        height: int,
        width: int,
        image_path: str,
        tiling_config: TilingConfig | None = None,
    ) -> tuple[mx.array, mx.array]:
        # Load and encode the input image
        input_image = LatentCreator.encode_image(
            vae=vae,
            image_path=image_path,
            height=height,
            width=width,
            tiling_config=tiling_config,
        )

        # Pack image latents for conditioning
        image_latents = FluxLatentCreator.pack_latents(
            latents=input_image,
            height=height,
            width=width,
            num_channels_latents=16,
        )

        # Create image IDs for positional embeddings
        image_ids = KontextUtil._create_image_ids(
            height=height,
            width=width,
        )

        return image_latents, image_ids

    @staticmethod
    def _create_image_ids(
        height: int,
        width: int,
    ) -> mx.array:
        # Create image IDs similar to the reference implementation
        latent_height = height // 16  # VAE downsampling factor
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
