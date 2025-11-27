from pathlib import Path

import mlx.core as mx

from mflux.models.flux.model.redux_encoder.redux_encoder import ReduxEncoder
from mflux.models.flux.model.siglip_vision_transformer.siglip_vision_transformer import SiglipVisionTransformer
from mflux.utils.image_util import ImageUtil


class ReduxUtil:
    @staticmethod
    def validate_redux_image_strengths(
        redux_image_paths: list[Path],
        redux_image_strengths: list[float] | None,
    ) -> list[float] | None:
        if not redux_image_strengths or len(redux_image_strengths) == 0:
            return redux_image_strengths

        # If strengths provided but not enough for all images, fill with default (1.0)
        if len(redux_image_strengths) < len(redux_image_paths):
            redux_image_strengths.extend([1.0] * (len(redux_image_paths) - len(redux_image_strengths)))

        # If too many strengths provided, raise error
        elif len(redux_image_strengths) > len(redux_image_paths):
            raise ValueError(
                f"Too many strengths provided ({len(redux_image_strengths)}), expted at most {len(redux_image_paths)}."
            )

        return redux_image_strengths

    @staticmethod
    def embed_images(
        image_paths: list[str] | list[Path],
        image_encoder: SiglipVisionTransformer,
        image_embedder: ReduxEncoder,
        image_strengths: list[float] | None = None,
    ) -> list[mx.array]:  # fmt:off
        image_embeds_list = []
        for idx, image_path in enumerate(image_paths):
            # Get the strength for this image (default to 1.0 if not specified)
            strength = 1.0
            if image_strengths is not None and idx < len(image_strengths):
                strength = image_strengths[idx]

            image_embeds = ReduxUtil._embed_single_image(
                image_path=image_path,
                image_encoder=image_encoder,
                image_embedder=image_embedder,
                strength=strength,
            )
            image_embeds_list.append(image_embeds)
        return image_embeds_list

    @staticmethod
    def _embed_single_image(
        image_path: str | Path,
        image_encoder: SiglipVisionTransformer,
        image_embedder: ReduxEncoder,
        strength: float = 1.0,
    ) -> mx.array:  # fmt:off
        image = ImageUtil.load_image(image_path).convert("RGB")
        image = ImageUtil.preprocess_for_model(image=image)
        image_latents, pooler_output = image_encoder(image)
        image_embeds = image_embedder(image_latents)

        # Apply strength factor to the image embeddings
        if strength != 1.0:
            image_embeds = image_embeds * strength

        return image_embeds
