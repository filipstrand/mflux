import mlx.core as mx

from mflux import ImageUtil
from mflux.models.redux_encoder.redux_encoder import ReduxEncoder
from mflux.models.siglip_vision_transformer.siglip_vision_transformer import SiglipVisionTransformer


class ReduxUtil:
    @staticmethod
    def embed_images(
        image_paths: list[str],
        image_encoder: SiglipVisionTransformer,
        image_embedder: ReduxEncoder,
    ) -> list[mx.array]:  # fmt:off
        image_embeds_list = []
        for image_path in image_paths:
            image_embeds = ReduxUtil._embed_single_image(
                image_path=image_path,
                image_encoder=image_encoder,
                image_embedder=image_embedder,
            )
            image_embeds_list.append(image_embeds)
        return image_embeds_list

    @staticmethod
    def _embed_single_image(
        image_path: str,
        image_encoder: SiglipVisionTransformer,
        image_embedder: ReduxEncoder,
    ) -> mx.array:  # fmt:off
        image = ImageUtil.load_image(image_path).convert("RGB")
        image = ImageUtil.preprocess_for_model(image=image)
        image_latents, pooler_output = image_encoder(image)
        image_embeds = image_embedder(image_latents)
        return image_embeds
