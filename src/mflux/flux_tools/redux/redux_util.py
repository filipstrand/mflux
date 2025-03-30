import mlx.core as mx

from mflux import ImageUtil
from mflux.models.redux_encoder.redux_encoder import ReduxEncoder
from mflux.models.siglip_vision_transformer.siglip_vision_transformer import SiglipVisionTransformer


class ReduxUtil:
    @staticmethod
    def embed_image(
        image_path: str,
        image_encoder: SiglipVisionTransformer,
        image_embedder: ReduxEncoder,
    ) -> mx.array:  # fmt:off
        image = ImageUtil.load_image(image_path).convert("RGB")
        image = ImageUtil.preprocess_for_model(image=image)
        image_latents, pooler_output = image_encoder(image)
        image_embeds = image_embedder(image_latents)
        return image_embeds
