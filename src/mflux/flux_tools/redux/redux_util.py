import mlx.core as mx
import numpy as np
from PIL import Image

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
        image = ReduxUtil._preprocess(image)
        image_latents, pooler_output = image_encoder(image)
        image_embeds = image_embedder(image_latents)
        return image_embeds

    @staticmethod
    def _preprocess(image: Image.Image) -> mx.array:
        # Define constants
        rescale_factor = 1 / 255.0
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]

        # Resize, scale, normalize
        image = image.resize((384, 384), resample=3)
        image_np = np.array(image)
        image_np = image_np.astype(np.float64) * rescale_factor
        mean = np.array(image_mean)
        std = np.array(image_std)
        image_np = (image_np - mean) / std
        image_mx = mx.array(image_np.transpose(2, 0, 1))
        image_mx = mx.expand_dims(image_mx, axis=0)
        return image_mx
