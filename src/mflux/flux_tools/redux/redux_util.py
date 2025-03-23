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
        # 0. load the PIL image
        image = ImageUtil.load_image(image_path).convert("RGB")

        # 1. Preprocess image
        image = ReduxUtil._preprocess(image)

        # 2. Result
        image_latents, pooler_output = image_encoder(image)

        # 3. Embed
        image_embeds = image_embedder(image_latents)

        return image_embeds

    @staticmethod
    def join_embeddings(
        prompt_embeds_txt: mx.array,
        pooled_prompt_embeds_txt: mx.array,
        image_embeds: mx.array,
        prompt_embeds_scale: int = 1.0,
        pooled_prompt_embeds_scale: int = 1.0,
    ) -> (mx.array, mx.array):
        # 1. Concatenate image and text embeddings
        prompt_embeds = mx.concatenate([prompt_embeds_txt, image_embeds], axis=1)

        # 2. Scale embeddings
        prompt_embeds_scale_tensor = mx.array(prompt_embeds_scale)
        prompt_embeds = prompt_embeds * prompt_embeds_scale_tensor[:, None, None]

        # 3. Weighted sum
        prompt_embeds = mx.sum(prompt_embeds, axis=0, keepdims=True)

        # 4. Scale pooled embeddings
        pooled_prompt_embeds_scale_tensor = mx.array(pooled_prompt_embeds_scale)
        pooled_prompt_embeds_txt = pooled_prompt_embeds_txt * pooled_prompt_embeds_scale_tensor[:, None]

        # 5. Weighted sum for pooled embeddings
        pooled_prompt_embeds = mx.sum(pooled_prompt_embeds_txt, axis=0, keepdims=True)

        return prompt_embeds, pooled_prompt_embeds

    @staticmethod
    def _preprocess(image: Image.Image) -> mx.array:
        # Define constants
        rescale_factor = 1 / 255.0
        image_mean = [0.5, 0.5, 0.5]
        image_std = [0.5, 0.5, 0.5]

        # Resize image
        image = image.resize((384, 384), resample=3)

        # Convert to numpy array
        image_np = np.array(image)

        # Rescale pixel values
        image_np = image_np.astype(np.float64) * rescale_factor

        # Normalize
        mean = np.array(image_mean)
        std = np.array(image_std)
        image_np = (image_np - mean) / std

        # Convert to MLX array and ensure (batch, channels, height, width)
        image_mx = mx.array(image_np.transpose(2, 0, 1))
        image_mx = mx.expand_dims(image_mx, axis=0)

        return image_mx
