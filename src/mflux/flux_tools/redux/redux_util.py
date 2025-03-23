import mlx.core as mx
import numpy as np
import PIL
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
        # Resize to expected dimensions (384x384 for Siglip)
        target_size = (384, 384)
        image = image.resize(target_size, resample=PIL.Image.LANCZOS)

        # Convert to numpy array in range [0, 1]
        image_np = np.array(image).astype(np.float32) / 255.0

        # Standard Siglip normalization with ImageNet mean and std
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_np = (image_np - mean) / std

        # Convert to mx.array and ensure BCHW format (batch, channels, height, width)
        image_mx = mx.array(image_np.transpose(2, 0, 1))  # CHW format
        image_mx = mx.expand_dims(image_mx, axis=0)  # Add batch dimension

        return image_mx
