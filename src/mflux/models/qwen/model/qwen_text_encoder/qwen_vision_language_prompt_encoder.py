from typing import Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.qwen.model.qwen_text_encoder.qwen_prompt_encoder import QwenPromptEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
from mflux.models.qwen.tokenizer.qwen_vision_language_tokenizer import QwenVisionLanguageTokenizer


class QwenVisionLanguagePromptEncoder:
    @staticmethod
    def encode_prompt_with_image(
        prompt: str,
        image: Union[Image.Image, np.ndarray, str],
        negative_prompt: Optional[str] = None,
        prompt_cache: Optional[dict] = None,
        qwen_vl_tokenizer: Optional[QwenVisionLanguageTokenizer] = None,
        qwen_vl_encoder: Optional[QwenVisionLanguageEncoder] = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        if qwen_vl_tokenizer is None or qwen_vl_encoder is None:
            raise ValueError("qwen_vl_tokenizer and qwen_vl_encoder are required")

        input_ids, attention_mask, pixel_values, image_grid_thw = qwen_vl_tokenizer.tokenize_with_image(
            prompt=prompt, image=image
        )

        prompt_embeds, prompt_mask = qwen_vl_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )

        if negative_prompt is not None and negative_prompt != "":
            neg_input_ids, neg_attention_mask, neg_pixel_values, neg_image_grid_thw = (
                qwen_vl_tokenizer.tokenize_with_image(prompt=negative_prompt, image=image)
            )

            neg_prompt_embeds, neg_prompt_mask = qwen_vl_encoder(
                input_ids=neg_input_ids,
                attention_mask=neg_attention_mask,
                pixel_values=neg_pixel_values,
                image_grid_thw=neg_image_grid_thw,
            )
        else:
            # If no negative prompt provided, mirror shapes with zeros masks/embeds to disable CFG cleanly
            neg_prompt_embeds, neg_prompt_mask = prompt_embeds, prompt_mask

        # 3. Return the result including image_grid_thw for conditioning
        result = (prompt_embeds, prompt_mask, neg_prompt_embeds, neg_prompt_mask, image_grid_thw)
        return result

    @staticmethod
    def encode_prompt_auto(
        prompt: str,
        negative_prompt: Optional[str] = None,
        image: Optional[Union[Image.Image, np.ndarray, str]] = None,
        prompt_cache: Optional[dict] = None,
        qwen_tokenizer: Optional[Tokenizer] = None,
        qwen_text_encoder: Optional[QwenTextEncoder] = None,
        qwen_vl_tokenizer: Optional[QwenVisionLanguageTokenizer] = None,
        qwen_vl_encoder: Optional[QwenVisionLanguageEncoder] = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        if prompt_cache is None:
            prompt_cache = {}

        if image is not None and qwen_vl_tokenizer is not None and qwen_vl_encoder is not None:
            # Use vision-language encoding for edit tasks
            return QwenVisionLanguagePromptEncoder.encode_prompt_with_image(
                prompt=prompt,
                image=image,
                negative_prompt=negative_prompt,
                prompt_cache=prompt_cache,
                qwen_vl_tokenizer=qwen_vl_tokenizer,
                qwen_vl_encoder=qwen_vl_encoder,
            )
        else:
            # Fall back to text-only encoding for regular generation
            if qwen_tokenizer is None or qwen_text_encoder is None:
                raise ValueError(
                    "Text-only components (qwen_tokenizer, qwen_text_encoder) are required when no image is provided"
                )

            return QwenPromptEncoder.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                prompt_cache=prompt_cache,
                qwen_tokenizer=qwen_tokenizer,
                qwen_text_encoder=qwen_text_encoder,
            )
