import mlx.core as mx
from PIL import Image
from typing import Union, Optional
import numpy as np

from mflux.models.qwen.model.qwen_text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.models.qwen.model.qwen_text_encoder.qwen_vision_language_encoder import QwenVisionLanguageEncoder
from mflux.models.qwen.tokenizer.qwen_tokenizer import TokenizerQwen
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
        # 1. Encode positive prompt with image
        input_ids, attention_mask, pixel_values, image_grid_thw = qwen_vl_tokenizer.tokenize_with_image(
            prompt=prompt, 
            image=image
        )
        
        prompt_embeds, prompt_mask = qwen_vl_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        )
        
        # 2. Encode negative prompt with image (if provided)
        if negative_prompt is not None and negative_prompt != "":
            neg_input_ids, neg_attention_mask, neg_pixel_values, neg_image_grid_thw = qwen_vl_tokenizer.tokenize_with_image(
                prompt=negative_prompt,
                image=image
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
        # Text-only components
        qwen_tokenizer: Optional[TokenizerQwen] = None,
        qwen_text_encoder: Optional[QwenTextEncoder] = None,
        # Vision-language components  
        qwen_vl_tokenizer: Optional[QwenVisionLanguageTokenizer] = None,
        qwen_vl_encoder: Optional[QwenVisionLanguageEncoder] = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        """
        Automatically choose between text-only and vision-language encoding
        based on whether an image is provided.
        
        Args:
            prompt: Text prompt
            negative_prompt: Optional negative prompt
            image: Optional input image (if provided, uses vision-language encoding)
            prompt_cache: Cache for encoded prompts
            qwen_tokenizer: Text-only tokenizer (for backward compatibility)
            qwen_text_encoder: Text-only encoder (for backward compatibility)
            qwen_vl_tokenizer: Vision-language tokenizer (for edit tasks)
            qwen_vl_encoder: Vision-language encoder (for edit tasks)
            
        Returns:
            Tuple of (prompt_embeds, prompt_mask, neg_prompt_embeds, neg_prompt_mask)
        """
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
                raise ValueError("Text-only components (qwen_tokenizer, qwen_text_encoder) are required when no image is provided")
            
            return QwenVisionLanguagePromptEncoder.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt or "",
                prompt_cache=prompt_cache,
                qwen_tokenizer=qwen_tokenizer,
                qwen_text_encoder=qwen_text_encoder,
            )
