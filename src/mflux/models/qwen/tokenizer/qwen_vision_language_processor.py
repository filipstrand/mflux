from typing import Optional, Union

import mlx.core as mx
import numpy as np
from PIL import Image

from mflux.models.qwen.tokenizer.qwen_image_processor import QwenImageProcessor


class QwenVisionLanguageProcessor:
    def __init__(
        self,
        tokenizer,
        image_processor: Optional[QwenImageProcessor] = None,
        image_token: str = "<|image_pad|>",
        video_token: str = "<|video_pad|>",
    ):
        self.tokenizer = tokenizer
        self.image_processor = image_processor or QwenImageProcessor()

        self.image_token = image_token
        self.video_token = video_token
        self.image_token_id = (
            tokenizer.image_token_id
            if hasattr(tokenizer, "image_token_id")
            else tokenizer.convert_tokens_to_ids(self.image_token)
        )
        self.video_token_id = (
            tokenizer.video_token_id
            if hasattr(tokenizer, "video_token_id")
            else tokenizer.convert_tokens_to_ids(self.video_token)
        )

    def __call__(
        self,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        text: Optional[Union[str, list[str]]] = None,
        padding: bool = True,
        return_tensors: Optional[str] = None,
    ) -> dict:
        image_inputs = {}
        if images is not None:
            pixel_values, image_grid_thw = self.image_processor.preprocess(images)
            image_inputs = {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }

        if text is not None:
            if not isinstance(text, list):
                text = [text]
            text = text.copy()

            if images is not None:
                merge_length = self.image_processor.merge_size**2
                index = 0
                for i in range(len(text)):
                    while self.image_token in text[i]:
                        if index < len(image_grid_thw):
                            num_image_tokens = int(np.prod(image_grid_thw[index])) // merge_length
                            text[i] = text[i].replace(
                                self.image_token,
                                "<|placeholder|>" * num_image_tokens,
                                1,
                            )
                            index += 1
                        else:
                            break
                    text[i] = text[i].replace("<|placeholder|>", self.image_token)

            text_inputs = self.tokenizer(
                text,
                padding=padding,
                return_tensors="pt" if return_tensors == "pt" else "np",
            )

            if return_tensors == "pt":
                import torch

                if isinstance(text_inputs["input_ids"], torch.Tensor):
                    input_ids = mx.array(text_inputs["input_ids"].numpy())
                    attention_mask = mx.array(text_inputs["attention_mask"].numpy())
                else:
                    input_ids = mx.array(text_inputs["input_ids"])
                    attention_mask = mx.array(text_inputs["attention_mask"])
            else:
                if isinstance(text_inputs["input_ids"], np.ndarray):
                    input_ids = mx.array(text_inputs["input_ids"])
                    attention_mask = mx.array(text_inputs["attention_mask"])
                else:
                    input_ids = mx.array(np.array(text_inputs["input_ids"]))
                    attention_mask = mx.array(np.array(text_inputs["attention_mask"]))
        else:
            input_ids = None
            attention_mask = None

        result = {**image_inputs}
        if input_ids is not None:
            result["input_ids"] = input_ids
        if attention_mask is not None:
            result["attention_mask"] = attention_mask

        return result
