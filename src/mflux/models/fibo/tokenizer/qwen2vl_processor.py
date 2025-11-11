from typing import Optional, Union

import numpy as np
from PIL import Image

from mflux.models.fibo.tokenizer.qwen2vl_image_processor import Qwen2VLImageProcessor


class Qwen2VLProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.image_processor = Qwen2VLImageProcessor()

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        return_tensors: Optional[str] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            return_tensors=return_tensors,
            return_dict=return_dict,
            **kwargs,
        )
        if tokenize and return_dict:
            if return_tensors == "pt":
                import torch

                if isinstance(formatted, dict):
                    result = {}
                    for key, value in formatted.items():
                        if isinstance(value, torch.Tensor):
                            result[key] = value.numpy()
                        else:
                            result[key] = value
                    return result
            elif return_tensors == "np" or return_tensors is None:
                if isinstance(formatted, dict):
                    result = {}
                    for key, value in formatted.items():
                        if hasattr(value, "numpy"):
                            result[key] = value.numpy()
                        elif isinstance(value, (list, tuple)):
                            result[key] = np.array(value)
                        else:
                            result[key] = value
                    return result
        return formatted

    def __call__(
        self,
        text: Optional[Union[str, list[str]]] = None,
        images: Optional[Union[Image.Image, list[Image.Image]]] = None,
        padding: bool = True,
        return_tensors: Optional[str] = None,
        **kwargs,
    ):
        result = {}

        if images is not None:
            pixel_values, image_grid_thw = self.image_processor.preprocess(images)
            result["pixel_values"] = pixel_values
            result["image_grid_thw"] = image_grid_thw

        if text is not None:
            if not isinstance(text, list):
                text = [text]

            text_inputs = self.tokenizer(
                text,
                padding=padding,
                return_tensors=return_tensors or "np",
                **kwargs,
            )

            if return_tensors == "pt":
                result["input_ids"] = text_inputs["input_ids"]
                result["attention_mask"] = text_inputs.get("attention_mask")
            else:
                result["input_ids"] = np.array(text_inputs["input_ids"])
                if "attention_mask" in text_inputs:
                    result["attention_mask"] = np.array(text_inputs["attention_mask"])

        return result
