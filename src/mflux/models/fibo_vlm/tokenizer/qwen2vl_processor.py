from typing import Optional, Union

import numpy as np
from PIL import Image

from mflux.models.fibo_vlm.tokenizer.qwen2vl_image_processor import Qwen2VLImageProcessor


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

            # If images are provided, replace <|image_pad|> tokens with actual image tokens
            # The number of image tokens is calculated as: prod(image_grid_thw) // merge_size^2
            if images is not None and "input_ids" in text_inputs:
                image_token_id = 151655  # <|image_pad|> token ID
                input_ids = text_inputs["input_ids"]

                # Calculate number of image tokens per image based on image_grid_thw and merge_size
                # This matches transformers behavior: num_tokens = prod(image_grid_thw) // merge_size^2
                merge_size = self.image_processor.merge_size
                merge_length = merge_size**2
                num_images = len(images) if isinstance(images, list) else 1

                # Get image_grid_thw from result (already computed above)
                image_grid_thw = result["image_grid_thw"]

                # Calculate tokens per image
                if return_tensors == "pt":
                    import torch

                    if isinstance(image_grid_thw, torch.Tensor):
                        image_grid_thw_np = image_grid_thw.cpu().numpy()
                    else:
                        image_grid_thw_np = image_grid_thw
                else:
                    if isinstance(image_grid_thw, np.ndarray):
                        image_grid_thw_np = image_grid_thw
                    else:
                        image_grid_thw_np = np.array(image_grid_thw)

                # Calculate tokens for each image
                num_image_tokens_per_image_list = []
                for i in range(num_images):
                    if i < len(image_grid_thw_np):
                        grid = image_grid_thw_np[i]
                        num_tokens = int(np.prod(grid)) // merge_length
                        num_image_tokens_per_image_list.append(num_tokens)
                    else:
                        num_image_tokens_per_image_list.append(256)  # fallback

                # Replace image pad tokens with actual image tokens
                if return_tensors == "pt":
                    import torch

                    new_input_ids_list = []
                    for seq in input_ids:
                        new_seq = []
                        image_idx = 0
                        for token_id in seq:
                            if token_id == image_token_id and image_idx < num_images:
                                # Replace with calculated number of image tokens for this image
                                num_tokens = (
                                    num_image_tokens_per_image_list[image_idx]
                                    if image_idx < len(num_image_tokens_per_image_list)
                                    else 256
                                )
                                new_seq.extend([image_token_id] * num_tokens)
                                image_idx += 1
                            else:
                                new_seq.append(token_id)
                        new_input_ids_list.append(new_seq)

                    # Pad sequences to same length
                    max_len = max(len(seq) for seq in new_input_ids_list)
                    padded_input_ids = []
                    padded_attention_mask = []
                    for seq in new_input_ids_list:
                        pad_len = max_len - len(seq)
                        padded_seq = (
                            seq + [self.tokenizer.pad_token_id] * pad_len
                            if self.tokenizer.pad_token_id is not None
                            else seq + [0] * pad_len
                        )
                        padded_input_ids.append(padded_seq)
                        padded_attention_mask.append([1] * len(seq) + [0] * pad_len)

                    text_inputs["input_ids"] = torch.tensor(padded_input_ids)
                    if "attention_mask" in text_inputs:
                        text_inputs["attention_mask"] = torch.tensor(padded_attention_mask)
                else:
                    # NumPy version
                    new_input_ids_list = []
                    for seq in input_ids:
                        new_seq = []
                        image_idx = 0
                        for token_id in seq:
                            if token_id == image_token_id and image_idx < num_images:
                                # Replace with calculated number of image tokens for this image
                                num_tokens = (
                                    num_image_tokens_per_image_list[image_idx]
                                    if image_idx < len(num_image_tokens_per_image_list)
                                    else 256
                                )
                                new_seq.extend([image_token_id] * num_tokens)
                                image_idx += 1
                            else:
                                new_seq.append(token_id)
                        new_input_ids_list.append(new_seq)

                    # Pad sequences to same length
                    max_len = max(len(seq) for seq in new_input_ids_list)
                    padded_input_ids = []
                    padded_attention_mask = []
                    for seq in new_input_ids_list:
                        pad_len = max_len - len(seq)
                        padded_seq = (
                            seq
                            + [self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0] * pad_len
                        )
                        padded_input_ids.append(padded_seq)
                        padded_attention_mask.append([1] * len(seq) + [0] * pad_len)

                    text_inputs["input_ids"] = np.array(padded_input_ids)
                    if "attention_mask" in text_inputs:
                        text_inputs["attention_mask"] = np.array(padded_attention_mask)

            if return_tensors == "pt":
                result["input_ids"] = text_inputs["input_ids"]
                result["attention_mask"] = text_inputs.get("attention_mask")
            else:
                result["input_ids"] = np.array(text_inputs["input_ids"])
                if "attention_mask" in text_inputs:
                    result["attention_mask"] = np.array(text_inputs["attention_mask"])

        return result
