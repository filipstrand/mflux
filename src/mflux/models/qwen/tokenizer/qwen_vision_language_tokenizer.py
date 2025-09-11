"""
Qwen Vision-Language Tokenizer for Image Edit Tasks

This module provides vision-language tokenization for Qwen Image Edit,
using Qwen2VLProcessor to handle both text and image inputs.
"""

import os
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import Qwen2VLProcessor


class QwenVisionLanguageTokenizer:
    """
    Vision-Language tokenizer that processes both text and image inputs
    for Qwen Image Edit tasks.
    """

    def __init__(self, processor: Qwen2VLProcessor, max_length: int = 1024):
        self.processor = processor
        self.max_length = max_length

        # Edit-specific prompt template with vision tokens
        self.edit_template = (
            "<|im_start|>system\n"
            "Describe the key features of the input image (color, shape, size, texture, objects, background), "
            "then explain how the user's text instruction should alter or modify the image. "
            "Generate a new image that meets the user's requirements while maintaining consistency "
            "with the original input where appropriate.<|im_end|>\n"
            "<|im_start|>user\n"
            "<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        self.edit_template_start_idx = 64

    def tokenize_with_image(
        self,
        prompt: str,
        image: Union[Image.Image, np.ndarray, str],
        vl_width: int | None = None,
        vl_height: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        print(f"🔎 VLTokenizer: tokenize_with_image called with prompt='{prompt[:50] if prompt else None}...'")  # Debug entry point
        """
        Tokenize text prompt with image input for vision-language processing.

        Args:
            prompt: Text prompt for editing instruction
            image: Input image (PIL Image, numpy array, or path string)
            vl_width, vl_height: Optional VL dimensions (calculated like Diffusers)

        Returns:
            Tuple of (input_ids, attention_mask, pixel_values, image_grid_thw)
        """
        # Format the prompt with the edit template
        formatted_text = self.edit_template.format(prompt)

        # Process image if it's a path string or Path object
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Use provided VL dimensions (calculated like Diffusers) or fallback to patch-aligned calculation
        img_w, img_h = image.size
        
        if vl_width is not None and vl_height is not None:
            # Use Diffusers-calculated dimensions
            calc_w = vl_width
            calc_h = vl_height
            print(f"🔎 VLTokenizer: Using provided VL dimensions {calc_w}x{calc_h} (Diffusers-calculated)")
        else:
            # Fallback: Match HuggingFace smart_resize behavior exactly
            # HF uses factor = patch_size * merge_size = 14 * 2 = 28
            # Then rounds to nearest multiple of factor (not floor/ceil)
            patch_size = 14
            merge_size = 2
            factor = patch_size * merge_size  # 28
            
            # Use round() like HF smart_resize (not floor/ceil)
            calc_w = round(img_w / factor) * factor
            calc_h = round(img_h / factor) * factor
            print(f"🔎 VLTokenizer: Using fallback patch-aligned dimensions {calc_w}x{calc_h}")
        
        # Store the calculated dimensions for EditUtil consistency
        self._vl_image_width = calc_w
        self._vl_image_height = calc_h

        if calc_w != img_w or calc_h != img_h:
            image = image.resize((calc_w, calc_h), Image.BICUBIC)
            print(f"🔎 VLTokenizer: resized image from {img_w}x{img_h} to {calc_w}x{calc_h} (patch-aligned)")
        else:
            print(f"🔎 VLTokenizer: image {img_w}x{img_h} already patch-aligned")

        # BYPASS HuggingFace image processor - it does internal vision processing with wrong config
        # Instead, do native preprocessing and use HF only for text tokenization
        
        # 1. Use HF processor for text tokenization only (no images)
        text_inputs = self.processor(
            text=[formatted_text],
            images=None,  # No images - bypass HF image processing
            padding=True,
            return_tensors="pt",
        )
        
        # 2. Do native image preprocessing to get raw pixel data for our vision transformer
        print(f"🔎 VLTokenizer: Using NATIVE image preprocessing (bypassing HF image processor)")
        
        # Preprocess image natively - exactly as our vision transformer expects
        from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import (
            preprocess_image_for_vision_transformer,
            extract_image_patches,
        )
        
        # Preprocess image to get raw pixel tensor
        image_tensor = preprocess_image_for_vision_transformer(image, (calc_w, calc_h))
        
        # Extract patches to get 5D tensor [num_patches, C, T, H, W] 
        pixel_values_native, image_grid_thw_native = extract_image_patches(image_tensor)
        
        print(f"🔎 VLTokenizer: Native preprocessing complete:")
        print(f"🔎 VLTokenizer:   pixel_values: {pixel_values_native.shape} (5D raw patches)")
        print(f"🔎 VLTokenizer:   image_grid_thw: {image_grid_thw_native}")
        
        # 3. Insert image tokens into text tokenization
        # We need to replace the <|image_pad|> placeholder with the correct number of image tokens
        input_ids = text_inputs["input_ids"].clone()
        attention_mask = text_inputs["attention_mask"].clone()
        
        # Calculate number of image tokens needed (no spatial merging, like diffusers)
        t, h, w = image_grid_thw_native[0]
        num_image_tokens = int(h * w)  # No spatial merging - each patch becomes one token
        print(f"🔎 VLTokenizer: Replacing <|image_pad|> with {num_image_tokens} image tokens for grid {h}x{w}")
        
        # Find the <|image_pad|> token (151655) and replace it with multiple image tokens
        import torch
        image_token_id = 151655  # <|image_pad|> token
        
        # Find where <|image_pad|> appears in the sequence
        image_pad_positions = (input_ids == image_token_id).nonzero(as_tuple=True)
        
        if len(image_pad_positions[1]) > 0:
            # Replace the first <|image_pad|> token with multiple image tokens
            pad_pos = image_pad_positions[1][0].item()
            print(f"🔎 VLTokenizer: Found <|image_pad|> at position {pad_pos}, replacing with {num_image_tokens} tokens")
            
            # Create the new sequence with multiple image tokens
            before_tokens = input_ids[:, :pad_pos]
            after_tokens = input_ids[:, pad_pos + 1:]  # Skip the original <|image_pad|> token
            image_tokens = torch.full((1, num_image_tokens), image_token_id, dtype=input_ids.dtype)
            
            # Concatenate: before + multiple_image_tokens + after
            input_ids = torch.cat([before_tokens, image_tokens, after_tokens], dim=1)
            
            # Update attention mask accordingly
            before_mask = attention_mask[:, :pad_pos]
            after_mask = attention_mask[:, pad_pos + 1:]
            image_mask = torch.ones((1, num_image_tokens), dtype=attention_mask.dtype)
            attention_mask = torch.cat([before_mask, image_mask, after_mask], dim=1)
        else:
            print(f"🔎 VLTokenizer: WARNING - No <|image_pad|> token found in sequence!")
            # Fallback: prepend image tokens
            image_tokens = torch.full((1, num_image_tokens), image_token_id, dtype=input_ids.dtype)
            input_ids = torch.cat([image_tokens, input_ids], dim=1)
            attention_mask = torch.cat([torch.ones_like(image_tokens), attention_mask], dim=1)
        
        # 4. Combine everything
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": np.array(pixel_values_native),  # Convert MLX to numpy for consistency
            "image_grid_thw": np.array(image_grid_thw_native),
        }

        # Convert PyTorch tensors to MLX arrays
        input_ids = mx.array(model_inputs["input_ids"].numpy())
        attention_mask = mx.array(model_inputs["attention_mask"].numpy())

        # Handle pixel values and image grid (already numpy arrays from native preprocessing)
        pixel_values = None
        image_grid_thw = None

        if "pixel_values" in model_inputs:
            pixel_values = mx.array(model_inputs["pixel_values"])  # Already numpy array

        if "image_grid_thw" in model_inputs:
            image_grid_thw = mx.array(model_inputs["image_grid_thw"])  # Already numpy array

        # Debug prints
        # Best-effort debug prints; avoid raising if shapes are missing
        if hasattr(input_ids, "shape") and hasattr(attention_mask, "shape"):
            print(f"🔎 VLTokenizer: input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}")
        if pixel_values is not None and hasattr(pixel_values, "shape"):
            print(f"🔎 VLTokenizer: pixel_values.shape={pixel_values.shape}")
        if image_grid_thw is not None:
            print(f"🔎 VLTokenizer: image_grid_thw={image_grid_thw}")

        return input_ids, attention_mask, pixel_values, image_grid_thw

    def tokenize_text_only(self, prompt: str) -> tuple[mx.array, mx.array]:
        """
        Fallback to text-only tokenization (for compatibility).

        Args:
            prompt: Text prompt

        Returns:
            Tuple of (input_ids, attention_mask)
        """
        # Use the regular text-only template
        text_template = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, quantity, text, "
            "spatial relationships of the objects and background:<|im_end|>\n"
            "<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        formatted_text = text_template.format(prompt)
        tokens = self.processor.tokenizer(
            formatted_text,
            max_length=self.max_length + 34,  # Original template start idx
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = mx.array(tokens["input_ids"].numpy())
        attention_mask = mx.array(tokens["attention_mask"].numpy())

        return input_ids, attention_mask
