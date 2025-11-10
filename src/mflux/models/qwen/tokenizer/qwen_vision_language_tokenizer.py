"""
Qwen Vision-Language Tokenizer for Image Edit Tasks

This module provides vision-language tokenization for Qwen Image Edit,
using Qwen2_5_VLProcessor to handle both text and image inputs.
"""

import math
from pathlib import Path
from typing import Union

import mlx.core as mx
import numpy as np
from PIL import Image
from transformers import Qwen2_5_VLProcessor


class QwenVisionLanguageTokenizer:
    """
    Vision-Language tokenizer that processes both text and image inputs
    for Qwen Image Edit tasks.
    """

    def __init__(
        self,
        processor: Qwen2_5_VLProcessor,
        max_length: int = 1024,
        use_picture_prefix: bool = True,
    ):
        """
        Initialize Qwen Vision-Language Tokenizer.

        Args:
            processor: Qwen2_5_VLProcessor instance
            max_length: Maximum sequence length
            use_picture_prefix: If True, adds "Picture N:" prefix (Edit Plus format).
                              If False, uses regular Edit format (vision tokens in template).
        """
        self.processor = processor
        self.max_length = max_length
        self.use_picture_prefix = use_picture_prefix

        # Edit-specific prompt template
        # Edit Plus: template.format("Picture 1: <vision_tokens>" + user_prompt)
        # Regular Edit: template.format(user_prompt) where template includes vision tokens
        if use_picture_prefix:
            # Edit Plus format: empty placeholder, adds Picture N: dynamically
            self.edit_template = (
                "<|im_start|>system\n"
                "Describe the key features of the input image (color, shape, size, texture, objects, background), "
                "then explain how the user's text instruction should alter or modify the image. "
                "Generate a new image that meets the user's requirements while maintaining consistency "
                "with the original input where appropriate.<|im_end|>\n"
                "<|im_start|>user\n"
                "{}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
        else:
            # Regular Edit format: vision tokens in template
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
        image: Union[Image.Image, np.ndarray, str, list],
        vl_width: int | None = None,
        vl_height: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        print(
            f"🔎 VLTokenizer: tokenize_with_image called with prompt='{prompt[:50] if prompt else None}...'"
        )  # Debug entry point
        """
        Tokenize text prompt with image input for vision-language processing.

        Args:
            prompt: Text prompt for editing instruction
            image: Input image(s) - PIL Image, numpy array, path string, or list of any of these
            vl_width, vl_height: Optional VL dimensions (calculated like Diffusers)

        Returns:
            Tuple of (input_ids, attention_mask, pixel_values, image_grid_thw)
        """
        # Normalize image to list format
        if not isinstance(image, list):
            images = [image]
        else:
            images = image

        # Format prompt based on tokenizer mode
        if self.use_picture_prefix:
            # Edit Plus format: Add "Picture N:" prefix for each image
            # Match PyTorch Plus pipeline behavior exactly:
            # For multiple images: "Picture 1: ... Picture 2: ... Picture N: ..."
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            base_img_prompt = ""
            for i in range(len(images)):
                base_img_prompt += img_prompt_template.format(i + 1)
            formatted_text = self.edit_template.format(base_img_prompt + prompt)
        else:
            # Regular Edit format: Vision tokens already in template
            # Just format with user prompt directly
            formatted_text = self.edit_template.format(prompt)

        # Process images: convert to PIL Images and resize to CONDITION_IMAGE_SIZE
        CONDITION_IMAGE_SIZE = 384 * 384  # Match PyTorch's CONDITION_IMAGE_SIZE

        processed_images = []
        for img in images:
            # Convert to PIL Image
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif not isinstance(img, Image.Image):
                raise ValueError(f"Unsupported image type: {type(img)}")

            # Resize to CONDITION_IMAGE_SIZE (384×384) maintaining aspect ratio
            img_w, img_h = img.size
            ratio = img_w / img_h
            condition_width = math.sqrt(CONDITION_IMAGE_SIZE * ratio)
            condition_height = condition_width / ratio
            condition_width = round(condition_width / 32) * 32
            condition_height = round(condition_height / 32) * 32

            img = img.resize((int(condition_width), int(condition_height)), Image.BICUBIC)
            processed_images.append(img)

        # Use HF processor for both text and images to get exact same format as Diffusers
        # Processor accepts list of images and handles multiple images correctly
        model_inputs = self.processor(
            text=[formatted_text],
            images=processed_images,  # Pass list of resized images (matches PyTorch)
            padding=True,
            return_tensors="pt",
        )

        # Store dimensions that HF processor used (for EditUtil consistency)
        # Calculate from grid_thw: each grid cell is patch_size × merge_size = 14 × 2 = 28 pixels
        grid_thw = model_inputs.image_grid_thw[0]  # [t, h, w]
        factor = 14 * 2  # patch_size * merge_size
        self._vl_image_width = int(grid_thw[2]) * factor
        self._vl_image_height = int(grid_thw[1]) * factor

        # HF processor has already inserted image tokens correctly!
        # Just convert PyTorch tensors to MLX arrays
        input_ids = mx.array(model_inputs.input_ids.numpy())
        attention_mask = mx.array(model_inputs.attention_mask.numpy())
        pixel_values = mx.array(model_inputs.pixel_values.numpy())
        image_grid_thw = mx.array(model_inputs.image_grid_thw.numpy())

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
