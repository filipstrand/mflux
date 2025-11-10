"""
Qwen Vision-Language Tokenizer for Image Edit Tasks

This module provides vision-language tokenization for Qwen Image Edit,
using Qwen2_5_VLProcessor to handle both text and image inputs.
"""

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

            # Debug checkpoint: Check formatted text before tokenization
            from mflux_debugger.semantic_checkpoint import debug_checkpoint

            debug_checkpoint(
                "mlx_formatted_text_before_tokenization",
                skip=True,
                verified=True,
                formatted_text=formatted_text,
                base_img_prompt=base_img_prompt,
                prompt=prompt,
            )
        else:
            # Regular Edit format: Vision tokens already in template
            # Just format with user prompt directly
            formatted_text = self.edit_template.format(prompt)

            # Debug checkpoint: Check formatted text before tokenization
            from mflux_debugger.semantic_checkpoint import debug_checkpoint

            debug_checkpoint(
                "mlx_formatted_text_before_tokenization",
                skip=True,
                verified=True,
                formatted_text=formatted_text,
                base_img_prompt="",
                prompt=prompt,
            )

        # Process images: convert to PIL Images and resize to CONDITION_IMAGE_SIZE
        import math

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

            if len(images) == 1:
                print(
                    f"🔎 VLTokenizer: Resized image from {img_w}x{img_h} to {condition_width}x{condition_height} (CONDITION_IMAGE_SIZE={CONDITION_IMAGE_SIZE})"
                )
            else:
                print(
                    f"🔎 VLTokenizer: Resized image {len(processed_images)}/{len(images)} from {img_w}x{img_h} to {condition_width}x{condition_height}"
                )

        print(f"🔎 VLTokenizer: Processing {len(processed_images)} image(s) with HF processor")

        # Use HF processor for both text and images to get exact same format as Diffusers
        # Processor accepts list of images and handles multiple images correctly
        model_inputs = self.processor(
            text=[formatted_text],
            images=processed_images,  # Pass list of resized images (matches PyTorch)
            padding=True,
            return_tensors="pt",
        )

        # CRITICAL: For Edit Plus models with multiple images, DO NOT truncate
        # PyTorch doesn't manually truncate - it lets the processor handle it based on max_length
        # The processor has its own max_length (typically 1024 or higher), so multi-image prompts can be longer
        # For single-image Edit Plus, we previously observed 314 tokens, but for multi-image it can be much longer
        # We must match PyTorch exactly: no manual truncation, let processor handle it
        # The processor will automatically truncate if needed based on its internal max_length setting
        original_length = model_inputs.input_ids.shape[1]
        if self.use_picture_prefix:
            # For Edit Plus: Don't truncate - match PyTorch's behavior exactly
            # PyTorch doesn't truncate manually, so we shouldn't either
            print(
                f"🔎 VLTokenizer: Edit Plus model - using full sequence length: {original_length} tokens (no truncation)"
            )
        else:
            # For regular Edit model: Keep the truncation logic if needed
            # But for now, also don't truncate to match PyTorch
            print(f"🔎 VLTokenizer: Regular Edit model - using full sequence length: {original_length} tokens")

        # Store dimensions that HF processor used (for EditUtil consistency)
        # Calculate from grid_thw: each grid cell is patch_size × merge_size = 14 × 2 = 28 pixels
        grid_thw = model_inputs.image_grid_thw[0]  # [t, h, w]
        factor = 14 * 2  # patch_size * merge_size
        self._vl_image_width = int(grid_thw[2]) * factor
        self._vl_image_height = int(grid_thw[1]) * factor

        print("🔎 VLTokenizer: HF processor output:")
        print(f"🔎 VLTokenizer:   pixel_values: {model_inputs.pixel_values.shape}")
        print(f"🔎 VLTokenizer:   image_grid_thw: {model_inputs.image_grid_thw}")

        # HF processor has already inserted image tokens correctly!
        # Just convert PyTorch tensors to MLX arrays

        input_ids = mx.array(model_inputs.input_ids.numpy())
        attention_mask = mx.array(model_inputs.attention_mask.numpy())
        pixel_values = mx.array(model_inputs.pixel_values.numpy())
        image_grid_thw = mx.array(model_inputs.image_grid_thw.numpy())

        print("🔎 VLTokenizer: Converted to MLX:")
        print(f"🔎 VLTokenizer:   input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}")
        print(f"🔎 VLTokenizer:   pixel_values.shape={pixel_values.shape}")
        print(f"🔎 VLTokenizer:   image_grid_thw={image_grid_thw}")

        # Checkpoint: Analyze token distribution for multiple images
        if len(images) > 1:
            from mflux_debugger.semantic_checkpoint import debug_checkpoint

            image_token_id = 151655  # <|image_pad|> token
            image_positions = input_ids == image_token_id
            image_positions_list = image_positions.flatten().tolist()
            image_token_indices = [i for i, is_img in enumerate(image_positions_list) if is_img]

            # Try to find boundaries between Picture 1 and Picture 2
            # Look for "Picture" token (if available) or estimate based on token count
            debug_checkpoint(
                "mlx_tokenization_analysis",
                {
                    "num_images": len(images),
                    "total_tokens": input_ids.shape[1],
                    "image_token_indices": image_token_indices[:50]
                    if len(image_token_indices) > 50
                    else image_token_indices,
                    "num_image_tokens": len(image_token_indices),
                    "image_grid_thw": image_grid_thw.tolist(),
                },
            )

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
