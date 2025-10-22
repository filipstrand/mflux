"""
Vision-Language encoder for Qwen Image Edit - Integrated approach matching Diffusers.

This module provides a simplified VL encoder that mirrors the Diffusers reference
implementation using the integrated Qwen2_5_VLForConditionalGeneration approach.
"""

import os

import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_text_encoder.qwen_encoder import QwenEncoder


class QwenVisionLanguageEncoder(nn.Module):
    """
    Vision-Language encoder for Qwen Image Edit - integrated approach matching Diffusers.

    This is essentially a wrapper around the text encoder that handles vision-language
    inputs in the same way as Qwen2_5_VLForConditionalGeneration in Diffusers.
    """

    def __init__(self, encoder=None):
        super().__init__()
        self.encoder = encoder or QwenEncoder()
        self._hf_fallback_enabled = os.getenv("MFLUX_QWEN_VL_USE_HF", "0") in {"1", "true", "TRUE"}
        self._hf_model = None  # Lazy-loaded torch model when fallback is enabled

        # Edit-specific prompt template matching Diffusers exactly
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
        self.edit_template_start_idx = 64  # Number of tokens before the actual prompt

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
        precomputed_image_embeds: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        """
        Forward pass for vision-language encoding - integrated approach like Diffusers.

        Args:
            input_ids: Tokenized text+vision input (already processed by VL tokenizer)
            attention_mask: Attention mask for the combined input
            pixel_values: Pixel values (handled by integrated model weights)
            image_grid_thw: Image grid dimensions (handled by integrated model weights)

        Returns:
            Tuple of (prompt_embeds, encoder_attention_mask)
        """
        print(f"🔎 VLEncoder: input_ids.shape={input_ids.shape}, attention_mask.shape={attention_mask.shape}")
        if pixel_values is not None:
            print(f"🔎 VLEncoder: pixel_values.shape={pixel_values.shape}")
        if image_grid_thw is not None:
            print(f"🔎 VLEncoder: image_grid_thw={image_grid_thw}")

        # Fallback path: use local cached HF VL model to produce hidden states (no new downloads)
        if self._hf_fallback_enabled:
            try:
                import numpy as np
                import torch
                from huggingface_hub import snapshot_download
                from transformers import Qwen2_5_VLForConditionalGeneration

                if self._hf_model is None:
                    root = snapshot_download(
                        repo_id="Qwen/Qwen-Image-Edit",
                        local_files_only=True,
                        allow_patterns=[
                            "text_encoder/**",
                        ],
                    )
                    self._hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        root, subfolder="text_encoder", torch_dtype=torch.float32, local_files_only=True
                    )
                    self._hf_model.eval()

                def to_t(x: mx.array | None):
                    if x is None:
                        return None
                    return torch.from_numpy(np.array(x))

                with torch.no_grad():
                    out = self._hf_model(
                        input_ids=to_t(input_ids),
                        attention_mask=to_t(attention_mask),
                        pixel_values=to_t(pixel_values),
                        image_grid_thw=to_t(image_grid_thw),
                        output_hidden_states=True,
                    )
                hs_np = out.hidden_states[-1].cpu().numpy()
                hidden_states = mx.array(hs_np)
                print(f"🔎 VLEncoder[HF]: hidden_states.shape={hidden_states.shape}")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ VLEncoder: HF fallback failed ({e}), using integrated encoder")
                self._hf_fallback_enabled = False

        # Integrated path: our encoder handles VL fusion internally
        if not self._hf_fallback_enabled:
            # The integrated text encoder handles vision-language fusion internally
            # Matches Diffusers: text_encoder(input_ids, attention_mask, pixel_values, image_grid_thw)
            if precomputed_image_embeds is not None:
                # Pass precomputed image embeddings to bypass vision processing
                hidden_states = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=None,  # Skip vision processing
                    image_grid_thw=image_grid_thw,
                    precomputed_image_embeds=precomputed_image_embeds,
                )
            else:
                # Normal path with vision processing
                hidden_states = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                )
            print(f"🔎 VLEncoder: hidden_states.shape={hidden_states.shape}")

        from mflux_debugger.semantic_checkpoint import debug_checkpoint

        # Large tensors will be automatically serialized as previews (first 10 + last 10 values)
        debug_checkpoint(
            "mlx_after_encoder_raw",
            {
                "hidden_states": hidden_states,
                "attention_mask": attention_mask,
            },
        )

        # Mask-based extraction, then drop template tokens, then pad to batch max length
        # This mirrors the reference behavior used in Diffusers pipelines
        if attention_mask is None:
            # If mask is missing, fall back to full sequence
            trimmed = [hidden_states[i] for i in range(hidden_states.shape[0])]
        else:
            trimmed = []
            batch = hidden_states.shape[0]
            for i in range(batch):
                valid_len = int(mx.sum(attention_mask[i]).item())
                trimmed.append(hidden_states[i, :valid_len, :])

        # Drop the first 64 tokens after masking
        drop_idx = self.edit_template_start_idx
        trimmed_after_drop = [t[drop_idx:] if t.shape[0] > drop_idx else t for t in trimmed]
        debug_checkpoint(
            "mlx_after_extract_and_drop",
            {
                "trimmed_after_drop_0_shape": trimmed_after_drop[0].shape if len(trimmed_after_drop) > 0 else None,
                "trimmed_after_drop_0_preview": trimmed_after_drop[0][:5, :10].tolist()
                if len(trimmed_after_drop) > 0 and trimmed_after_drop[0].shape[0] > 0
                else None,
            },
        )
        trimmed = trimmed_after_drop

        # Pad sequences to max length across batch
        max_len = max(t.shape[0] for t in trimmed) if trimmed else 0
        hidden_dim = hidden_states.shape[2]
        padded_embeds = []
        padded_masks = []
        for t in trimmed:
            cur_len = t.shape[0]
            if cur_len < max_len:
                pad_e = mx.zeros((max_len - cur_len, hidden_dim), dtype=t.dtype)
                t_pad = mx.concatenate([t, pad_e], axis=0)
                pad_m = mx.concatenate(
                    [mx.ones(cur_len, dtype=mx.int32), mx.zeros(max_len - cur_len, dtype=mx.int32)], axis=0
                )
            else:
                t_pad = t
                pad_m = mx.ones(cur_len, dtype=mx.int32)
            padded_embeds.append(t_pad)
            padded_masks.append(pad_m)

        prompt_embeds = mx.stack(padded_embeds, axis=0) if padded_embeds else hidden_states
        encoder_attention_mask = mx.stack(padded_masks, axis=0) if padded_masks else attention_mask

        return prompt_embeds, encoder_attention_mask
