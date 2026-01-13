"""Hunyuan-DiT Prompt Encoder.

Handles encoding prompts using dual text encoders (CLIP + T5/mT5).
Both encoders provide full sequence outputs for cross-attention.
"""

import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer


class HunyuanPromptEncoder:
    """Prompt encoder for Hunyuan-DiT using dual text encoders.

    Encodes prompts using:
    - Chinese CLIP (1024 dim, max 77 tokens)
    - mT5-XXL (2048 dim, max 256 tokens)

    Both provide full sequence outputs for cross-attention.
    """

    @staticmethod
    def encode_prompt(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        clip_tokenizer: Tokenizer,
        t5_tokenizer: Tokenizer,
        clip_encoder,  # HunyuanCLIPEncoder
        t5_encoder,    # HunyuanT5Encoder
        max_clip_length: int = 77,
        max_t5_length: int = 256,
    ) -> tuple[mx.array, mx.array]:
        """
        Encode a prompt using dual text encoders.

        Args:
            prompt: Text prompt to encode
            prompt_cache: Cache for encoded prompts
            clip_tokenizer: CLIP tokenizer
            t5_tokenizer: T5/mT5 tokenizer
            clip_encoder: CLIP text encoder model
            t5_encoder: T5/mT5 text encoder model
            max_clip_length: Maximum CLIP sequence length (77)
            max_t5_length: Maximum T5 sequence length (256)

        Returns:
            Tuple of (clip_embeds [batch, 77, 1024], t5_embeds [batch, 256, 2048])
        """
        # Return cached encodings if available
        if prompt in prompt_cache:
            return prompt_cache[prompt]

        # Tokenize with both tokenizers
        clip_tokens = clip_tokenizer.tokenize(prompt, max_length=max_clip_length)
        t5_tokens = t5_tokenizer.tokenize(prompt, max_length=max_t5_length)

        # Encode with both models
        clip_embeds = clip_encoder(clip_tokens.input_ids)  # [batch, 77, 1024]
        t5_embeds = t5_encoder(t5_tokens.input_ids)        # [batch, 256, 2048]

        # Cache and return
        prompt_cache[prompt] = (clip_embeds, t5_embeds)
        return clip_embeds, t5_embeds

    @staticmethod
    def get_attention_masks(
        clip_tokens,
        t5_tokens,
    ) -> tuple[mx.array, mx.array]:
        """
        Get attention masks for both encoders.

        Args:
            clip_tokens: CLIP tokenizer output
            t5_tokens: T5 tokenizer output

        Returns:
            Tuple of (clip_mask, t5_mask) where 1 = valid, 0 = padding
        """
        clip_mask = clip_tokens.attention_mask if hasattr(clip_tokens, 'attention_mask') else None
        t5_mask = t5_tokens.attention_mask if hasattr(t5_tokens, 'attention_mask') else None
        return clip_mask, t5_mask
