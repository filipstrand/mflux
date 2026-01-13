"""Prompt encoder for FLUX.2 using Mistral3.

Unlike FLUX.1 which uses CLIP + T5, FLUX.2 uses a single Mistral3 encoder
that provides both sequence embeddings and pooled embeddings.
"""

import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.flux2.model.flux2_text_encoder.mistral3_encoder import Mistral3TextEncoder


class Flux2PromptEncoder:
    """Encodes prompts using Mistral3 for FLUX.2."""

    @staticmethod
    def encode_prompt(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        mistral3_tokenizer: Tokenizer,
        text_encoder: Mistral3TextEncoder,
    ) -> tuple[mx.array, mx.array]:
        """Encode a text prompt for FLUX.2 generation.

        Args:
            prompt: The text prompt to encode
            prompt_cache: Cache for storing encoded prompts
            mistral3_tokenizer: Mistral3 tokenizer
            text_encoder: Mistral3 text encoder

        Returns:
            Tuple of:
            - prompt_embeds: Sequence embeddings [1, seq_len, 15360]
            - pooled_prompt_embeds: Pooled embeddings [1, 5120]
        """
        # Return cached encodings if available
        if prompt in prompt_cache:
            return prompt_cache[prompt]

        # Tokenize the prompt
        tokenized = mistral3_tokenizer.tokenize(prompt)

        # Encode using Mistral3
        prompt_embeds, pooled_prompt_embeds = text_encoder(
            input_ids=tokenized.input_ids,
            attention_mask=tokenized.attention_mask,
        )

        # Cache the results
        prompt_cache[prompt] = (prompt_embeds, pooled_prompt_embeds)

        return prompt_embeds, pooled_prompt_embeds
