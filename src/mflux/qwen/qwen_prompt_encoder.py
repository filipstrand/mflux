import mlx.core as mx

from mflux.models.text_encoder.qwen_text_encoder import QwenTextEncoder
from mflux.tokenizer.qwen_tokenizer import TokenizerQwen


class QwenPromptEncoder:
    """
    Qwen Prompt Encoder for the MFLUX pipeline.

    This class handles the text encoding process for Qwen models,
    following the same pattern as the existing PromptEncoder but
    adapted for Qwen's specific requirements.
    """

    @staticmethod
    def encode_prompt(
        prompt: str,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        qwen_tokenizer: TokenizerQwen,
        qwen_text_encoder: QwenTextEncoder,
    ) -> tuple[mx.array, mx.array]:
        """
        Encode a text prompt using Qwen text encoder.

        Args:
            prompt: Text prompt to encode
            prompt_cache: Cache for encoded prompts
            qwen_tokenizer: Qwen tokenizer instance
            qwen_text_encoder: Qwen text encoder instance

        Returns:
            tuple: (prompt_embeds, prompt_attention_mask)
                - prompt_embeds: Text embeddings [batch_size, seq_len, 3584]
                - prompt_attention_mask: Attention mask [batch_size, seq_len]
        """
        # Return cached result if available
        if prompt in prompt_cache:
            return prompt_cache[prompt]

        # Tokenize the prompt with Qwen template formatting
        input_ids, attention_mask = qwen_tokenizer.tokenize(prompt)

        # Encode with template prefix extraction
        prompt_embeds, encoder_attention_mask = qwen_text_encoder.encode_with_mask_extraction(
            input_ids=input_ids,
            attention_mask=attention_mask,
            template_start_idx=qwen_tokenizer.get_template_start_idx(),
        )

        # Cache the result
        prompt_cache[prompt] = (prompt_embeds, encoder_attention_mask)

        return prompt_embeds, encoder_attention_mask

    @staticmethod
    def encode_negative_prompt(
        negative_prompt: str | None,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        qwen_tokenizer: TokenizerQwen,
        qwen_text_encoder: QwenTextEncoder,
    ) -> tuple[mx.array, mx.array]:
        """
        Encode a negative prompt for classifier-free guidance.

        Args:
            negative_prompt: Negative text prompt (or None for empty prompt)
            prompt_cache: Cache for encoded prompts
            qwen_tokenizer: Qwen tokenizer instance
            qwen_text_encoder: Qwen text encoder instance

        Returns:
            tuple: (negative_prompt_embeds, negative_prompt_attention_mask)
        """
        if negative_prompt is None:
            negative_prompt = ""

        return QwenPromptEncoder.encode_prompt(
            prompt=negative_prompt,
            prompt_cache=prompt_cache,
            qwen_tokenizer=qwen_tokenizer,
            qwen_text_encoder=qwen_text_encoder,
        )
