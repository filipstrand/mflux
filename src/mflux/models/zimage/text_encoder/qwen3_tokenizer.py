from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer


class Qwen3Tokenizer:
    """Tokenizer wrapper for Qwen3-4B.

    Uses HuggingFace tokenizers for compatibility with existing mflux patterns.
    """

    MAX_LENGTH = 256  # Default max tokens for prompts
    PAD_TOKEN_ID = 0

    def __init__(self, tokenizer_path: str | Path | None = None):
        """Initialize tokenizer.

        Args:
            tokenizer_path: Path to tokenizer files or HF model ID.
                           If None, uses default Z-Image model.
        """
        if tokenizer_path is None:
            # Default to HuggingFace model
            tokenizer_path = "Tongyi-MAI/Z-Image-Turbo"

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            subfolder="tokenizer",
            trust_remote_code=True,
        )

        # Set padding side
        self.tokenizer.padding_side = "right"

    def __call__(
        self,
        text: str | list[str],
        max_length: int | None = None,
        padding: bool = True,
        truncation: bool = True,
        apply_chat_template: bool = True,
    ) -> dict[str, mx.array]:
        """Tokenize text input.

        Args:
            text: Single string or list of strings
            max_length: Maximum sequence length (default: MAX_LENGTH)
            padding: Whether to pad to max_length
            truncation: Whether to truncate to max_length
            apply_chat_template: Whether to apply Qwen3 chat template (like diffusers)

        Returns:
            Dict with 'input_ids' and 'attention_mask' as mx.arrays
        """
        max_length = max_length or self.MAX_LENGTH

        # Handle single string
        if isinstance(text, str):
            text = [text]

        # Apply chat template if requested (matches diffusers behavior)
        if apply_chat_template:
            formatted_text = []
            for prompt_item in text:
                messages = [{"role": "user", "content": prompt_item}]
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                formatted_text.append(formatted)
            text = formatted_text

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length" if padding else False,
            truncation=truncation,
            return_tensors="np",
        )

        # Convert to MLX arrays
        return {
            "input_ids": mx.array(encoded["input_ids"]),
            "attention_mask": mx.array(encoded["attention_mask"]),
        }

    def decode(self, token_ids: mx.array | list) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: Token IDs to decode

        Returns:
            Decoded text string
        """
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.tokenizer)

    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.tokenizer.pad_token_id or self.PAD_TOKEN_ID

    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.tokenizer.eos_token_id
