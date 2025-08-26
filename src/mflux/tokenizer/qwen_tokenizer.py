import mlx.core as mx
from transformers import Qwen2Tokenizer


class TokenizerQwen:
    def __init__(self, tokenizer: Qwen2Tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Qwen uses specific template formatting for image generation tasks
        self.prompt_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.template_start_idx = 34  # Number of tokens to drop from template prefix

    def tokenize(self, prompt: str) -> tuple[mx.array, mx.array]:
        """
        Tokenize prompt with Qwen template formatting.

        Returns:
            tuple: (input_ids, attention_mask) as MLX arrays
        """
        # Apply Qwen template formatting
        formatted_prompt = self.prompt_template.format(prompt)

        # Tokenize with padding and truncation (match HuggingFace pipeline exactly)
        result = self.tokenizer(
            [formatted_prompt],
            padding=True,  # Pad to longest in batch, not max_length
            max_length=self.max_length + self.template_start_idx,
            truncation=True,
            return_tensors="pt",  # Use PyTorch tensors first, convert to MLX
        )

        # Convert to MLX arrays
        input_ids = mx.array(result.input_ids.numpy())
        attention_mask = mx.array(result.attention_mask.numpy())

        return input_ids, attention_mask

    def get_template_start_idx(self) -> int:
        """Get the number of tokens to drop from the template prefix."""
        return self.template_start_idx
