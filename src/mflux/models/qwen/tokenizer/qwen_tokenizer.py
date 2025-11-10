import mlx.core as mx
from transformers import Qwen2Tokenizer


class TokenizerQwen:
    def __init__(self, tokenizer: Qwen2Tokenizer, max_length: int = 1024):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_template = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.template_start_idx = 34

    def tokenize(self, prompt: str) -> tuple[mx.array, mx.array]:
        formatted_text = self.prompt_template.format(prompt)
        tokens = self.tokenizer(
            formatted_text,
            max_length=self.max_length + self.template_start_idx,
            padding=False,
            truncation=True,
            return_tensors="np",
        )
        input_ids = mx.array(tokens["input_ids"])
        attention_mask = mx.array(tokens["attention_mask"])
        return input_ids, attention_mask
