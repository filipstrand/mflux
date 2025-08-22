import mlx.core as mx
from mlx import nn
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import Qwen2Tokenizer

from .utils import load_model, process_text_embeddings_mlx


class QwenTextEncoderAlternative(nn.Module):
    def __init__(
        self,
        model_path: str | Path | None = None,
        template_drop_idx: int = 34,
        tokenizer_max_length: int = 1024
    ):
        super().__init__()
        self.template_drop_idx = template_drop_idx
        self.tokenizer_max_length = tokenizer_max_length
        self.template = (
            "<|im_start|>system\n"
            "Describe the image by detailing the color, shape, size, texture, "
            "quantity, text, spatial relationships of the objects and background:"
            "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        )
        self.model_path = model_path
        self._qwen_text_encoder = None
        self._tokenizer = None
        
    def _initialize_if_needed(self):
        if self._qwen_text_encoder is None:
            if self.model_path:
                root_path = Path(self.model_path)
            else:
                # Default to downloading Qwen-Image model
                root_path = Path(
                    snapshot_download(
                        repo_id="Qwen/Qwen-Image",
                    )
                )
            self._qwen_text_encoder = load_model(root_path / "text_encoder")
            self._tokenizer = Qwen2Tokenizer.from_pretrained(root_path / "tokenizer")
    
    def _tokenize_and_process(self, prompt: str) -> tuple[mx.array, mx.array]:
        self._initialize_if_needed()
        
        # Format with diffusers template
        formatted_prompt = self.template.format(prompt)
        
        # Tokenize
        inputs = self._tokenizer(
            formatted_prompt,
            max_length=self.tokenizer_max_length + self.template_drop_idx,
            padding=True,
            truncation=True,
            return_tensors="mlx"
        )
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        # Get hidden states from the model
        hidden_states = self._qwen_text_encoder(input_ids)
        
        # Process embeddings using the diffusers-compatible function
        prompt_embeds, prompt_mask = process_text_embeddings_mlx(
            hidden_states=hidden_states, 
            attention_mask=attention_mask, 
            drop_idx=self.template_drop_idx, 
            dtype=mx.bfloat16
        )
        
        return prompt_embeds, prompt_mask

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: int = 0,
    ) -> tuple[mx.array, mx.array]:
        self._initialize_if_needed()
        if input_ids.ndim > 1:
            token_sequence = input_ids[0]
        else:
            token_sequence = input_ids

        try:
            token_list = token_sequence.tolist()
            prompt = self._tokenizer.decode(token_list, skip_special_tokens=True)
        except Exception:
            prompt = ""
        
        # Process using the alternative implementation
        prompt_embeds, prompt_mask = self._tokenize_and_process(prompt)
        
        # Apply position_ids offset if specified (for compatibility)
        if position_ids > 0:
            seq_len = prompt_embeds.shape[1]
            if position_ids < seq_len:
                prompt_embeds = prompt_embeds[:, position_ids:, :]
                prompt_mask = prompt_mask[:, position_ids:]
        
        return prompt_embeds, prompt_mask
