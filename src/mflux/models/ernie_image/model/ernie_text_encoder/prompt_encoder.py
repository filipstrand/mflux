import mlx.core as mx

from mflux.models.common.tokenizer import Tokenizer
from mflux.models.ernie_image.model.ernie_text_encoder.text_encoder import ErnieMistralTextEncoder


class ErniePromptEncoder:
    @staticmethod
    def encode_prompt(
        prompt: str,
        tokenizer: Tokenizer,
        text_encoder: ErnieMistralTextEncoder,
        max_length: int = 256,
    ) -> tuple[mx.array, int]:
        """Encode a single prompt. Returns (embeddings [T, 3072], token_count)."""
        output = tokenizer.tokenize(prompt, max_length=max_length)
        hidden = text_encoder(output.input_ids, output.attention_mask)  # [1, T, 3072]
        num_valid = int(mx.sum(output.attention_mask[0]).item())
        return hidden[0, :num_valid, :], num_valid  # [T_actual, 3072]

    @staticmethod
    def build_text_batch(
        prompts: list[str],
        tokenizer: Tokenizer,
        text_encoder: ErnieMistralTextEncoder,
        max_length: int = 256,
        hidden_size: int = 3072,
    ) -> tuple[mx.array, mx.array]:
        """Encode a list of prompts, pad to the same length.

        Returns:
            text_bth: [B, Tmax, hidden_size]
            text_lens: [B] int32
        """
        embeddings = []
        lengths = []
        for p in prompts:
            emb, length = ErniePromptEncoder.encode_prompt(p, tokenizer, text_encoder, max_length)
            embeddings.append(emb)
            lengths.append(length)

        t_max = max(lengths)
        batch = []
        for emb in embeddings:
            pad_len = t_max - emb.shape[0]
            if pad_len > 0:
                pad = mx.zeros((pad_len, hidden_size), dtype=emb.dtype)
                emb = mx.concatenate([emb, pad], axis=0)
            batch.append(emb)

        text_bth = mx.stack(batch, axis=0)  # [B, Tmax, hidden_size]
        text_lens = mx.array(lengths, dtype=mx.int32)
        return text_bth, text_lens
