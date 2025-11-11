from typing import List

import mlx.core as mx
from mlx import nn

from .smol_lm3_3b_encoder_layer import SmolLM3_3B_EncoderLayer
from .smol_lm3_3b_rms_norm import SmolLM3_3B_RMSNorm
from .smol_lm3_3b_rope import SmolLM3_3B_RotaryEmbedding


class SmolLM3_3B_TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 128_256,
        hidden_size: int = 2048,
        intermediate_size: int = 11_008,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        max_position_embeddings: int = 65_536,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        hidden_act: str = "silu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = hidden_act
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers: List[SmolLM3_3B_EncoderLayer] = [
            SmolLM3_3B_EncoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                intermediate_size=intermediate_size,
                rms_norm_eps=rms_norm_eps,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                hidden_act=hidden_act,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = SmolLM3_3B_RMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = SmolLM3_3B_RotaryEmbedding(
            dim=hidden_size // num_attention_heads,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        output_hidden_states: bool = True,
    ) -> List[mx.array] | mx.array:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)
        attention_mask_4d = SmolLM3_3B_TextEncoder._build_attention_mask(attention_mask)
        cos, sin = self.rotary_emb(seq_len)
        hidden_states_list: List[mx.array] = [hidden_states]
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states=hidden_states,
                attention_mask=attention_mask_4d,
                cos_sin=(cos, sin),
                layer_idx=layer_idx,
            )
            if output_hidden_states:
                hidden_states_list.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            hidden_states_list[-1] = hidden_states
            return hidden_states_list
        return hidden_states

    @staticmethod
    def _build_attention_mask(attention_mask: mx.array) -> mx.array:
        batch_size, seq_len = attention_mask.shape
        mask_dtype = mx.float32
        min_dtype_value = mx.finfo(mask_dtype).min
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros_like(attention_mask).astype(mask_dtype),
            mx.ones_like(attention_mask).astype(mask_dtype) * min_dtype_value,
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)
        idx = mx.arange(seq_len, dtype=mx.int32)
        j = mx.expand_dims(idx, axis=0)
        i = mx.expand_dims(idx, axis=1)
        tri_bool = j > i
        zeros_2d = mx.zeros((seq_len, seq_len), dtype=mask_dtype)
        minval_2d = mx.ones((seq_len, seq_len), dtype=mask_dtype) * min_dtype_value
        causal_tri_mask = mx.where(tri_bool, minval_2d, zeros_2d)
        causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)
        causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_tri_mask + padding_mask
        return attention_mask_4d
