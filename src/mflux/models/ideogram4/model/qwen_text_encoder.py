from __future__ import annotations

import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_rotary_embedding import Qwen3TextRotaryEmbedding
from mflux.models.ideogram4.constants import QWEN3_VL_ACTIVATION_LAYERS
from mflux.models.ideogram4.fp8 import Fp8Linear


class Qwen3VLMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = Fp8Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = Fp8Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = Fp8Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        gate_output = nn.silu(self.gate_proj(hidden_states))
        up_output = self.up_proj(hidden_states)
        return self.down_proj(gate_output * up_output)


class Qwen3VLAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        del max_position_embeddings
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.q_proj = Fp8Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.k_proj = Fp8Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.v_proj = Fp8Linear(hidden_size, num_key_value_heads * head_dim, bias=False)
        self.o_proj = Fp8Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = Qwen3VLRMSNorm(head_dim, eps=rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape

        q_proj = self.q_proj(hidden_states)
        k_proj = self.k_proj(hidden_states)
        v_proj = self.v_proj(hidden_states)

        query_states = q_proj.reshape(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        key_states = k_proj.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = v_proj.reshape(batch_size, seq_len, self.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        query_states = query_states.transpose(0, 2, 1, 3)
        key_states = key_states.transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = _apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.num_key_value_heads != self.num_attention_heads:
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)

        attn_output = scaled_dot_product_attention(
            query_states.astype(mx.float32),
            key_states.astype(mx.float32),
            value_states.astype(mx.float32),
            scale=self.scaling,
            mask=attention_mask,
        )
        attn_output = attn_output.astype(hidden_states.dtype)
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attn_output)


class Qwen3VLDecoderLayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        rms_norm_eps: float,
        intermediate_size: int,
    ):
        super().__init__()
        self.input_layernorm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.self_attn = Qwen3VLAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            rope_theta=rope_theta,
            rms_norm_eps=rms_norm_eps,
        )
        self.post_attention_layernorm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.mlp = Qwen3VLMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        position_embeddings: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 4096,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 12288,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5_000_000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [
            Qwen3VLDecoderLayer(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                max_position_embeddings=max_position_embeddings,
                rope_theta=rope_theta,
                rms_norm_eps=rms_norm_eps,
                intermediate_size=intermediate_size,
            )
            for _ in range(num_hidden_layers)
        ]
        self.norm = Qwen3VLRMSNorm(hidden_size, eps=rms_norm_eps)
        self.rotary_emb = Qwen3TextRotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            base=rope_theta,
        )

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
        position_ids: mx.array | None = None,
        tap_layers: tuple[int, ...] = QWEN3_VL_ACTIVATION_LAYERS,
    ) -> list[mx.array]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)
        if position_ids is None:
            position_ids = mx.broadcast_to(
                mx.arange(seq_len, dtype=mx.int32)[None, :],
                (batch_size, seq_len),
            )

        mask_dtype = hidden_states.dtype
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = padding_mask[:, None, None, :]

        idx = mx.arange(seq_len, dtype=mx.int32)
        causal = idx[None, :] > idx[:, None]
        causal_mask = mx.where(
            causal,
            mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype),
            mx.zeros((seq_len, seq_len), dtype=mask_dtype),
        )
        causal_mask = mx.broadcast_to(causal_mask[None, None, :, :], (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_mask + padding_mask
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        tap_set = set(tap_layers)
        captured: dict[int, mx.array] = {}
        for layer_idx, layer in enumerate(self.layers):
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask_4d,
                position_embeddings=position_embeddings,
            )
            if layer_idx in tap_set:
                captured[layer_idx] = hidden_states
        return [captured[i] for i in tap_layers]

    def get_prompt_embeds(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
        position_ids: mx.array,
        tap_layers: tuple[int, ...] = QWEN3_VL_ACTIVATION_LAYERS,
    ) -> mx.array:
        selected = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            tap_layers=tap_layers,
        )
        stacked = mx.stack(selected, axis=0)
        stacked = mx.transpose(stacked, (1, 2, 3, 0))
        batch_size, seq_len, hidden_dim, num_layers = stacked.shape
        return stacked.reshape(batch_size, seq_len, hidden_dim * num_layers)


def _repeat_kv(hidden_states: mx.array, n_rep: int) -> mx.array:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    hidden_states = mx.expand_dims(hidden_states, axis=2)
    hidden_states = mx.broadcast_to(hidden_states, (batch, num_key_value_heads, n_rep, slen, head_dim))
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
    unsqueeze_dim: int = 1,
) -> tuple[mx.array, mx.array]:
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x: mx.array) -> mx.array:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)
