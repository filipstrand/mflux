from __future__ import annotations

import math
from dataclasses import dataclass

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common.config import ModelConfig
from mflux.models.ideogram4.constants import (
    LLM_TOKEN_INDICATOR,
    OUTPUT_IMAGE_INDICATOR,
    QWEN3_VL_ACTIVATION_LAYERS,
)
from mflux.models.ideogram4.fp8 import Fp8Linear


@dataclass(slots=True)
class Ideogram4Config:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512
    in_channels: int = 128
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)
    rope_theta: int = 5_000_000
    mrope_section: tuple[int, ...] = (24, 20, 20)
    norm_eps: float = 1e-5


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.mean(mx.square(x), axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return (self.weight.astype(mx.float32) * x).astype(input_dtype)


class Ideogram4LayerNormNoAffine(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        mean = mx.mean(x, axis=-1, keepdims=True)
        variance = mx.mean(mx.square(x - mean), axis=-1, keepdims=True)
        return ((x - mean) * mx.rsqrt(variance + self.eps)).astype(input_dtype)


class Ideogram4MRoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (base ** (mx.arange(0, head_dim, 2, dtype=mx.float32) / head_dim))
        self.mrope_section = tuple(mrope_section)
        self.head_dim = head_dim

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        if position_ids.ndim != 3 or position_ids.shape[-1] != 3:
            raise ValueError("position_ids must have shape (batch, seq, 3)")
        pos = position_ids.astype(mx.float32)
        freqs = []
        for axis in range(3):
            axis_pos = pos[:, :, axis][..., None]
            freqs.append(axis_pos * self.inv_freq[None, None, :])

        axis_selector = [0] * self.inv_freq.shape[0]
        for axis, offset in ((1, 1), (2, 2)):
            length = self.mrope_section[axis] * 3
            for idx in range(offset, length, 3):
                axis_selector[idx] = axis
        selector = mx.array(axis_selector, dtype=mx.int32)
        selector = mx.broadcast_to(
            selector[None, None, None, :],
            (position_ids.shape[0], position_ids.shape[1], 1, selector.shape[0]),
        )
        freq_stack = mx.stack(freqs, axis=-2)
        freqs_t = mx.squeeze(mx.take_along_axis(freq_stack, selector, axis=-2), axis=-2)

        emb = mx.concatenate([freqs_t, freqs_t], axis=-1)
        return mx.cos(emb), mx.sin(emb)


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.qkv = Fp8Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = Fp8Linear(hidden_size, hidden_size, bias=False)

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array,
        cos: mx.array,
        sin: mx.array,
    ) -> mx.array:
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0, :, :]
        k = qkv[:, :, 1, :, :]
        v = qkv[:, :, 2, :, :]

        q = self.norm_q(q).transpose(0, 2, 1, 3)
        k = self.norm_k(k).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        same_segment = segment_ids[:, :, None] == segment_ids[:, None, :]
        mask = mx.where(
            same_segment[:, None, :, :],
            mx.zeros((batch_size, 1, seq_len, seq_len), dtype=mx.float32),
            mx.full((batch_size, 1, seq_len, seq_len), -float("inf"), dtype=mx.float32),
        )
        out = scaled_dot_product_attention(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=self.scaling,
            mask=mask,
        )
        out = out.astype(x.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.hidden_size)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = Fp8Linear(dim, hidden_dim, bias=False)
        self.w2 = Fp8Linear(hidden_dim, dim, bias=False)
        self.w3 = Fp8Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adanln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)
        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.adaln_modulation = Fp8Linear(adanln_dim, 4 * hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        segment_ids: mx.array,
        cos: mx.array,
        sin: mx.array,
        adaln_input: mx.array,
    ) -> mx.array:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(mod, 4, axis=-1)
        gate_msa = mx.tanh(gate_msa)
        gate_mlp = mx.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            segment_ids=segment_ids,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        return x + gate_mlp * self.ffn_norm2(ffn_out)


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        self.mlp_in = Fp8Linear(dim, dim, bias=True)
        self.mlp_out = Fp8Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = x.astype(mx.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim).astype(ModelConfig.precision)
        emb = nn.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
        super().__init__()
        self.norm_final = Ideogram4LayerNormNoAffine(hidden_size, eps=1e-6)
        self.linear = Fp8Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = Fp8Linear(adanln_dim, hidden_size, bias=True)

    def __call__(self, x: mx.array, c: mx.array) -> mx.array:
        scale = 1.0 + self.adaln_modulation(nn.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(nn.Module):
    def __init__(self, config: Ideogram4Config | None = None) -> None:
        super().__init__()
        self.config = config or Ideogram4Config()
        head_dim = self.config.emb_dim // self.config.num_heads
        self.input_proj = Fp8Linear(self.config.in_channels, self.config.emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(self.config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = Fp8Linear(self.config.llm_features_dim, self.config.emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(self.config.emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = Fp8Linear(self.config.emb_dim, self.config.adanln_dim)
        self.embed_image_indicator = nn.Embedding(2, self.config.emb_dim)
        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=self.config.rope_theta,
            mrope_section=self.config.mrope_section,
        )
        self.layers = [
            Ideogram4TransformerBlock(
                hidden_size=self.config.emb_dim,
                intermediate_size=self.config.intermediate_size,
                num_heads=self.config.num_heads,
                norm_eps=self.config.norm_eps,
                adanln_dim=self.config.adanln_dim,
            )
            for _ in range(self.config.num_layers)
        ]
        self.final_layer = Ideogram4FinalLayer(
            hidden_size=self.config.emb_dim,
            out_channels=self.config.in_channels,
            adanln_dim=self.config.adanln_dim,
        )

    def __call__(
        self,
        *,
        llm_features: mx.array,
        x: mx.array,
        t: mx.array,
        position_ids: mx.array,
        segment_ids: mx.array,
        indicator: mx.array,
    ) -> mx.array:
        _, _, in_channels = x.shape
        if in_channels != self.config.in_channels:
            raise ValueError(f"x has {in_channels} channels, expected {self.config.in_channels}")
        x = x.astype(ModelConfig.precision)
        t = t.astype(ModelConfig.precision)
        llm_features = llm_features.astype(ModelConfig.precision)

        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).astype(x.dtype)[..., None]
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).astype(x.dtype)[..., None]

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask
        x = self.input_proj(x) * output_image_mask

        t_cond = self.t_embedding(t)
        if t.ndim == 1:
            t_cond = t_cond[:, None, :]
        adaln_input = nn.silu(self.adaln_proj(t_cond))

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask
        h = x + llm_features
        h = h + self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).astype(mx.int32))

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.astype(h.dtype)
        sin = sin.astype(h.dtype)
        for layer in self.layers:
            h = layer(
                h,
                segment_ids=segment_ids,
                cos=cos,
                sin=sin,
                adaln_input=adaln_input,
            )
        return self.final_layer(h, c=adaln_input).astype(mx.float32)


def _sinusoidal_embedding(t: mx.array, dim: int, scale: float = 1e4) -> mx.array:
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = mx.exp(mx.arange(half, dtype=mx.float32) * -freq)
    emb = t[..., None] * freq
    emb = mx.concatenate([mx.sin(emb), mx.cos(emb)], axis=-1)
    if dim % 2:
        emb = mx.pad(emb, [(0, 0)] * (emb.ndim - 1) + [(0, 1)])
    return emb


def _apply_rotary_pos_emb(
    q: mx.array,
    k: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> tuple[mx.array, mx.array]:
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x: mx.array) -> mx.array:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([-x2, x1], axis=-1)
