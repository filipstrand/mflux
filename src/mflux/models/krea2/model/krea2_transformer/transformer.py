import math

import mlx.core as mx
from mlx import nn

from mflux.models.krea2.model.krea2_transformer.config import Krea2TransformerConfig
from mflux.models.krea2.model.krea2_transformer.rope_embedder import Krea2RotaryPosEmbed, apply_rotary_emb


def _krea2_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    attention_mask: mx.array | None,
    num_heads: int,
    num_kv_heads: int,
) -> mx.array:
    """Numerically-careful GQA attention for Krea 2.

    Krea 2's per-head q/k RMSNorm has scale weights up to ~52 on individual dims, so q/k norms reach
    ~600 and the raw QK^T scores reach ~1e5. MLX's GPU matmul accumulates in reduced precision and
    loses enough bits there to flip ~15% of argmax decisions (the softmax is near-one-hot at that
    magnitude), giving cos ~0.94 vs the float32 reference. The QK^T score reduction is therefore run
    on the CPU stream (true float32 accumulation, matches torch); the value aggregation (weights in
    [0, 1]) is precision-safe and stays on the default (GPU) stream.
    """
    # GQA: repeat kv heads (interleave: kv head i serves query heads [i*n_rep, (i+1)*n_rep))
    if num_kv_heads != num_heads:
        n_rep = num_heads // num_kv_heads
        k = mx.repeat(k, n_rep, axis=1)
        v = mx.repeat(v, n_rep, axis=1)

    qf = q.astype(mx.float32)
    kf = k.astype(mx.float32)
    with mx.stream(mx.cpu):
        scores = (qf @ kf.transpose(0, 1, 3, 2)) * scale  # (B, H, Sq, Sk)
        if attention_mask is not None:
            scores = scores + attention_mask.astype(mx.float32)
        weights = mx.softmax(scores, axis=-1)
        mx.eval(weights)
    out = weights @ v.astype(mx.float32)
    return out


class Krea2RMSNorm(nn.Module):
    """RMSNorm with a zero-centered scale: effective multiplier is (1 + weight). Compute in float32."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.zeros((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        dtype = x.dtype
        xf = x.astype(mx.float32)
        variance = mx.mean(xf * xf, axis=-1, keepdims=True)
        xf = xf * mx.rsqrt(variance + self.eps)
        out = xf * (self.weight.astype(mx.float32) + 1.0)
        return out.astype(dtype)


class Krea2Attention(nn.Module):
    """GQA self-attention with q/k RMSNorm, optional RoPE, and a sigmoid output gate."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, eps: float = 1e-5):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(hidden_size, self.head_dim * num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = [nn.Linear(hidden_size, hidden_size, bias=False)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        rotary_cos_sin: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        b, s, _ = hidden_states.shape
        q = self.to_q(hidden_states).reshape(b, s, self.num_heads, self.head_dim)
        k = self.to_k(hidden_states).reshape(b, s, self.num_kv_heads, self.head_dim)
        v = self.to_v(hidden_states).reshape(b, s, self.num_kv_heads, self.head_dim)
        gate = self.to_gate(hidden_states)

        q = self.norm_q(q)
        k = self.norm_k(k)

        if rotary_cos_sin is not None:
            cos, sin = rotary_cos_sin
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        # to (B, H, S, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        out = _krea2_attention(q, k, v, self.scale, attention_mask, self.num_heads, self.num_kv_heads)
        out = out.astype(hidden_states.dtype)
        out = out.transpose(0, 2, 1, 3).reshape(b, s, self.num_heads * self.head_dim)
        out = out * mx.sigmoid(gate)
        return self.to_out[0](out)


class Krea2SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class Krea2TextFusionBlock(nn.Module):
    """Pre-norm block (no RoPE, no time modulation) for the text-fusion stage."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, intermediate_size: int, eps: float):
        super().__init__()
        self.norm1 = Krea2RMSNorm(dim, eps=eps)
        self.norm2 = Krea2RMSNorm(dim, eps=eps)
        self.attn = Krea2Attention(dim, num_heads, num_kv_heads, eps=eps)
        self.ff = Krea2SwiGLU(dim, intermediate_size)

    def __call__(self, hidden_states: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        hidden_states = hidden_states + self.attn(self.norm1(hidden_states), attention_mask=attention_mask)
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class Krea2TextFusion(nn.Module):
    def __init__(
        self,
        num_text_layers: int,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        num_layerwise_blocks: int,
        num_refiner_blocks: int,
        eps: float,
    ):
        super().__init__()
        self.layerwise_blocks = [
            Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
            for _ in range(num_layerwise_blocks)
        ]
        self.projector = nn.Linear(num_text_layers, 1, bias=False)
        self.refiner_blocks = [
            Krea2TextFusionBlock(dim, num_heads, num_kv_heads, intermediate_size, eps)
            for _ in range(num_refiner_blocks)
        ]

    def __call__(self, encoder_hidden_states: mx.array, attention_mask: mx.array | None = None) -> mx.array:
        b, s, num_text_layers, dim = encoder_hidden_states.shape

        # layerwise: attend across the layer axis per token
        hidden_states = encoder_hidden_states.reshape(b * s, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states)

        # collapse the layer axis via the (1, num_text_layers) projector
        hidden_states = hidden_states.reshape(b, s, num_text_layers, dim).transpose(0, 1, 3, 2)
        hidden_states = self.projector(hidden_states).squeeze(-1)  # (b, s, dim)

        # refiner: attend across tokens
        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return hidden_states


class Krea2TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, num_heads: int, num_kv_heads: int, norm_eps: float):
        super().__init__()
        self.scale_shift_table = mx.zeros((6, hidden_size))
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        rotary_cos_sin: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        # temb: (B, 1, 6*hidden), shared across blocks; add per-block table.
        b = temb.shape[0]
        modulation = temb.reshape(b, 1, 6, -1) + self.scale_shift_table  # (B, 1, 6, hidden)
        prescale = modulation[:, :, 0]
        preshift = modulation[:, :, 1]
        pregate = modulation[:, :, 2]
        postscale = modulation[:, :, 3]
        postshift = modulation[:, :, 4]
        postgate = modulation[:, :, 5]

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            rotary_cos_sin=rotary_cos_sin,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


def _gelu_tanh(x: mx.array) -> mx.array:
    return nn.gelu_approx(x)


class Krea2TimestepEmbedding(nn.Module):
    """Sinusoidal (cos-first, input x1000) + 2-layer MLP (gelu-tanh)."""

    def __init__(self, embed_dim: int, hidden_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, timestep: mx.array, dtype: mx.Dtype) -> mx.array:
        half = self.embed_dim // 2
        freqs = mx.exp(-math.log(1e4) * mx.arange(half, dtype=mx.float32) / half)
        args = (timestep.astype(mx.float32) * 1e3)[:, None, None] * freqs  # (B, 1, half)
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(dtype)  # (B, 1, embed_dim)
        return self.linear_2(_gelu_tanh(self.linear_1(emb)))


class Krea2TextProjection(nn.Module):
    def __init__(self, text_dim: int, hidden_size: int, eps: float):
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(_gelu_tanh(hidden_states))


class Krea2FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, eps: float):
        super().__init__()
        self.scale_shift_table = mx.zeros((2, hidden_size))
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, hidden_states: mx.array, temb: mx.array) -> mx.array:
        # temb: (B, hidden). modulation: (B, 2, hidden)
        modulation = temb[:, None, :] + self.scale_shift_table  # (B, 2, hidden)
        scale = modulation[:, 0:1]  # (B, 1, hidden)
        shift = modulation[:, 1:2]
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


class Krea2Transformer(nn.Module):
    def __init__(self, **overrides):
        super().__init__()
        config = Krea2TransformerConfig(**overrides)
        self.config = config
        hidden_size = config.hidden_size

        self.img_in = nn.Linear(config.in_channels, hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(config.timestep_embed_dim, hidden_size)
        self.time_mod_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=config.num_text_layers,
            dim=config.text_hidden_dim,
            num_heads=config.text_num_attention_heads,
            num_kv_heads=config.text_num_key_value_heads,
            intermediate_size=config.text_intermediate_size,
            num_layerwise_blocks=config.num_layerwise_text_blocks,
            num_refiner_blocks=config.num_refiner_text_blocks,
            eps=config.norm_eps,
        )
        self.txt_in = Krea2TextProjection(config.text_hidden_dim, hidden_size, eps=config.norm_eps)
        self.rotary_emb = Krea2RotaryPosEmbed(theta=config.rope_theta, axes_dim=list(config.axes_dims_rope))
        self.transformer_blocks = [
            Krea2TransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=config.intermediate_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                norm_eps=config.norm_eps,
            )
            for _ in range(config.num_layers)
        ]
        self.final_layer = Krea2FinalLayer(hidden_size, out_channels=config.in_channels, eps=config.norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        position_ids: mx.array,
        encoder_attention_mask: mx.array | None = None,
        rotary_cos_sin: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        b, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]
        dtype = hidden_states.dtype

        temb = self.time_embed(timestep, dtype=dtype)  # (B, 1, hidden)
        temb_mod = self.time_mod_proj(_gelu_tanh(temb))  # (B, 1, 6*hidden)

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            # encoder_attention_mask: (B, text_seq_len) bool/0-1. Build additive key-padding masks.
            neg = mx.array(-1e9, dtype=mx.float32)
            zero = mx.array(0.0, dtype=mx.float32)
            text_bias = mx.where(encoder_attention_mask, zero, neg)  # (B, T)
            text_attention_mask = text_bias[:, None, None, :]  # (B,1,1,T)
            image_ones = mx.ones((b, image_seq_len), dtype=encoder_attention_mask.dtype)
            full = mx.concatenate([encoder_attention_mask, image_ones], axis=1)
            full_bias = mx.where(full, zero, neg)
            attention_mask = full_bias[:, None, None, :]  # (B,1,1,T+I)

        encoder_hidden_states = self.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)

        if rotary_cos_sin is None:
            rotary_cos_sin = self.rotary_emb.compute(position_ids, dtype=dtype)

        # Optional gradient checkpointing: recompute each block's activations during backward instead
        # of storing them, which trades compute for a large drop in peak memory (the 28-block activation
        # graph of a ~12B transformer dominates training RAM). Off by default (inference unaffected); the
        # training adapter turns it on. nn.utils.checkpoint checkpoints w.r.t. the module's trainable
        # params too, so LoRA gradients stay exact.
        gradient_checkpointing = getattr(self, "gradient_checkpointing", False)
        for block in self.transformer_blocks:
            run = nn.utils.checkpoint(block) if gradient_checkpointing else block
            hidden_states = run(hidden_states, temb_mod, rotary_cos_sin, attention_mask)

        hidden_states = hidden_states[:, text_seq_len:]
        return self.final_layer(hidden_states, temb[:, 0, :])
