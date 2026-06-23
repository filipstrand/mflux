import math

import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention

from mflux.models.common_models.qwen3_vl.qwen3_vl_attention import Qwen3VLAttention


def _gelu_tanh(x: mx.array) -> mx.array:
    return 0.5 * x * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * mx.power(x, 3))))


class Krea2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = mx.zeros((dim,), dtype=mx.float32)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.mean(mx.square(hidden_states), axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.eps)
        hidden_states = hidden_states * (self.weight.astype(mx.float32) + 1.0)
        return hidden_states.astype(dtype)


class Krea2SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(hidden_states)) * self.up(hidden_states))


class Krea2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int | None = None,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.to_q = nn.Linear(hidden_size, self.head_dim * self.num_heads, bias=False)
        self.to_k = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_v = nn.Linear(hidden_size, self.head_dim * self.num_kv_heads, bias=False)
        self.to_gate = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm_q = Krea2RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Krea2RMSNorm(self.head_dim, eps=eps)
        self.to_out = [nn.Linear(hidden_size, hidden_size, bias=False)]

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array | None = None,
        image_rotary_emb: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        batch_size, seq_len, _ = hidden_states.shape
        query = self.to_q(hidden_states).reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        key = self.to_k(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value = self.to_v(hidden_states).reshape(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        gate = self.to_gate(hidden_states)

        query = self.norm_q(query).transpose(0, 2, 1, 3)
        key = self.norm_k(key).transpose(0, 2, 1, 3)
        value = value.transpose(0, 2, 1, 3)

        if image_rotary_emb is not None:
            query = Krea2Attention._apply_rotary(query, image_rotary_emb)
            key = Krea2Attention._apply_rotary(key, image_rotary_emb)

        if self.num_kv_heads != self.num_heads:
            key = Qwen3VLAttention._repeat_kv(key, self.num_key_value_groups)
            value = Qwen3VLAttention._repeat_kv(value, self.num_key_value_groups)

        attended = scaled_dot_product_attention(
            query.astype(mx.float32),
            key.astype(mx.float32),
            value.astype(mx.float32),
            scale=self.scale,
            mask=attention_mask,
        )
        attended = attended.astype(hidden_states.dtype).transpose(0, 2, 1, 3)
        attended = attended.reshape(batch_size, seq_len, self.hidden_size)
        attended = attended * mx.sigmoid(gate)
        return self.to_out[0](attended)

    @staticmethod
    def _apply_rotary(x: mx.array, image_rotary_emb: tuple[mx.array, mx.array]) -> mx.array:
        cos, sin = image_rotary_emb
        dtype = x.dtype
        x = x.astype(mx.float32)
        x = x.reshape(*x.shape[:-1], -1, 2)
        x_real = x[..., 0]
        x_imag = x[..., 1]
        cos = cos[None, None, :, :].astype(mx.float32)
        sin = sin[None, None, :, :].astype(mx.float32)
        out_real = x_real * cos - x_imag * sin
        out_imag = x_real * sin + x_imag * cos
        return mx.stack([out_real, out_imag], axis=-1).reshape(*x.shape[:-2], -1).astype(dtype)


class Krea2TextFusionBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, intermediate_size: int, eps: float) -> None:
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
    ) -> None:
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
        batch_size, seq_len, num_text_layers, dim = encoder_hidden_states.shape
        hidden_states = encoder_hidden_states.reshape(batch_size * seq_len, num_text_layers, dim)
        for block in self.layerwise_blocks:
            hidden_states = block(hidden_states)

        hidden_states = hidden_states.reshape(batch_size, seq_len, num_text_layers, dim)
        hidden_states = hidden_states.transpose(0, 1, 3, 2)
        hidden_states = self.projector(hidden_states).squeeze(-1)

        for block in self.refiner_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)

        return hidden_states


class Krea2TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        num_kv_heads: int,
        norm_eps: float,
    ) -> None:
        super().__init__()
        self.scale_shift_table = mx.zeros((6, hidden_size), dtype=mx.float32)
        self.norm1 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Krea2RMSNorm(hidden_size, eps=norm_eps)
        self.attn = Krea2Attention(hidden_size, num_heads, num_kv_heads, eps=norm_eps)
        self.ff = Krea2SwiGLU(hidden_size, intermediate_size)

    def __call__(
        self,
        hidden_states: mx.array,
        temb: mx.array,
        image_rotary_emb: tuple[mx.array, mx.array],
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        modulation = temb.reshape(temb.shape[0], 1, 6, -1) + self.scale_shift_table.reshape(1, 1, 6, -1)
        prescale = modulation[:, :, 0]
        preshift = modulation[:, :, 1]
        pregate = modulation[:, :, 2]
        postscale = modulation[:, :, 3]
        postshift = modulation[:, :, 4]
        postgate = modulation[:, :, 5]

        attn_out = self.attn(
            (1.0 + prescale) * self.norm1(hidden_states) + preshift,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )
        hidden_states = hidden_states + pregate * attn_out
        ff_out = self.ff((1.0 + postscale) * self.norm2(hidden_states) + postshift)
        hidden_states = hidden_states + postgate * ff_out
        return hidden_states


class Krea2TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear_1 = nn.Linear(embed_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, timestep: mx.array, dtype: mx.Dtype) -> mx.array:
        half = self.embed_dim // 2
        freqs = mx.exp(-math.log(1e4) * mx.arange(half, dtype=mx.float32) / half)
        args = (timestep.astype(mx.float32) * 1e3)[:, None, None] * freqs
        emb = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1).astype(dtype)
        return self.linear_2(_gelu_tanh(self.linear_1(emb)))


class Krea2TextProjection(nn.Module):
    def __init__(self, text_dim: int, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.norm = Krea2RMSNorm(text_dim, eps=eps)
        self.linear_1 = nn.Linear(text_dim, hidden_size, bias=True)
        self.linear_2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.linear_1(self.norm(hidden_states))
        return self.linear_2(_gelu_tanh(hidden_states))


class Krea2FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, eps: float) -> None:
        super().__init__()
        self.scale_shift_table = mx.zeros((2, hidden_size), dtype=mx.float32)
        self.norm = Krea2RMSNorm(hidden_size, eps=eps)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def __call__(self, hidden_states: mx.array, temb: mx.array) -> mx.array:
        modulation = temb + self.scale_shift_table
        scale = modulation[:, 0:1]
        shift = modulation[:, 1:2]
        hidden_states = (1.0 + scale) * self.norm(hidden_states) + shift
        return self.linear(hidden_states)


class Krea2RotaryPosEmbed(nn.Module):
    def __init__(self, theta: float, axes_dim: list[int]) -> None:
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def __call__(self, position_ids: mx.array) -> tuple[mx.array, mx.array]:
        pos = position_ids.astype(mx.float32)
        cos_out = []
        sin_out = []
        for axis, dim in enumerate(self.axes_dim):
            half = dim // 2
            scale = mx.arange(half, dtype=mx.float32) / half
            omega = 1.0 / (self.theta**scale)
            freqs = pos[:, axis : axis + 1] * omega
            cos_out.append(mx.cos(freqs))
            sin_out.append(mx.sin(freqs))
        return mx.concatenate(cos_out, axis=-1), mx.concatenate(sin_out, axis=-1)


class Krea2Transformer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        num_layers: int = 28,
        attention_head_dim: int = 128,
        num_attention_heads: int = 48,
        num_key_value_heads: int = 12,
        intermediate_size: int = 16384,
        timestep_embed_dim: int = 256,
        text_hidden_dim: int = 2560,
        num_text_layers: int = 12,
        text_num_attention_heads: int = 20,
        text_num_key_value_heads: int = 20,
        text_intermediate_size: int = 6912,
        num_layerwise_text_blocks: int = 2,
        num_refiner_text_blocks: int = 2,
        axes_dims_rope: tuple[int, int, int] = (32, 48, 48),
        rope_theta: float = 1000.0,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        hidden_size = attention_head_dim * num_attention_heads
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.hidden_size = hidden_size
        self.img_in = nn.Linear(in_channels, hidden_size, bias=True)
        self.time_embed = Krea2TimestepEmbedding(timestep_embed_dim, hidden_size)
        self.time_mod_proj = nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        self.text_fusion = Krea2TextFusion(
            num_text_layers=num_text_layers,
            dim=text_hidden_dim,
            num_heads=text_num_attention_heads,
            num_kv_heads=text_num_key_value_heads,
            intermediate_size=text_intermediate_size,
            num_layerwise_blocks=num_layerwise_text_blocks,
            num_refiner_blocks=num_refiner_text_blocks,
            eps=norm_eps,
        )
        self.txt_in = Krea2TextProjection(text_hidden_dim, hidden_size, eps=norm_eps)
        self.rotary_emb = Krea2RotaryPosEmbed(theta=rope_theta, axes_dim=list(axes_dims_rope))
        self.transformer_blocks = [
            Krea2TransformerBlock(
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_heads=num_attention_heads,
                num_kv_heads=num_key_value_heads,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ]
        self.final_layer = Krea2FinalLayer(hidden_size, out_channels=in_channels, eps=norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        timestep: mx.array,
        position_ids: mx.array,
        encoder_attention_mask: mx.array | None = None,
    ) -> mx.array:
        batch_size, image_seq_len, _ = hidden_states.shape
        text_seq_len = encoder_hidden_states.shape[1]

        temb = self.time_embed(timestep, dtype=hidden_states.dtype)
        temb_mod = self.time_mod_proj(_gelu_tanh(temb))

        text_attention_mask = None
        attention_mask = None
        if encoder_attention_mask is not None:
            text_attention_mask = Krea2Transformer._key_padding_mask(encoder_attention_mask)
            image_mask = mx.ones((batch_size, image_seq_len), dtype=encoder_attention_mask.dtype)
            full_mask = mx.concatenate([encoder_attention_mask, image_mask], axis=1)
            attention_mask = Krea2Transformer._key_padding_mask(full_mask)

        encoder_hidden_states = self.text_fusion(encoder_hidden_states, attention_mask=text_attention_mask)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)

        hidden_states = self.img_in(hidden_states)
        hidden_states = mx.concatenate([encoder_hidden_states, hidden_states], axis=1)
        image_rotary_emb = self.rotary_emb(position_ids)

        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb_mod, image_rotary_emb, attention_mask)

        hidden_states = hidden_states[:, text_seq_len:]
        return self.final_layer(hidden_states, temb)

    @staticmethod
    def _key_padding_mask(mask: mx.array) -> mx.array | None:
        mask = mask.astype(mx.float32)
        additive = (1.0 - mask) * -1e9
        return additive[:, None, None, :]

    @staticmethod
    def position_ids(text_seq_len: int, image_seq_len: int, height: int, width: int) -> mx.array:
        latent_height = height // 16
        latent_width = width // 16
        if latent_height * latent_width != image_seq_len:
            raise ValueError(
                f"Packed image sequence length {image_seq_len} does not match {latent_height}x{latent_width}."
            )
        text_ids = mx.zeros((text_seq_len, 3), dtype=mx.float32)
        rows = mx.arange(latent_height, dtype=mx.float32)
        cols = mx.arange(latent_width, dtype=mx.float32)
        row_ids = mx.broadcast_to(rows[:, None], (latent_height, latent_width))
        col_ids = mx.broadcast_to(cols[None, :], (latent_height, latent_width))
        time_ids = mx.zeros((latent_height, latent_width), dtype=mx.float32)
        image_ids = mx.stack([time_ids, row_ids, col_ids], axis=-1).reshape(image_seq_len, 3)
        return mx.concatenate([text_ids, image_ids], axis=0)
