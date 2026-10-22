import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_small_text_encoder.qwen35_gated_delta_net import Qwen35GatedDeltaNet
from mflux.models.qwen21.model.qwen21_small_text_encoder.qwen35_norms import Qwen35RMSNorm


class Qwen35RotaryEmbedding:
    def __init__(self, head_dim: int, partial_rotary_factor: float, theta: float, mrope_section: list[int]):
        dim = int(head_dim * partial_rotary_factor)
        self.inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
        # Interleaved M-RoPE: frequency i reads the T, H or W position as [T H W T H W ... T T].
        axis = [0] * (dim // 2)
        for section, offset in ((1, 1), (2, 2)):
            for index in range(offset, mrope_section[section] * 3, 3):
                axis[index] = section
        self.axis = mx.array(axis, dtype=mx.int32)

    def __call__(self, positions: mx.array, dtype: mx.Dtype) -> tuple[mx.array, mx.array]:
        # positions: (3, batch, length)
        freqs = positions.astype(mx.float32)[..., None] * self.inv_freq
        freqs = mx.take_along_axis(freqs, self.axis[None, None, None, :], axis=0)[0]
        emb = mx.concatenate([freqs, freqs], axis=-1)
        return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)

    @staticmethod
    def apply(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        rotary = cos.shape[-1]
        x_rot, x_pass = x[..., :rotary], x[..., rotary:]
        half = rotary // 2
        rotated = mx.concatenate([-x_rot[..., half:], x_rot[..., :half]], axis=-1)
        return mx.concatenate([x_rot * cos + rotated * sin, x_pass], axis=-1)


class Qwen35Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int, eps: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # q_proj also yields the sigmoid output gate (attn_output_gate).
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim * 2, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.q_norm = Qwen35RMSNorm(head_dim, eps=eps)
        self.k_norm = Qwen35RMSNorm(head_dim, eps=eps)

    def __call__(self, hidden: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        batch, length, _ = hidden.shape
        query, gate = mx.split(self.q_proj(hidden).reshape(batch, length, self.num_heads, 2 * self.head_dim), 2, -1)
        gate = gate.reshape(batch, length, -1)
        query = self.q_norm(query).transpose(0, 2, 1, 3)
        key = self.k_norm(self.k_proj(hidden).reshape(batch, length, self.num_kv_heads, -1)).transpose(0, 2, 1, 3)
        value = self.v_proj(hidden).reshape(batch, length, self.num_kv_heads, -1).transpose(0, 2, 1, 3)
        cos, sin = cos[:, None], sin[:, None]
        query = Qwen35RotaryEmbedding.apply(query, cos, sin)
        key = Qwen35RotaryEmbedding.apply(key, cos, sin)
        out = mx.fast.scaled_dot_product_attention(query, key, value, scale=self.head_dim**-0.5, mask="causal")
        out = out.transpose(0, 2, 1, 3).reshape(batch, length, -1)
        return self.o_proj(out * mx.sigmoid(gate))


class Qwen35MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen35DecoderLayer(nn.Module):
    def __init__(self, config: dict, layer_type: str):
        super().__init__()
        hidden, eps = config["hidden_size"], config["rms_norm_eps"]
        self.layer_type = layer_type
        if layer_type == "linear_attention":
            self.linear_attn = Qwen35GatedDeltaNet(
                hidden_size=hidden,
                num_k_heads=config["linear_num_key_heads"],
                num_v_heads=config["linear_num_value_heads"],
                head_k_dim=config["linear_key_head_dim"],
                head_v_dim=config["linear_value_head_dim"],
                conv_kernel_size=config["linear_conv_kernel_dim"],
                eps=eps,
            )
        elif layer_type == "full_attention":
            self.self_attn = Qwen35Attention(
                hidden_size=hidden,
                num_heads=config["num_attention_heads"],
                num_kv_heads=config["num_key_value_heads"],
                head_dim=config["head_dim"],
                eps=eps,
            )
        else:
            raise ValueError(f"Unsupported Qwen3.5 layer type: {layer_type!r}")
        self.mlp = Qwen35MLP(hidden, config["intermediate_size"])
        self.input_layernorm = Qwen35RMSNorm(hidden, eps=eps)
        self.post_attention_layernorm = Qwen35RMSNorm(hidden, eps=eps)

    def __call__(self, hidden: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        normed = self.input_layernorm(hidden)
        if self.layer_type == "linear_attention":
            hidden = hidden + self.linear_attn(normed)
        else:
            hidden = hidden + self.self_attn(normed, cos, sin)
        return hidden + self.mlp(self.post_attention_layernorm(hidden))


class Qwen35LanguageModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        rope = config["rope_parameters"]
        if not rope.get("mrope_interleaved", False):
            raise ValueError("Only the interleaved M-RoPE layout of Qwen3.5 is supported.")
        self.embed_tokens = nn.Embedding(config["vocab_size"], config["hidden_size"])
        self.layers = [Qwen35DecoderLayer(config, layer_type) for layer_type in config["layer_types"]]
        self.norm = Qwen35RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.rotary_emb = Qwen35RotaryEmbedding(
            head_dim=config["head_dim"],
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            theta=rope["rope_theta"],
            mrope_section=rope["mrope_section"],
        )

    def hidden_states(self, hidden: mx.array, positions: mx.array, layers: tuple[int, ...]) -> list[mx.array]:
        # Index i follows transformers' output_hidden_states: 0 is the embedding, i the output of layer i,
        # and the last index is the final-norm output.
        cos, sin = self.rotary_emb(positions, hidden.dtype)
        collected = {0: hidden} if 0 in layers else {}
        for index, layer in enumerate(self.layers, start=1):
            hidden = layer(hidden, cos, sin)
            if index == len(self.layers):
                hidden = self.norm(hidden)
            if index in layers:
                collected[index] = hidden
            mx.eval(hidden)
        return [collected[index] for index in layers]
