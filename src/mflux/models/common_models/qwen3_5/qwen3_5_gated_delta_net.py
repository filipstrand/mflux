import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_5.qwen3_5_rms_norm import Qwen35GatedRMSNorm


class Qwen35CausalConv1d(nn.Module):
    # Depthwise causal convolution. The weight keeps the checkpoint layout (channels, 1, kernel).
    def __init__(self, channels: int, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight = mx.zeros((channels, 1, kernel_size))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, C)
        seq_len = x.shape[1]
        padded = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])
        weight = self.weight[:, 0, :].astype(mx.float32)
        out = mx.zeros(x.shape, dtype=mx.float32)
        for tap in range(self.kernel_size):
            out = out + padded[:, tap : tap + seq_len, :].astype(mx.float32) * weight[:, tap]
        return out.astype(x.dtype)


class Qwen35GatedDeltaNet(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_key_heads: int,
        num_value_heads: int,
        key_head_dim: int,
        value_head_dim: int,
        conv_kernel_size: int,
        rms_norm_eps: float,
    ):
        super().__init__()
        self.num_key_heads = num_key_heads
        self.num_value_heads = num_value_heads
        self.key_head_dim = key_head_dim
        self.value_head_dim = value_head_dim
        self.key_dim = key_head_dim * num_key_heads
        self.value_dim = value_head_dim * num_value_heads

        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = Qwen35CausalConv1d(conv_dim, conv_kernel_size)
        self.dt_bias = mx.ones((num_value_heads,))
        self.A_log = mx.zeros((num_value_heads,))
        self.norm = Qwen35GatedRMSNorm(value_head_dim, eps=rms_norm_eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)
        self.in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_value_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_value_heads, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        batch, seq_len, _ = hidden_states.shape

        mixed_qkv = nn.silu(self.conv1d(self.in_proj_qkv(hidden_states)))
        query, key, value = mx.split(mixed_qkv, [self.key_dim, 2 * self.key_dim], axis=-1)
        query = query.reshape(batch, seq_len, self.num_key_heads, self.key_head_dim)
        key = key.reshape(batch, seq_len, self.num_key_heads, self.key_head_dim)
        value = value.reshape(batch, seq_len, self.num_value_heads, self.value_head_dim)

        z = self.in_proj_z(hidden_states).reshape(batch, seq_len, self.num_value_heads, self.value_head_dim)
        beta = mx.sigmoid(self.in_proj_b(hidden_states).astype(mx.float32))
        a = self.in_proj_a(hidden_states).astype(mx.float32)
        g = -mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(a + self.dt_bias.astype(mx.float32))

        repeats = self.num_value_heads // self.num_key_heads
        if repeats > 1:
            query = mx.repeat(query, repeats, axis=2)
            key = mx.repeat(key, repeats, axis=2)

        core = Qwen35GatedDeltaNet._gated_delta_rule(query, key, value, g, beta).astype(hidden_states.dtype)
        core = self.norm(core, z).reshape(batch, seq_len, self.value_dim)
        return self.out_proj(core)

    @staticmethod
    def _gated_delta_rule(query: mx.array, key: mx.array, value: mx.array, g: mx.array, beta: mx.array) -> mx.array:
        # Sequential form of the gated delta rule, in float32. It gives the same result as the
        # chunked form of the reference and is fast enough for prompt-length inputs.
        query = Qwen35GatedDeltaNet._l2_normalize(query.astype(mx.float32))
        key = Qwen35GatedDeltaNet._l2_normalize(key.astype(mx.float32))
        value = value.astype(mx.float32)
        query = query * (query.shape[-1] ** -0.5)

        batch, seq_len, heads, key_dim = key.shape
        state = mx.zeros((batch, heads, key_dim, value.shape[-1]), dtype=mx.float32)
        outputs = []
        for t in range(seq_len):
            state = state * mx.exp(g[:, t])[:, :, None, None]
            k_t = key[:, t]
            kv_mem = mx.sum(state * k_t[..., None], axis=-2)
            delta = (value[:, t] - kv_mem) * beta[:, t][..., None]
            state = state + k_t[..., None] * delta[..., None, :]
            outputs.append(mx.sum(state * query[:, t][..., None], axis=-2))
        return mx.stack(outputs, axis=1)

    @staticmethod
    def _l2_normalize(x: mx.array, eps: float = 1e-6) -> mx.array:
        return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)
