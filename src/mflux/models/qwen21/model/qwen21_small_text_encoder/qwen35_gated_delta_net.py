import mlx.core as mx
from mlx import nn

from mflux.models.qwen21.model.qwen21_small_text_encoder.qwen35_norms import Qwen35RMSNormGated


class Qwen35GatedDeltaNet(nn.Module):
    CHUNK_SIZE = 64

    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int,
        eps: float,
    ):
        super().__init__()
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = num_k_heads * head_k_dim
        self.value_dim = num_v_heads * head_v_dim
        conv_dim = 2 * self.key_dim + self.value_dim
        self.in_proj_qkv = nn.Linear(hidden_size, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(hidden_size, num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(hidden_size, num_v_heads, bias=False)
        # Depthwise causal conv, kept in the checkpoint layout (channels, 1, kernel).
        self.conv1d = nn.Module()
        self.conv1d.weight = mx.zeros((conv_dim, 1, conv_kernel_size))
        self.dt_bias = mx.ones((num_v_heads,))
        self.A_log = mx.zeros((num_v_heads,))
        self.norm = Qwen35RMSNormGated(head_v_dim, eps=eps)
        self.out_proj = nn.Linear(self.value_dim, hidden_size, bias=False)

    def __call__(self, hidden: mx.array) -> mx.array:
        batch, length, _ = hidden.shape
        mixed = nn.silu(self._causal_conv(self.in_proj_qkv(hidden)))
        query, key, value = mx.split(mixed, [self.key_dim, 2 * self.key_dim], axis=-1)
        query = query.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        key = key.reshape(batch, length, self.num_k_heads, self.head_k_dim)
        value = value.reshape(batch, length, self.num_v_heads, self.head_v_dim)
        beta = mx.sigmoid(self.in_proj_b(hidden))
        a = self.in_proj_a(hidden).astype(mx.float32)
        decay = -mx.exp(self.A_log.astype(mx.float32)) * nn.softplus(a + self.dt_bias.astype(mx.float32))
        repeats = self.num_v_heads // self.num_k_heads
        if repeats > 1:
            query = mx.repeat(query, repeats, axis=2)
            key = mx.repeat(key, repeats, axis=2)
        core = Qwen35GatedDeltaNet.chunk_gated_delta_rule(query, key, value, decay, beta, self.CHUNK_SIZE)
        z = self.in_proj_z(hidden).reshape(-1, self.head_v_dim)
        core = self.norm(core.reshape(-1, self.head_v_dim), z).reshape(batch, length, -1)
        return self.out_proj(core)

    @staticmethod
    def chunk_gated_delta_rule(
        query: mx.array,
        key: mx.array,
        value: mx.array,
        decay: mx.array,
        beta: mx.array,
        chunk_size: int = 64,
    ) -> mx.array:
        # Prefill form of the gated delta rule (port of transformers' torch_chunk_gated_delta_rule), in float32.
        dtype = value.dtype
        query = Qwen35GatedDeltaNet._l2norm(query.astype(mx.float32))
        key = Qwen35GatedDeltaNet._l2norm(key.astype(mx.float32))
        query, key, value = (x.astype(mx.float32).transpose(0, 2, 1, 3) for x in (query, key, value))
        beta = beta.astype(mx.float32).transpose(0, 2, 1)
        decay = decay.astype(mx.float32).transpose(0, 2, 1)
        batch, heads, length, k_dim = key.shape
        v_dim = value.shape[-1]
        pad = (chunk_size - length % chunk_size) % chunk_size
        if pad:
            query, key, value = (mx.pad(x, [(0, 0), (0, 0), (0, pad), (0, 0)]) for x in (query, key, value))
            beta, decay = (mx.pad(x, [(0, 0), (0, 0), (0, pad)]) for x in (beta, decay))
        chunks = (length + pad) // chunk_size
        query = query * (k_dim**-0.5)
        v_beta = value * beta[..., None]
        k_beta = key * beta[..., None]
        query, key, value, k_beta, v_beta = (
            x.reshape(batch, heads, chunks, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
        )
        decay = mx.cumsum(decay.reshape(batch, heads, chunks, chunk_size), axis=-1)
        lower = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_))
        decay_mask = mx.where(lower, mx.exp(decay[..., :, None] - decay[..., None, :]), 0.0)
        strictly_lower = mx.tril(mx.ones((chunk_size, chunk_size), dtype=mx.bool_), k=-1)
        attn = mx.where(strictly_lower, -((k_beta @ key.swapaxes(-1, -2)) * decay_mask), 0.0)
        for i in range(1, chunk_size):
            row = attn[..., i : i + 1, :i]
            attn[..., i : i + 1, :i] = row + row @ attn[..., :i, :i]
        attn = attn + mx.eye(chunk_size, dtype=mx.float32)
        value = attn @ v_beta
        k_cumdecay = attn @ (k_beta * mx.exp(decay)[..., None])
        state = mx.zeros((batch, heads, k_dim, v_dim), dtype=mx.float32)
        outputs = []
        for i in range(chunks):
            q_i, k_i, v_i, g_i = query[:, :, i], key[:, :, i], value[:, :, i], decay[:, :, i]
            intra = (q_i @ k_i.swapaxes(-1, -2)) * decay_mask[:, :, i]
            v_new = v_i - k_cumdecay[:, :, i] @ state
            outputs.append((q_i * mx.exp(g_i)[..., None]) @ state + intra @ v_new)
            last = g_i[..., -1:]
            state = state * mx.exp(last)[..., None] + (k_i * mx.exp(last - g_i)[..., None]).swapaxes(-1, -2) @ v_new
        out = mx.concatenate(outputs, axis=2)[:, :, :length]
        return out.transpose(0, 2, 1, 3).astype(dtype)

    def _causal_conv(self, x: mx.array) -> mx.array:
        weight = self.conv1d.weight.astype(mx.float32)
        kernel = weight.shape[-1]
        length = x.shape[1]
        padded = mx.pad(x.astype(mx.float32), [(0, 0), (kernel - 1, 0), (0, 0)])
        out = sum(padded[:, k : k + length] * weight[:, 0, k] for k in range(kernel))
        return out.astype(x.dtype)

    @staticmethod
    def _l2norm(x: mx.array, eps: float = 1e-6) -> mx.array:
        return x * mx.rsqrt(mx.sum(x * x, axis=-1, keepdims=True) + eps)
