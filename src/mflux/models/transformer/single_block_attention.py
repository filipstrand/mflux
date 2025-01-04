import mlx.core as mx
from mlx import nn

from mflux.flux.v_cache import VCache


class SingleBlockAttention(nn.Module):
    head_dimension = 128
    batch_size = 1
    num_heads = 24

    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.to_q = nn.Linear(3072, 3072)
        self.to_k = nn.Linear(3072, 3072)
        self.to_v = nn.Linear(3072, 3072)
        self.norm_q = nn.RMSNorm(128)
        self.norm_k = nn.RMSNorm(128)

    def __call__(self, t: float, hidden_states: mx.array, image_rotary_emb: mx.array) -> (mx.array, mx.array):
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)

        # Handle the values from inversion
        key_hash = hash((t, id(self)))
        value = self.to_v(hidden_states)

        if self.layer > 15:
            if VCache.is_inverting:
                if t <= VCache.t_max:
                    VCache.v_cache[key_hash] = mx.array(value)
            else:
                if t <= VCache.t_max:
                    value = VCache.v_cache.get(key_hash, None)
                    value = value if value is not None else self.to_v(hidden_states)

        query = mx.transpose(mx.reshape(query, (1, -1, 24, 128)), (0, 2, 1, 3))
        key = mx.transpose(mx.reshape(key, (1, -1, 24, 128)), (0, 2, 1, 3))
        value = mx.transpose(mx.reshape(value, (1, -1, 24, 128)), (0, 2, 1, 3))

        query = self.norm_q(query)
        key = self.norm_k(key)

        query, key = SingleBlockAttention.apply_rope(query, key, image_rotary_emb)

        hidden_states = SingleBlockAttention.attention(query, key, value)
        hidden_states = mx.transpose(hidden_states, (0, 2, 1, 3))
        hidden_states = mx.reshape(
            hidden_states,
            (self.batch_size, -1, self.num_heads * self.head_dimension),
        )

        return hidden_states

    @staticmethod
    def attention(query, key, value):
        scale = 1 / mx.sqrt(query.shape[-1])
        scores = (query * scale) @ key.transpose(0, 1, 3, 2)
        attn = mx.softmax(scores, axis=-1)
        hidden_states = attn @ value
        return hidden_states

    @staticmethod
    def apply_rope(xq: mx.array, xk: mx.array, freqs_cis: mx.array):
        xq_ = xq.astype(mx.float32).reshape(*xq.shape[:-1], -1, 1, 2)
        xk_ = xk.astype(mx.float32).reshape(*xk.shape[:-1], -1, 1, 2)
        xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
        xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
        return xq_out.reshape(*xq.shape).astype(mx.float32), xk_out.reshape(*xk.shape).astype(mx.float32)
