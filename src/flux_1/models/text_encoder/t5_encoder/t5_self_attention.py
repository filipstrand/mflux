import math

import mlx.core as mx
from mlx import nn


class T5SelfAttention(nn.Module):

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(4096, 4096, bias=False)
        self.k = nn.Linear(4096, 4096, bias=False)
        self.v = nn.Linear(4096, 4096, bias=False)
        self.relative_attention_bias = nn.Embedding(32, 64)
        self.o = nn.Linear(4096, 4096, bias=False)

    def forward(self, hidden_states: mx.array) -> mx.array:
        query_states = T5SelfAttention.shape(self.q(hidden_states))
        key_states = T5SelfAttention.shape(self.k(hidden_states))
        value_states = T5SelfAttention.shape(self.v(hidden_states))
        scores = mx.matmul(query_states, mx.transpose(key_states, (0, 1, 3, 2)))
        seq_length = hidden_states.shape[1]
        position_bias = self._compute_bias(seq_length=seq_length)
        scores += position_bias
        attn_weights = nn.softmax(scores, axis=-1)
        attn_output = T5SelfAttention.un_shape(mx.matmul(attn_weights, value_states))
        attn_output = self.o(attn_output)
        return attn_output

    @staticmethod
    def shape(states):
        return mx.transpose(mx.reshape(states, (1, -1, 64, 64)), (0, 2, 1, 3))

    @staticmethod
    def un_shape(states):
        return mx.reshape(mx.transpose(states, (0, 2, 1, 3)), (1, -1, 4096))

    def _compute_bias(self, seq_length):
        context_position = mx.arange(start=0, stop=seq_length, step=1)[:, None]
        memory_position = mx.arange(start=0, stop=seq_length, step=1)[None, :]
        relative_position = memory_position - context_position
        relative_position_bucket = T5SelfAttention._relative_position_bucket(relative_position)
        values = self.relative_attention_bias(relative_position_bucket)
        values = mx.transpose(values, (2, 0, 1))
        values = mx.expand_dims(values, 0)
        return values

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        relative_buckets = mx.zeros_like(relative_position)
        num_buckets //= 2
        relative_buckets += mx.where(relative_position > 0, num_buckets, 0)
        relative_position = mx.abs(relative_position)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_position_if_large = max_exact + mx.floor(
            mx.log(relative_position.astype(mx.float32) / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).astype(mx.int32)
        relative_position_if_large = mx.minimum(
            relative_position_if_large,
            mx.full(relative_position_if_large.shape, num_buckets - 1)
        )

        relative_buckets += mx.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets
