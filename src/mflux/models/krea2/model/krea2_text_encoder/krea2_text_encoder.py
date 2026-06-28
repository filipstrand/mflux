import mlx.core as mx
from mlx import nn

from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder_layer import Qwen3VLDecoderLayer
from mflux.models.common_models.qwen3_vl.qwen3_vl_rms_norm import Qwen3VLRMSNorm
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_rotary_embedding import Qwen3TextRotaryEmbedding


class Krea2TextEncoder(nn.Module):
    """Qwen3-VL language model (text-only) as used by Krea 2.

    Differs from the Flux 2 Qwen3 encoder in two Krea-specific ways:
    - rope_theta = 5,000,000 (Krea 2 text config), not 1,000,000.
    - position ids are the *cumulative count of valid tokens* (padding does not advance a position),
      passed in explicitly. Krea 2 pads in the middle of the chat template, so the assistant suffix
      must sit at the position right after the prompt, not at ~max_length. For text-only input all 3
      mRoPE axes share these positions, so the interleaved mRoPE collapses to plain 1D RoPE on them.
    """

    def __init__(
        self,
        vocab_size: int = 151936,
        hidden_size: int = 2560,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        intermediate_size: int = 9728,
        max_position_embeddings: int = 262144,
        rope_theta: float = 5000000.0,
        rms_norm_eps: float = 1e-6,
        head_dim: int = 128,
        attention_bias: bool = False,
    ):
        super().__init__()
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
                mrope_section=None,
                attention_bias=attention_bias,
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
        output_hidden_states: bool = True,
    ) -> list[mx.array]:
        batch_size, seq_len = input_ids.shape
        hidden_states = self.embed_tokens(input_ids)

        if attention_mask is None:
            attention_mask = mx.ones((batch_size, seq_len), dtype=mx.int32)

        mask_dtype = hidden_states.dtype
        padding_mask = mx.where(
            attention_mask == 1,
            mx.zeros(attention_mask.shape, dtype=mask_dtype),
            mx.full(attention_mask.shape, -float("inf"), dtype=mask_dtype),
        )
        padding_mask = mx.expand_dims(mx.expand_dims(padding_mask, axis=1), axis=1)

        if seq_len == 1:
            causal_tri_mask = mx.zeros((batch_size, 1, 1, 1), dtype=mask_dtype)
        else:
            idx = mx.arange(seq_len, dtype=mx.int32)
            j = mx.expand_dims(idx, axis=0)
            i = mx.expand_dims(idx, axis=1)
            tri_bool = j > i
            zeros_2d = mx.zeros((seq_len, seq_len), dtype=mask_dtype)
            neginf_2d = mx.full((seq_len, seq_len), -float("inf"), dtype=mask_dtype)
            causal_tri_mask = mx.where(tri_bool, neginf_2d, zeros_2d)
            causal_tri_mask = mx.expand_dims(mx.expand_dims(causal_tri_mask, axis=0), axis=0)
            causal_tri_mask = mx.broadcast_to(causal_tri_mask, (batch_size, 1, seq_len, seq_len))
        attention_mask_4d = causal_tri_mask + padding_mask

        if position_ids is None:
            position_ids = mx.arange(seq_len, dtype=mx.int32)
            position_ids = mx.broadcast_to(mx.expand_dims(position_ids, axis=0), (batch_size, seq_len))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Match HF: hidden_states[0] is the embedding output.
        hidden_states_list = [hidden_states]
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask_4d,
                position_embeddings,
                past_key_value=None,
            )
            hidden_states_list.append(hidden_states)

        return hidden_states_list
