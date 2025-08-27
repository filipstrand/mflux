import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils


class JointAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dimension = 128
        self.batch_size = 1
        self.num_heads = 24

        self.to_q = nn.Linear(3072, 3072)
        self.to_k = nn.Linear(3072, 3072)
        self.to_v = nn.Linear(3072, 3072)
        self.to_out = [nn.Linear(3072, 3072)]
        self.add_q_proj = nn.Linear(3072, 3072)
        self.add_k_proj = nn.Linear(3072, 3072)
        self.add_v_proj = nn.Linear(3072, 3072)
        self.to_add_out = nn.Linear(3072, 3072)
        self.norm_q = nn.RMSNorm(128)
        self.norm_k = nn.RMSNorm(128)
        self.norm_added_q = nn.RMSNorm(128)
        self.norm_added_k = nn.RMSNorm(128)

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb: mx.array,
    ) -> tuple[mx.array, mx.array]:
        # 1a. Compute Q,K,V for hidden_states
        query, key, value = AttentionUtils.process_qkv(
            hidden_states=hidden_states,
            to_q=self.to_q,
            to_k=self.to_k,
            to_v=self.to_v,
            norm_q=self.norm_q,
            norm_k=self.norm_k,
            num_heads=self.num_heads,
            head_dim=self.head_dimension,
        )

        # 1b. Compute Q,K,V for encoder_hidden_states
        enc_query, enc_key, enc_value = AttentionUtils.process_qkv(
            hidden_states=encoder_hidden_states,
            to_q=self.add_q_proj,
            to_k=self.add_k_proj,
            to_v=self.add_v_proj,
            norm_q=self.norm_added_q,
            norm_k=self.norm_added_k,
            num_heads=self.num_heads,
            head_dim=self.head_dimension,
        )

        # 1c. Concatenate results
        query = mx.concatenate([enc_query, query], axis=2)
        key = mx.concatenate([enc_key, key], axis=2)
        value = mx.concatenate([enc_value, value], axis=2)

        # 1d. Apply rope to Q,K
        query, key = AttentionUtils.apply_rope(xq=query, xk=key, freqs_cis=image_rotary_emb)

        # 2. Compute attention
        hidden_states = AttentionUtils.compute_attention(
            query=query,
            key=key,
            value=value,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dimension,
        )

        # 3. Separate the results
        encoder_hidden_states, hidden_states = (
            hidden_states[:, : encoder_hidden_states.shape[1]],
            hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # 4. Project the output
        hidden_states = self.to_out[0](hidden_states)
        encoder_hidden_states = self.to_add_out(encoder_hidden_states)
        return hidden_states, encoder_hidden_states
