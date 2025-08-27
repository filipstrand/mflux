import mlx.core as mx
from mlx import nn

from mflux.models.flux.model.flux_transformer.common.attention_utils import AttentionUtils


class JointAttentionConcept(nn.Module):
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
        encoder_hidden_states_concept: mx.array,
        image_rotary_emb: mx.array,
        image_rotary_emb_concept: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        # Compute Q,K,V for hidden_states once (shared between both attention computations)
        image_query, image_key, image_value = AttentionUtils.process_qkv(
            hidden_states=hidden_states,
            to_q=self.to_q,
            to_k=self.to_k,
            to_v=self.to_v,
            norm_q=self.norm_q,
            norm_k=self.norm_k,
            num_heads=self.num_heads,
            head_dim=self.head_dimension,
        )

        # 1: Regular joint attention
        hidden_states_final, encoder_hidden_states_final, img_attn, _ = self._compute_joint_attention_optimized(
            image_query=image_query,
            image_key=image_key,
            image_value=image_value,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # 2: Concept-specific attention
        _, encoder_hidden_states_concept_final, _, concept_attn = self._compute_joint_attention_optimized(
            image_query=image_query,
            image_key=image_key,
            image_value=image_value,
            encoder_hidden_states=encoder_hidden_states_concept,
            image_rotary_emb=image_rotary_emb_concept,
        )

        return (
            hidden_states_final,
            encoder_hidden_states_final,
            encoder_hidden_states_concept_final,
            img_attn,
            concept_attn,
        )

    def _compute_joint_attention_optimized(
        self,
        image_query: mx.array,
        image_key: mx.array,
        image_value: mx.array,
        encoder_hidden_states: mx.array,
        image_rotary_emb: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        # 1. Compute Q,K,V for encoder_hidden_states
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

        # 2. Concatenate results (using pre-computed hidden states QKV)
        joint_query = mx.concatenate([enc_query, image_query], axis=2)
        joint_key = mx.concatenate([enc_key, image_key], axis=2)
        joint_value = mx.concatenate([enc_value, image_value], axis=2)

        # 3. Apply rope to Q,K
        joint_query, joint_key = AttentionUtils.apply_rope(
            xq=joint_query,
            xk=joint_key,
            freqs_cis=image_rotary_emb,
        )

        # 4. Compute attention
        joint_hidden_states = AttentionUtils.compute_attention(
            query=joint_query,
            key=joint_key,
            value=joint_value,
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            head_dim=self.head_dimension,
        )

        # 5. Separate the results
        encoder_output, hidden_states_output = (
            joint_hidden_states[:, : encoder_hidden_states.shape[1]],
            joint_hidden_states[:, encoder_hidden_states.shape[1] :],
        )

        # 6. Project outputs
        hidden_states_final = self.to_out[0](hidden_states_output)
        encoder_hidden_states_final = self.to_add_out(encoder_output)

        return hidden_states_final, encoder_hidden_states_final, hidden_states_output, encoder_output
