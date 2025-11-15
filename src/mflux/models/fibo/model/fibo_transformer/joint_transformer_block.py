import mlx.core as mx
from mlx import nn

from mflux.models.fibo.model.fibo_transformer.feed_forward import FiboFeedForward
from mflux.models.fibo.model.fibo_transformer.fibo_ada_layer_norm_zero import FiboAdaLayerNormZero
from mflux.models.fibo.model.fibo_transformer.fibo_attention import FiboJointAttention


class FiboJointTransformerBlock(nn.Module):
    """
    MLX port of diffusers.models.transformers.transformer_bria_fibo.BriaFiboTransformerBlock
    (dual-stream block: image + context).

    This reuses the Flux AdaLayerNormZero, FeedForward and JointAttention implementations
    but matches the FIBO block math (LayerNorm dims, gating, residuals, fp16 clipping).
    """

    def __init__(self, dim: int, num_attention_heads: int, attention_head_dim: int, eps: float = 1e-6):
        super().__init__()

        # AdaLayerNormZero for hidden and context streams
        self.norm1 = FiboAdaLayerNormZero(embedding_dim=dim)
        self.norm1_context = FiboAdaLayerNormZero(embedding_dim=dim)

        # BriaFibo-style joint attention (image + context)
        self.attn = FiboJointAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
        )

        # FFNs and LayerNorms (FIBO-specific FeedForward matching diffusers.FeedForward)
        self.norm2 = nn.LayerNorm(dims=dim, eps=eps, affine=False)
        self.ff = FiboFeedForward(dim=dim, dim_out=dim, mult=4, activation_fn="gelu-approximate")

        self.norm2_context = nn.LayerNorm(dims=dim, eps=eps, affine=False)
        self.ff_context = FiboFeedForward(dim=dim, dim_out=dim, mult=4, activation_fn="gelu-approximate")

        # Debug flag so we only log the first joint block invocation
        self._debug_logged_attn = False

    def __call__(
        self,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        temb: mx.array,
        image_rotary_emb: mx.array,
        attention_mask: mx.array | None = None,
    ) -> tuple[mx.array, mx.array]:
        # 1. AdaLayerNormZero for both streams
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states=hidden_states,
            text_embeddings=temb,
        )

        norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
            hidden_states=encoder_hidden_states,
            text_embeddings=temb,
        )

        # Debug: AdaLayerNormZero inputs/outputs for the first joint block
        if not self._debug_logged_attn:
            from mflux_debugger.tensor_debug import debug_save

            try:
                # High-level tensors we already compare
                debug_save(temb, "mlx_joint_block0_temb")
                debug_save(norm_hidden_states, "mlx_joint_block0_norm_hidden")
                debug_save(gate_msa, "mlx_joint_block0_gate_msa")
                debug_save(shift_mlp, "mlx_joint_block0_shift_mlp")
                debug_save(scale_mlp, "mlx_joint_block0_scale_mlp")
                debug_save(gate_mlp, "mlx_joint_block0_gate_mlp")

                debug_save(norm_encoder_hidden_states, "mlx_joint_block0_norm_context")
                debug_save(c_gate_msa, "mlx_joint_block0_c_gate_msa")
                debug_save(c_shift_mlp, "mlx_joint_block0_c_shift_mlp")
                debug_save(c_scale_mlp, "mlx_joint_block0_c_scale_mlp")
                debug_save(c_gate_mlp, "mlx_joint_block0_c_gate_mlp")

                # Finer-grained view inside AdaLayerNormZero for this first block:
                #  - raw LayerNorm outputs before modulation
                #  - shift/scale used for the MSA stream
                emb_hidden = self.norm1.linear(nn.silu(temb))
                chunk_size = emb_hidden.shape[1] // 6
                shift_msa = emb_hidden[:, 0 * chunk_size : 1 * chunk_size]
                scale_msa = emb_hidden[:, 1 * chunk_size : 2 * chunk_size]

                emb_context = self.norm1_context.linear(nn.silu(temb))
                c_shift_msa = emb_context[:, 0 * chunk_size : 1 * chunk_size]
                c_scale_msa = emb_context[:, 1 * chunk_size : 2 * chunk_size]

                # Raw LayerNorm outputs before modulation, using the
                # explicit PyTorch-style layer norm in FiboAdaLayerNormZero.
                norm_hidden_only = self.norm1._layer_norm(hidden_states)
                norm_context_only = self.norm1_context._layer_norm(encoder_hidden_states)

                debug_save(norm_hidden_only, "mlx_joint_block0_norm_hidden_only")
                debug_save(norm_context_only, "mlx_joint_block0_norm_context_only")
                debug_save(shift_msa, "mlx_joint_block0_shift_msa")
                debug_save(scale_msa, "mlx_joint_block0_scale_msa")
                debug_save(c_shift_msa, "mlx_joint_block0_c_shift_msa_only")
                debug_save(c_scale_msa, "mlx_joint_block0_c_scale_msa_only")
            except Exception:  # noqa: BLE001
                pass

        # 2. Joint attention over image + context streams.
        # IP-Adapter path is intentionally ignored here; FiboJointAttention returns 2 tensors.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            attention_mask=attention_mask,
        )

        # Debug + optional PT-attention injection for the first joint block only.
        if not self._debug_logged_attn:
            from mflux_debugger.tensor_debug import debug_load, debug_save

            try:
                # Save MLX raw attention outputs for comparison
                debug_save(attn_output, "mlx_joint_block0_attn_hidden_raw")
                debug_save(context_attn_output, "mlx_joint_block0_attn_context_raw")

                # Optional: load PyTorch raw attention outputs and override, to test downstream parity.
                pt_attn_hidden = debug_load("pt_joint_block0_attn_hidden_raw")
                pt_attn_context = debug_load("pt_joint_block0_attn_context_raw")
                if (
                    pt_attn_hidden is not None
                    and pt_attn_context is not None
                    and pt_attn_hidden.shape == attn_output.shape
                    and pt_attn_context.shape == context_attn_output.shape
                ):
                    attn_output = pt_attn_hidden.astype(attn_output.dtype)
                    context_attn_output = pt_attn_context.astype(context_attn_output.dtype)
            except Exception:  # noqa: BLE001
                pass
            self._debug_logged_attn = True

        # 3a. Process attention outputs for the image stream.
        attn_output = mx.expand_dims(gate_msa, axis=1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = mx.expand_dims(gate_mlp, axis=1) * ff_output

        hidden_states = hidden_states + ff_output

        # 3b. Process attention outputs for the context stream.
        context_attn_output = mx.expand_dims(c_gate_msa, axis=1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + mx.expand_dims(c_gate_mlp, axis=1) * context_ff_output

        # 4. FP16 clipping for numerical parity with PyTorch.
        if encoder_hidden_states.dtype == mx.float16:
            encoder_hidden_states = mx.clip(encoder_hidden_states, -65504.0, 65504.0)

        return encoder_hidden_states, hidden_states
