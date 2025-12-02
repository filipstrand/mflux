"""
Declarative weight mapping for Z-Image S3-DiT transformer and Qwen3 text encoder.

Maps HuggingFace weight names to MLX model structure paths.
"""

from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


class ZImageWeightMapping(WeightMapping):
    """Weight mappings for Z-Image model components."""

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        """Get transformer weight mappings (S3-DiT)."""
        return [
            # ========== Input Embeddings ==========
            # Patch embedder: all_x_embedder.2-1 → x_embedder.proj
            WeightTarget(
                mlx_path="x_embedder.proj.weight",
                hf_patterns=["all_x_embedder.2-1.weight"],
            ),
            WeightTarget(
                mlx_path="x_embedder.proj.bias",
                hf_patterns=["all_x_embedder.2-1.bias"],
            ),
            # Caption embedder: cap_embedder.0 → cap_embedder.linear1, cap_embedder.1 → cap_embedder.linear2
            WeightTarget(
                mlx_path="cap_embedder.linear1.weight",
                hf_patterns=["cap_embedder.0.weight"],
            ),
            WeightTarget(
                mlx_path="cap_embedder.linear2.weight",
                hf_patterns=["cap_embedder.1.weight"],
            ),
            WeightTarget(
                mlx_path="cap_embedder.linear2.bias",
                hf_patterns=["cap_embedder.1.bias"],
            ),
            # Timestep embedder MLP (nn.Sequential with Linear, SiLU, Linear)
            # HF: t_embedder.mlp.0 → layers.0, t_embedder.mlp.2 → layers.2
            WeightTarget(
                mlx_path="t_embedder.mlp.layers.0.weight",
                hf_patterns=["t_embedder.mlp.0.weight"],
            ),
            WeightTarget(
                mlx_path="t_embedder.mlp.layers.0.bias",
                hf_patterns=["t_embedder.mlp.0.bias"],
            ),
            WeightTarget(
                mlx_path="t_embedder.mlp.layers.2.weight",
                hf_patterns=["t_embedder.mlp.2.weight"],
            ),
            WeightTarget(
                mlx_path="t_embedder.mlp.layers.2.bias",
                hf_patterns=["t_embedder.mlp.2.bias"],
            ),
            # ========== Context Refiner (2 layers) ==========
            # Attention norms
            WeightTarget(
                mlx_path="context_refiner.{block}.attention_norm1.weight",
                hf_patterns=["context_refiner.{block}.attention_norm1.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="context_refiner.{block}.attention_norm2.weight",
                hf_patterns=["context_refiner.{block}.attention_norm2.weight"],
                max_blocks=2,
            ),
            # FFN norms
            WeightTarget(
                mlx_path="context_refiner.{block}.ffn_norm1.weight",
                hf_patterns=["context_refiner.{block}.ffn_norm1.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="context_refiner.{block}.ffn_norm2.weight",
                hf_patterns=["context_refiner.{block}.ffn_norm2.weight"],
                max_blocks=2,
            ),
            # Attention QKV projections
            WeightTarget(
                mlx_path="context_refiner.{block}.attn.to_q.weight",
                hf_patterns=["context_refiner.{block}.attention.to_q.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="context_refiner.{block}.attn.to_k.weight",
                hf_patterns=["context_refiner.{block}.attention.to_k.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="context_refiner.{block}.attn.to_v.weight",
                hf_patterns=["context_refiner.{block}.attention.to_v.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="context_refiner.{block}.attn.to_out.weight",
                hf_patterns=["context_refiner.{block}.attention.to_out.0.weight"],
                max_blocks=2,
            ),
            # Attention QK norms
            WeightTarget(
                mlx_path="context_refiner.{block}.attn.norm_q.weight",
                hf_patterns=["context_refiner.{block}.attention.norm_q.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="context_refiner.{block}.attn.norm_k.weight",
                hf_patterns=["context_refiner.{block}.attention.norm_k.weight"],
                max_blocks=2,
            ),
            # SwiGLU FFN (w1=gate, w2=down, w3=up)
            WeightTarget(
                mlx_path="context_refiner.{block}.ff.w1.weight",
                hf_patterns=["context_refiner.{block}.feed_forward.w1.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="context_refiner.{block}.ff.w2.weight",
                hf_patterns=["context_refiner.{block}.feed_forward.w2.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="context_refiner.{block}.ff.w3.weight",
                hf_patterns=["context_refiner.{block}.feed_forward.w3.weight"],
                max_blocks=2,
            ),
            # ========== Noise Refiner (2 layers) ==========
            # Uses S3DiTBlock structure (same as main transformer blocks)
            # AdaLN modulation
            WeightTarget(
                mlx_path="noise_refiner.{block}.adaLN.weight",
                hf_patterns=["noise_refiner.{block}.adaLN_modulation.0.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="noise_refiner.{block}.adaLN.bias",
                hf_patterns=["noise_refiner.{block}.adaLN_modulation.0.bias"],
                max_blocks=2,
            ),
            # Attention layer norms
            WeightTarget(
                mlx_path="noise_refiner.{block}.attention_norm1.weight",
                hf_patterns=["noise_refiner.{block}.attention_norm1.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="noise_refiner.{block}.attention_norm2.weight",
                hf_patterns=["noise_refiner.{block}.attention_norm2.weight"],
                max_blocks=2,
            ),
            # Attention QKV projections
            WeightTarget(
                mlx_path="noise_refiner.{block}.attn.to_q.weight",
                hf_patterns=["noise_refiner.{block}.attention.to_q.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="noise_refiner.{block}.attn.to_k.weight",
                hf_patterns=["noise_refiner.{block}.attention.to_k.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="noise_refiner.{block}.attn.to_v.weight",
                hf_patterns=["noise_refiner.{block}.attention.to_v.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="noise_refiner.{block}.attn.to_out.weight",
                hf_patterns=["noise_refiner.{block}.attention.to_out.0.weight"],
                max_blocks=2,
            ),
            # QK norms
            WeightTarget(
                mlx_path="noise_refiner.{block}.attn.norm_q.weight",
                hf_patterns=["noise_refiner.{block}.attention.norm_q.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="noise_refiner.{block}.attn.norm_k.weight",
                hf_patterns=["noise_refiner.{block}.attention.norm_k.weight"],
                max_blocks=2,
            ),
            # FFN layer norms
            WeightTarget(
                mlx_path="noise_refiner.{block}.ffn_norm1.weight",
                hf_patterns=["noise_refiner.{block}.ffn_norm1.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="noise_refiner.{block}.ffn_norm2.weight",
                hf_patterns=["noise_refiner.{block}.ffn_norm2.weight"],
                max_blocks=2,
            ),
            # SwiGLU FFN
            WeightTarget(
                mlx_path="noise_refiner.{block}.ff.w1.weight",
                hf_patterns=["noise_refiner.{block}.feed_forward.w1.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="noise_refiner.{block}.ff.w2.weight",
                hf_patterns=["noise_refiner.{block}.feed_forward.w2.weight"],
                max_blocks=2,
            ),
            WeightTarget(
                mlx_path="noise_refiner.{block}.ff.w3.weight",
                hf_patterns=["noise_refiner.{block}.feed_forward.w3.weight"],
                max_blocks=2,
            ),
            # ========== Main Transformer Blocks (30 layers) ==========
            # HF uses "layers.{block}", model uses "transformer_blocks.{block}"
            # AdaLN modulation (nn.Linear, not nested)
            WeightTarget(
                mlx_path="transformer_blocks.{block}.adaLN.weight",
                hf_patterns=["layers.{block}.adaLN_modulation.0.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.adaLN.bias",
                hf_patterns=["layers.{block}.adaLN_modulation.0.bias"],
            ),
            # Attention layer norms
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attention_norm1.weight",
                hf_patterns=["layers.{block}.attention_norm1.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attention_norm2.weight",
                hf_patterns=["layers.{block}.attention_norm2.weight"],
            ),
            # Attention QKV projections
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_q.weight",
                hf_patterns=["layers.{block}.attention.to_q.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_k.weight",
                hf_patterns=["layers.{block}.attention.to_k.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_v.weight",
                hf_patterns=["layers.{block}.attention.to_v.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.to_out.weight",
                hf_patterns=["layers.{block}.attention.to_out.0.weight"],
            ),
            # QK norms
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.norm_q.weight",
                hf_patterns=["layers.{block}.attention.norm_q.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.attn.norm_k.weight",
                hf_patterns=["layers.{block}.attention.norm_k.weight"],
            ),
            # FFN layer norms
            WeightTarget(
                mlx_path="transformer_blocks.{block}.ffn_norm1.weight",
                hf_patterns=["layers.{block}.ffn_norm1.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.ffn_norm2.weight",
                hf_patterns=["layers.{block}.ffn_norm2.weight"],
            ),
            # SwiGLU FFN (w1=gate, w2=down, w3=up)
            WeightTarget(
                mlx_path="transformer_blocks.{block}.ff.w1.weight",
                hf_patterns=["layers.{block}.feed_forward.w1.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.ff.w2.weight",
                hf_patterns=["layers.{block}.feed_forward.w2.weight"],
            ),
            WeightTarget(
                mlx_path="transformer_blocks.{block}.ff.w3.weight",
                hf_patterns=["layers.{block}.feed_forward.w3.weight"],
            ),
            # ========== Final Layer ==========
            # AdaLN modulation
            WeightTarget(
                mlx_path="final_layer.adaLN.weight",
                hf_patterns=["all_final_layer.2-1.adaLN_modulation.1.weight"],
            ),
            WeightTarget(
                mlx_path="final_layer.adaLN.bias",
                hf_patterns=["all_final_layer.2-1.adaLN_modulation.1.bias"],
            ),
            # LayerNorm
            WeightTarget(
                mlx_path="final_layer.norm.weight",
                hf_patterns=["all_final_layer.2-1.norm.weight"],
                required=False,  # May not exist in all checkpoints
            ),
            # Output linear
            WeightTarget(
                mlx_path="final_layer.linear.weight",
                hf_patterns=["all_final_layer.2-1.linear.weight"],
            ),
            WeightTarget(
                mlx_path="final_layer.linear.bias",
                hf_patterns=["all_final_layer.2-1.linear.bias"],
            ),
        ]

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        """Get text encoder weight mappings (Qwen3-4B)."""
        return [
            # Token embeddings
            WeightTarget(
                mlx_path="embed_tokens.weight",
                hf_patterns=["model.embed_tokens.weight"],
            ),
            # Decoder layers (36 layers)
            # Self-attention projections
            WeightTarget(
                mlx_path="layers.{layer}.self_attn.q_proj.weight",
                hf_patterns=["model.layers.{layer}.self_attn.q_proj.weight"],
            ),
            WeightTarget(
                mlx_path="layers.{layer}.self_attn.k_proj.weight",
                hf_patterns=["model.layers.{layer}.self_attn.k_proj.weight"],
            ),
            WeightTarget(
                mlx_path="layers.{layer}.self_attn.v_proj.weight",
                hf_patterns=["model.layers.{layer}.self_attn.v_proj.weight"],
            ),
            WeightTarget(
                mlx_path="layers.{layer}.self_attn.o_proj.weight",
                hf_patterns=["model.layers.{layer}.self_attn.o_proj.weight"],
            ),
            # QK norms
            WeightTarget(
                mlx_path="layers.{layer}.self_attn.q_norm.weight",
                hf_patterns=["model.layers.{layer}.self_attn.q_norm.weight"],
            ),
            WeightTarget(
                mlx_path="layers.{layer}.self_attn.k_norm.weight",
                hf_patterns=["model.layers.{layer}.self_attn.k_norm.weight"],
            ),
            # MLP projections
            WeightTarget(
                mlx_path="layers.{layer}.mlp.gate_proj.weight",
                hf_patterns=["model.layers.{layer}.mlp.gate_proj.weight"],
            ),
            WeightTarget(
                mlx_path="layers.{layer}.mlp.up_proj.weight",
                hf_patterns=["model.layers.{layer}.mlp.up_proj.weight"],
            ),
            WeightTarget(
                mlx_path="layers.{layer}.mlp.down_proj.weight",
                hf_patterns=["model.layers.{layer}.mlp.down_proj.weight"],
            ),
            # Layer norms
            WeightTarget(
                mlx_path="layers.{layer}.input_layernorm.weight",
                hf_patterns=["model.layers.{layer}.input_layernorm.weight"],
            ),
            WeightTarget(
                mlx_path="layers.{layer}.post_attention_layernorm.weight",
                hf_patterns=["model.layers.{layer}.post_attention_layernorm.weight"],
            ),
            # Final norm
            WeightTarget(
                mlx_path="norm.weight",
                hf_patterns=["model.norm.weight"],
            ),
        ]

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        """Get VAE weight mappings.

        Z-Image uses the same VAE as FLUX, so we can reuse FLUX VAE weights directly.
        The VAE structure matches, so this returns an empty list (direct mapping).
        """
        # VAE weights can be loaded directly without remapping
        # They match FLUX VAE structure
        return []
