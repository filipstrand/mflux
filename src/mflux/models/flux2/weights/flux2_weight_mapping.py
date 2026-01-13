"""Weight mappings for FLUX.2 model.

Maps HuggingFace weight names to MFLUX model attribute paths.
"""

from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


class Flux2WeightMapping(WeightMapping):
    """Weight mappings for FLUX.2 model components."""

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        """Get transformer weight mapping."""
        return [
            # Input embedders
            WeightTarget(
                to_pattern="x_embedder.weight",
                from_pattern=["x_embedder.weight"],
            ),
            WeightTarget(
                to_pattern="context_embedder.weight",
                from_pattern=["context_embedder.weight"],
            ),
            # Output projection
            WeightTarget(
                to_pattern="proj_out.weight",
                from_pattern=["proj_out.weight"],
            ),
            # Norm out (AdaLN-style output norm)
            WeightTarget(
                to_pattern="norm_out.linear.weight",
                from_pattern=["norm_out.linear.weight"],
            ),
            # Time/guidance embeddings
            WeightTarget(
                to_pattern="time_guidance_embed.timestep_embedder.linear_1.weight",
                from_pattern=["time_guidance_embed.timestep_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="time_guidance_embed.timestep_embedder.linear_2.weight",
                from_pattern=["time_guidance_embed.timestep_embedder.linear_2.weight"],
            ),
            WeightTarget(
                to_pattern="time_guidance_embed.guidance_embedder.linear_1.weight",
                from_pattern=["time_guidance_embed.guidance_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="time_guidance_embed.guidance_embedder.linear_2.weight",
                from_pattern=["time_guidance_embed.guidance_embedder.linear_2.weight"],
            ),
            # Global modulation layers
            WeightTarget(
                to_pattern="double_stream_modulation_img.linear.weight",
                from_pattern=["double_stream_modulation_img.linear.weight"],
            ),
            WeightTarget(
                to_pattern="double_stream_modulation_txt.linear.weight",
                from_pattern=["double_stream_modulation_txt.linear.weight"],
            ),
            WeightTarget(
                to_pattern="single_stream_modulation.linear.weight",
                from_pattern=["single_stream_modulation.linear.weight"],
            ),
            # Joint transformer blocks (8 blocks)
            # Attention projections
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_q.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_k.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_v.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_v.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_out.0.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_out.0.weight"],
            ),
            # Context/text projections
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_q_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_q_proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_k_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_k_proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_v_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_v_proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_add_out.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_add_out.weight"],
            ),
            # RMSNorm for Q, K
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_q.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_k.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_added_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_added_q.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_added_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_added_k.weight"],
            ),
            # Feed-forward (image stream)
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear_in.weight",
                from_pattern=["transformer_blocks.{block}.ff.linear_in.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear_out.weight",
                from_pattern=["transformer_blocks.{block}.ff.linear_out.weight"],
            ),
            # Feed-forward (context stream)
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear_in.weight",
                from_pattern=["transformer_blocks.{block}.ff_context.linear_in.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear_out.weight",
                from_pattern=["transformer_blocks.{block}.ff_context.linear_out.weight"],
            ),
            # Single transformer blocks (48 blocks)
            # Fused QKV + MLP projection
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.weight"],
            ),
            # Output projection
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_out.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.to_out.weight"],
            ),
            # RMSNorm for Q, K
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.norm_q.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.norm_q.weight"],
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.norm_k.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.norm_k.weight"],
            ),
        ]

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        """Get VAE weight mapping.

        FLUX.2 VAE is similar to FLUX.1 but with 32 latent channels.
        We reuse the FLUX.1 VAE mapping structure.
        """
        # Import FLUX.1 VAE mapping and use it as base
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping
        return FluxWeightMapping.get_vae_mapping()

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        """Get Mistral3 text encoder weight mapping."""
        return [
            # Embeddings
            WeightTarget(
                to_pattern="embed_tokens.weight",
                from_pattern=["model.embed_tokens.weight"],
            ),
            # Transformer layers
            WeightTarget(
                to_pattern="layers.{block}.input_layernorm.weight",
                from_pattern=["model.layers.{block}.input_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.post_attention_layernorm.weight",
                from_pattern=["model.layers.{block}.post_attention_layernorm.weight"],
            ),
            # Self attention
            WeightTarget(
                to_pattern="layers.{block}.self_attn.q_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.q_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.k_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.k_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.v_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.v_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.o_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.o_proj.weight"],
            ),
            # MLP
            WeightTarget(
                to_pattern="layers.{block}.mlp.gate_proj.weight",
                from_pattern=["model.layers.{block}.mlp.gate_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.mlp.up_proj.weight",
                from_pattern=["model.layers.{block}.mlp.up_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{block}.mlp.down_proj.weight",
                from_pattern=["model.layers.{block}.mlp.down_proj.weight"],
            ),
            # Final norm
            WeightTarget(
                to_pattern="norm.weight",
                from_pattern=["model.norm.weight"],
            ),
            # Output projection (for joint_attention_dim)
            # Mistral3 output projects hidden_size (5120) -> joint_attention_dim (15360)
            # Pattern verified against HuggingFace black-forest-labs/FLUX.2-dev
            WeightTarget(
                to_pattern="output_proj.weight",
                from_pattern=[
                    "output_projection.weight",  # Standard naming
                    "model.output_projection.weight",  # Alternative with model prefix
                    "lm_head.weight",  # Some implementations use lm_head
                ],
                required=False,  # May not exist if projection is handled separately
            ),
        ]
