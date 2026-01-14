from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping


class ChromaWeightMapping(WeightMapping):
    """
    Weight mapping for Chroma model.

    Key differences from FLUX:
    1. Uses DistilledGuidanceLayer instead of TimeTextEmbed
    2. No AdaLayerNorm in transformer blocks (no norm1, norm1_context weights)
    3. Single blocks don't have norm.linear weights
    4. No norm_out.linear weights - modulations applied directly
    """

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        """
        Mapping for Chroma transformer weights.

        Chroma has:
        - x_embedder, context_embedder, proj_out (same as FLUX)
        - distilled_guidance_layer (replaces time_text_embed)
        - transformer_blocks without norm1/norm1_context (19 blocks)
        - single_transformer_blocks without norm.linear (38 blocks)
        """
        return [
            # Shared embedders (same as FLUX)
            WeightTarget(
                to_pattern="x_embedder.weight",
                from_pattern=["x_embedder.weight"],
            ),
            WeightTarget(
                to_pattern="x_embedder.bias",
                from_pattern=["x_embedder.bias"],
            ),
            WeightTarget(
                to_pattern="context_embedder.weight",
                from_pattern=["context_embedder.weight"],
            ),
            WeightTarget(
                to_pattern="context_embedder.bias",
                from_pattern=["context_embedder.bias"],
            ),
            WeightTarget(
                to_pattern="proj_out.weight",
                from_pattern=["proj_out.weight"],
            ),
            WeightTarget(
                to_pattern="proj_out.bias",
                from_pattern=["proj_out.bias"],
            ),
            # DistilledGuidanceLayer (Chroma-specific, replaces TimeTextEmbed)
            # in_proj: Linear(64, 5120)
            WeightTarget(
                to_pattern="distilled_guidance_layer.in_proj.weight",
                from_pattern=["distilled_guidance_layer.in_proj.weight"],
            ),
            WeightTarget(
                to_pattern="distilled_guidance_layer.in_proj.bias",
                from_pattern=["distilled_guidance_layer.in_proj.bias"],
            ),
            # layers.{0-4}: each has linear_1 and linear_2 (5 layers)
            WeightTarget(
                to_pattern="distilled_guidance_layer.layers.{block}.linear_1.weight",
                from_pattern=["distilled_guidance_layer.layers.{block}.linear_1.weight"],
                max_blocks=5,
            ),
            WeightTarget(
                to_pattern="distilled_guidance_layer.layers.{block}.linear_1.bias",
                from_pattern=["distilled_guidance_layer.layers.{block}.linear_1.bias"],
                max_blocks=5,
            ),
            WeightTarget(
                to_pattern="distilled_guidance_layer.layers.{block}.linear_2.weight",
                from_pattern=["distilled_guidance_layer.layers.{block}.linear_2.weight"],
                max_blocks=5,
            ),
            WeightTarget(
                to_pattern="distilled_guidance_layer.layers.{block}.linear_2.bias",
                from_pattern=["distilled_guidance_layer.layers.{block}.linear_2.bias"],
                max_blocks=5,
            ),
            # norms.{0-4}: RMSNorm (weight only, no bias)
            WeightTarget(
                to_pattern="distilled_guidance_layer.norms.{block}.weight",
                from_pattern=["distilled_guidance_layer.norms.{block}.weight"],
                max_blocks=5,
            ),
            # out_proj: Linear(5120, 3072) -> outputs 344 modulation channels
            WeightTarget(
                to_pattern="distilled_guidance_layer.out_proj.weight",
                from_pattern=["distilled_guidance_layer.out_proj.weight"],
            ),
            WeightTarget(
                to_pattern="distilled_guidance_layer.out_proj.bias",
                from_pattern=["distilled_guidance_layer.out_proj.bias"],
            ),
            # Joint transformer blocks (19 blocks)
            # Note: Chroma does NOT have norm1 or norm1_context weights
            # Attention projections
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_q.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_q.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_q.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_k.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_k.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_k.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_v.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_v.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_v.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_v.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_out.0.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_out.0.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_out.0.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_out.0.bias"],
            ),
            # Context (add) projections for joint attention
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_q_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_q_proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_q_proj.bias",
                from_pattern=["transformer_blocks.{block}.attn.add_q_proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_k_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_k_proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_k_proj.bias",
                from_pattern=["transformer_blocks.{block}.attn.add_k_proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_v_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_v_proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_v_proj.bias",
                from_pattern=["transformer_blocks.{block}.attn.add_v_proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_add_out.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_add_out.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_add_out.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_add_out.bias"],
            ),
            # QK norms (RMSNorm-style, weight only)
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
            # Feed-forward (hidden stream)
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear1.weight",
                from_pattern=["transformer_blocks.{block}.ff.net.0.proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear1.bias",
                from_pattern=["transformer_blocks.{block}.ff.net.0.proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear2.weight",
                from_pattern=["transformer_blocks.{block}.ff.net.2.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear2.bias",
                from_pattern=["transformer_blocks.{block}.ff.net.2.bias"],
            ),
            # Feed-forward context
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear1.weight",
                from_pattern=["transformer_blocks.{block}.ff_context.net.0.proj.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear1.bias",
                from_pattern=["transformer_blocks.{block}.ff_context.net.0.proj.bias"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear2.weight",
                from_pattern=["transformer_blocks.{block}.ff_context.net.2.weight"],
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear2.bias",
                from_pattern=["transformer_blocks.{block}.ff_context.net.2.bias"],
            ),
            # Single transformer blocks (38 blocks)
            # Note: Chroma single blocks do NOT have norm.linear weights
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_q.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.to_q.weight"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_q.bias",
                from_pattern=["single_transformer_blocks.{block}.attn.to_q.bias"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_k.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.to_k.weight"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_k.bias",
                from_pattern=["single_transformer_blocks.{block}.attn.to_k.bias"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_v.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.to_v.weight"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_v.bias",
                from_pattern=["single_transformer_blocks.{block}.attn.to_v.bias"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.norm_q.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.norm_q.weight"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.norm_k.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.norm_k.weight"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.proj_mlp.weight",
                from_pattern=["single_transformer_blocks.{block}.proj_mlp.weight"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.proj_mlp.bias",
                from_pattern=["single_transformer_blocks.{block}.proj_mlp.bias"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.proj_out.weight",
                from_pattern=["single_transformer_blocks.{block}.proj_out.weight"],
                max_blocks=38,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.proj_out.bias",
                from_pattern=["single_transformer_blocks.{block}.proj_out.bias"],
                max_blocks=38,
            ),
        ]

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        """Reuse FLUX VAE mapping - identical architecture."""
        return FluxWeightMapping.get_vae_mapping()

    @staticmethod
    def get_t5_encoder_mapping() -> List[WeightTarget]:
        """Reuse FLUX T5 encoder mapping - identical architecture."""
        return FluxWeightMapping.get_t5_encoder_mapping()
