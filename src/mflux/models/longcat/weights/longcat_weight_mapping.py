"""
Weight mapping for LongCat-Image model.

LongCat-Image uses Flow Match architecture similar to FLUX with:
- 10 joint transformer blocks (cross-attention)
- 20 single transformer blocks (self-attention)
- Qwen2.5-VL text encoder
"""

from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping


class LongCatWeightMapping(WeightMapping):
    """
    Weight mapping for LongCat-Image model.

    LongCat has similar structure to FLUX but with:
    - 10 joint blocks (vs 19 in FLUX)
    - 20 single blocks (vs 38 in FLUX)
    - Different embedder structure (no guidance embeddings)
    - Qwen2.5-VL text encoder
    """

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        """
        Mapping for LongCat transformer weights.

        Structure:
        - x_embedder, context_embedder, proj_out
        - time_text_embed (timestep and pooled projections)
        - transformer_blocks (10 blocks)
        - single_transformer_blocks (20 blocks)
        """
        return [
            # Input/Output embedders
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
            # Time/text embeddings
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_1.weight",
                from_pattern=["time_text_embed.timestep_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_1.bias",
                from_pattern=["time_text_embed.timestep_embedder.linear_1.bias"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_2.weight",
                from_pattern=["time_text_embed.timestep_embedder.linear_2.weight"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.timestep_embedder.linear_2.bias",
                from_pattern=["time_text_embed.timestep_embedder.linear_2.bias"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.text_embedder.linear_1.weight",
                from_pattern=["time_text_embed.text_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.text_embedder.linear_1.bias",
                from_pattern=["time_text_embed.text_embedder.linear_1.bias"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.text_embedder.linear_2.weight",
                from_pattern=["time_text_embed.text_embedder.linear_2.weight"],
            ),
            WeightTarget(
                to_pattern="time_text_embed.text_embedder.linear_2.bias",
                from_pattern=["time_text_embed.text_embedder.linear_2.bias"],
            ),
            # Joint transformer blocks (10 blocks)
            # Norm layers
            WeightTarget(
                to_pattern="transformer_blocks.{block}.norm1.linear.weight",
                from_pattern=["transformer_blocks.{block}.norm1.linear.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.norm1.linear.bias",
                from_pattern=["transformer_blocks.{block}.norm1.linear.bias"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.norm1_context.linear.weight",
                from_pattern=["transformer_blocks.{block}.norm1_context.linear.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.norm1_context.linear.bias",
                from_pattern=["transformer_blocks.{block}.norm1_context.linear.bias"],
                max_blocks=10,
            ),
            # Attention projections (image)
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_q.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_q.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_q.bias"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_k.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_k.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_k.bias"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_v.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_v.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_v.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_v.bias"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_out.0.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_out.0.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_out.0.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_out.0.bias"],
                max_blocks=10,
            ),
            # Context (add) projections for joint attention
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_q_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_q_proj.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_q_proj.bias",
                from_pattern=["transformer_blocks.{block}.attn.add_q_proj.bias"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_k_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_k_proj.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_k_proj.bias",
                from_pattern=["transformer_blocks.{block}.attn.add_k_proj.bias"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_v_proj.weight",
                from_pattern=["transformer_blocks.{block}.attn.add_v_proj.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.add_v_proj.bias",
                from_pattern=["transformer_blocks.{block}.attn.add_v_proj.bias"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_add_out.weight",
                from_pattern=["transformer_blocks.{block}.attn.to_add_out.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.to_add_out.bias",
                from_pattern=["transformer_blocks.{block}.attn.to_add_out.bias"],
                max_blocks=10,
            ),
            # QK norms
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_q.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_k.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_added_q.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_added_q.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.attn.norm_added_k.weight",
                from_pattern=["transformer_blocks.{block}.attn.norm_added_k.weight"],
                max_blocks=10,
            ),
            # Feed-forward (image stream)
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear1.weight",
                from_pattern=["transformer_blocks.{block}.ff.net.0.proj.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear1.bias",
                from_pattern=["transformer_blocks.{block}.ff.net.0.proj.bias"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear2.weight",
                from_pattern=["transformer_blocks.{block}.ff.net.2.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff.linear2.bias",
                from_pattern=["transformer_blocks.{block}.ff.net.2.bias"],
                max_blocks=10,
            ),
            # Feed-forward context
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear1.weight",
                from_pattern=["transformer_blocks.{block}.ff_context.net.0.proj.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear1.bias",
                from_pattern=["transformer_blocks.{block}.ff_context.net.0.proj.bias"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear2.weight",
                from_pattern=["transformer_blocks.{block}.ff_context.net.2.weight"],
                max_blocks=10,
            ),
            WeightTarget(
                to_pattern="transformer_blocks.{block}.ff_context.linear2.bias",
                from_pattern=["transformer_blocks.{block}.ff_context.net.2.bias"],
                max_blocks=10,
            ),
            # Single transformer blocks (20 blocks)
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.norm.linear.weight",
                from_pattern=["single_transformer_blocks.{block}.norm.linear.weight"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.norm.linear.bias",
                from_pattern=["single_transformer_blocks.{block}.norm.linear.bias"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_q.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.to_q.weight"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_q.bias",
                from_pattern=["single_transformer_blocks.{block}.attn.to_q.bias"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_k.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.to_k.weight"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_k.bias",
                from_pattern=["single_transformer_blocks.{block}.attn.to_k.bias"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_v.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.to_v.weight"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.to_v.bias",
                from_pattern=["single_transformer_blocks.{block}.attn.to_v.bias"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.norm_q.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.norm_q.weight"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.attn.norm_k.weight",
                from_pattern=["single_transformer_blocks.{block}.attn.norm_k.weight"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.proj_mlp.weight",
                from_pattern=["single_transformer_blocks.{block}.proj_mlp.weight"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.proj_mlp.bias",
                from_pattern=["single_transformer_blocks.{block}.proj_mlp.bias"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.proj_out.weight",
                from_pattern=["single_transformer_blocks.{block}.proj_out.weight"],
                max_blocks=20,
            ),
            WeightTarget(
                to_pattern="single_transformer_blocks.{block}.proj_out.bias",
                from_pattern=["single_transformer_blocks.{block}.proj_out.bias"],
                max_blocks=20,
            ),
            # Norm out
            WeightTarget(
                to_pattern="norm_out.linear.weight",
                from_pattern=["norm_out.linear.weight"],
            ),
            WeightTarget(
                to_pattern="norm_out.linear.bias",
                from_pattern=["norm_out.linear.bias"],
            ),
        ]

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        """
        Mapping for Qwen2.5-VL text encoder weights.

        Qwen2.5-VL has:
        - Language model with 28 transformer layers
        - Vision encoder with 32 blocks (combined qkv)
        """
        mappings = [
            # Embedding
            WeightTarget(
                to_pattern="embed_tokens.weight",
                from_pattern=["model.embed_tokens.weight"],
            ),
            # Language model layers (28 layers)
            WeightTarget(
                to_pattern="layers.{block}.self_attn.q_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.q_proj.weight"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.q_proj.bias",
                from_pattern=["model.layers.{block}.self_attn.q_proj.bias"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.k_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.k_proj.weight"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.k_proj.bias",
                from_pattern=["model.layers.{block}.self_attn.k_proj.bias"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.v_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.v_proj.weight"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.v_proj.bias",
                from_pattern=["model.layers.{block}.self_attn.v_proj.bias"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.self_attn.o_proj.weight",
                from_pattern=["model.layers.{block}.self_attn.o_proj.weight"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.mlp.gate_proj.weight",
                from_pattern=["model.layers.{block}.mlp.gate_proj.weight"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.mlp.up_proj.weight",
                from_pattern=["model.layers.{block}.mlp.up_proj.weight"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.mlp.down_proj.weight",
                from_pattern=["model.layers.{block}.mlp.down_proj.weight"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.input_layernorm.weight",
                from_pattern=["model.layers.{block}.input_layernorm.weight"],
                max_blocks=28,
            ),
            WeightTarget(
                to_pattern="layers.{block}.post_attention_layernorm.weight",
                from_pattern=["model.layers.{block}.post_attention_layernorm.weight"],
                max_blocks=28,
            ),
            # Final norm
            WeightTarget(
                to_pattern="norm.weight",
                from_pattern=["model.norm.weight"],
            ),
            # Vision encoder (32 blocks with combined qkv)
            WeightTarget(
                to_pattern="visual.blocks.{block}.attn.qkv.weight",
                from_pattern=["visual.blocks.{block}.attn.qkv.weight"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.attn.qkv.bias",
                from_pattern=["visual.blocks.{block}.attn.qkv.bias"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.attn.proj.weight",
                from_pattern=["visual.blocks.{block}.attn.proj.weight"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.attn.proj.bias",
                from_pattern=["visual.blocks.{block}.attn.proj.bias"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.norm1.weight",
                from_pattern=["visual.blocks.{block}.norm1.weight"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.norm1.bias",
                from_pattern=["visual.blocks.{block}.norm1.bias"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.norm2.weight",
                from_pattern=["visual.blocks.{block}.norm2.weight"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.norm2.bias",
                from_pattern=["visual.blocks.{block}.norm2.bias"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.mlp.fc1.weight",
                from_pattern=["visual.blocks.{block}.mlp.fc1.weight"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.mlp.fc1.bias",
                from_pattern=["visual.blocks.{block}.mlp.fc1.bias"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.mlp.fc2.weight",
                from_pattern=["visual.blocks.{block}.mlp.fc2.weight"],
                max_blocks=32,
            ),
            WeightTarget(
                to_pattern="visual.blocks.{block}.mlp.fc2.bias",
                from_pattern=["visual.blocks.{block}.mlp.fc2.bias"],
                max_blocks=32,
            ),
            # Visual merger
            WeightTarget(
                to_pattern="visual.merger.mlp.0.weight",
                from_pattern=["visual.merger.mlp.0.weight"],
            ),
            WeightTarget(
                to_pattern="visual.merger.mlp.0.bias",
                from_pattern=["visual.merger.mlp.0.bias"],
            ),
            WeightTarget(
                to_pattern="visual.merger.mlp.2.weight",
                from_pattern=["visual.merger.mlp.2.weight"],
            ),
            WeightTarget(
                to_pattern="visual.merger.mlp.2.bias",
                from_pattern=["visual.merger.mlp.2.bias"],
            ),
            # Visual patch embed
            WeightTarget(
                to_pattern="visual.patch_embed.proj.weight",
                from_pattern=["visual.patch_embed.proj.weight"],
            ),
            WeightTarget(
                to_pattern="visual.patch_embed.proj.bias",
                from_pattern=["visual.patch_embed.proj.bias"],
            ),
        ]
        return mappings

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        """Reuse FLUX VAE mapping - standard AutoencoderKL."""
        return FluxWeightMapping.get_vae_mapping()
