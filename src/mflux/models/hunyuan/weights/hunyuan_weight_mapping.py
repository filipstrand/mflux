"""Weight mapping for Hunyuan-DiT model.

Maps HuggingFace Diffusers weight names to MFLUX model structure.

Hunyuan-DiT architecture:
- 28 DiT blocks with self-attention, cross-attention, and FFN
- Dual text encoders (Chinese CLIP + mT5)
- Standard 4-channel VAE
- AdaLN conditioning for self-attention only
"""

from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


class HunyuanWeightMapping(WeightMapping):
    """Weight mapping for Hunyuan-DiT model."""

    NUM_BLOCKS = 28

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        """
        Mapping for Hunyuan-DiT transformer weights.

        HunyuanDiT2DModel structure:
        - time_embed: timestep embedding MLP
        - patch_embed: image patch embedding
        - text_proj: text encoder projection
        - blocks.0-27: DiT transformer blocks
        - norm_out: final AdaLN
        - proj_out: output projection
        """
        mappings = []

        # Time embedding (timestep conditioning)
        # Maps from time_extra_emb.timestep_embedder
        mappings.extend([
            WeightTarget(
                to_pattern="time_embed.linear_1.weight",
                from_pattern=["time_extra_emb.timestep_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="time_embed.linear_1.bias",
                from_pattern=["time_extra_emb.timestep_embedder.linear_1.bias"],
            ),
            WeightTarget(
                to_pattern="time_embed.linear_2.weight",
                from_pattern=["time_extra_emb.timestep_embedder.linear_2.weight"],
            ),
            WeightTarget(
                to_pattern="time_embed.linear_2.bias",
                from_pattern=["time_extra_emb.timestep_embedder.linear_2.bias"],
            ),
        ])

        # Patch embedding (image to patches)
        mappings.extend([
            WeightTarget(
                to_pattern="patch_embed.proj.weight",
                from_pattern=["pos_embed.proj.weight"],
            ),
            WeightTarget(
                to_pattern="patch_embed.proj.bias",
                from_pattern=["pos_embed.proj.bias"],
            ),
        ])

        # Text projector (T5 text embedder with 2 layers)
        # The text_embedder in HF takes T5 output (2048) -> 8192 -> 1024
        mappings.extend([
            WeightTarget(
                to_pattern="text_proj.linear_1.weight",
                from_pattern=["text_embedder.linear_1.weight"],
            ),
            WeightTarget(
                to_pattern="text_proj.linear_1.bias",
                from_pattern=["text_embedder.linear_1.bias"],
            ),
            WeightTarget(
                to_pattern="text_proj.linear_2.weight",
                from_pattern=["text_embedder.linear_2.weight"],
            ),
            WeightTarget(
                to_pattern="text_proj.linear_2.bias",
                from_pattern=["text_embedder.linear_2.bias"],
            ),
        ])

        # DiT transformer blocks (28 blocks)
        for block_idx in range(HunyuanWeightMapping.NUM_BLOCKS):
            mappings.extend(HunyuanWeightMapping._get_dit_block_mapping(block_idx))

        # Output layers - norm_out is an AdaLN
        mappings.extend([
            WeightTarget(
                to_pattern="norm_out.linear.weight",
                from_pattern=["norm_out.linear.weight"],
            ),
            WeightTarget(
                to_pattern="norm_out.linear.bias",
                from_pattern=["norm_out.linear.bias"],
            ),
            WeightTarget(
                to_pattern="proj_out.weight",
                from_pattern=["proj_out.weight"],
            ),
            WeightTarget(
                to_pattern="proj_out.bias",
                from_pattern=["proj_out.bias"],
            ),
        ])

        return mappings

    @staticmethod
    def _get_dit_block_mapping(block_idx: int) -> List[WeightTarget]:
        """
        Get weight mapping for a single DiT block.

        Each block has:
        1. norm1 (AdaLN) + attn1 (self-attention)
        2. norm2 (simple LN) + attn2 (cross-attention)
        3. norm3 (simple LN) + ff (feed-forward)
        """
        prefix_to = f"blocks.{block_idx}"
        prefix_from = f"blocks.{block_idx}"

        mappings = []

        # AdaLN for self-attention (norm1) - has .norm and .linear sublayers
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.norm1.norm.weight",
                from_pattern=[f"{prefix_from}.norm1.norm.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.norm1.norm.bias",
                from_pattern=[f"{prefix_from}.norm1.norm.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.norm1.linear.weight",
                from_pattern=[f"{prefix_from}.norm1.linear.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.norm1.linear.bias",
                from_pattern=[f"{prefix_from}.norm1.linear.bias"],
            ),
        ])

        # Self-attention (attn1)
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.to_q.weight",
                from_pattern=[f"{prefix_from}.attn1.to_q.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.to_q.bias",
                from_pattern=[f"{prefix_from}.attn1.to_q.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.to_k.weight",
                from_pattern=[f"{prefix_from}.attn1.to_k.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.to_k.bias",
                from_pattern=[f"{prefix_from}.attn1.to_k.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.to_v.weight",
                from_pattern=[f"{prefix_from}.attn1.to_v.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.to_v.bias",
                from_pattern=[f"{prefix_from}.attn1.to_v.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.to_out.weight",
                from_pattern=[f"{prefix_from}.attn1.to_out.0.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.to_out.bias",
                from_pattern=[f"{prefix_from}.attn1.to_out.0.bias"],
            ),
            # QK norms
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.norm_q.weight",
                from_pattern=[f"{prefix_from}.attn1.norm_q.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.norm_q.bias",
                from_pattern=[f"{prefix_from}.attn1.norm_q.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.norm_k.weight",
                from_pattern=[f"{prefix_from}.attn1.norm_k.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.norm_k.bias",
                from_pattern=[f"{prefix_from}.attn1.norm_k.bias"],
            ),
        ])

        # Simple LayerNorm for cross-attention (norm2)
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.norm2.weight",
                from_pattern=[f"{prefix_from}.norm2.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.norm2.bias",
                from_pattern=[f"{prefix_from}.norm2.bias"],
            ),
        ])

        # Cross-attention (attn2)
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.to_q.weight",
                from_pattern=[f"{prefix_from}.attn2.to_q.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.to_q.bias",
                from_pattern=[f"{prefix_from}.attn2.to_q.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.to_k.weight",
                from_pattern=[f"{prefix_from}.attn2.to_k.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.to_k.bias",
                from_pattern=[f"{prefix_from}.attn2.to_k.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.to_v.weight",
                from_pattern=[f"{prefix_from}.attn2.to_v.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.to_v.bias",
                from_pattern=[f"{prefix_from}.attn2.to_v.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.to_out.weight",
                from_pattern=[f"{prefix_from}.attn2.to_out.0.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.to_out.bias",
                from_pattern=[f"{prefix_from}.attn2.to_out.0.bias"],
            ),
            # QK norms for cross-attention
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.norm_q.weight",
                from_pattern=[f"{prefix_from}.attn2.norm_q.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.norm_q.bias",
                from_pattern=[f"{prefix_from}.attn2.norm_q.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.norm_k.weight",
                from_pattern=[f"{prefix_from}.attn2.norm_k.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.norm_k.bias",
                from_pattern=[f"{prefix_from}.attn2.norm_k.bias"],
            ),
        ])

        # Simple LayerNorm for FFN (norm3)
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.norm3.weight",
                from_pattern=[f"{prefix_from}.norm3.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.norm3.bias",
                from_pattern=[f"{prefix_from}.norm3.bias"],
            ),
        ])

        # Feed-forward network
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.ff.net_0_proj.weight",
                from_pattern=[f"{prefix_from}.ff.net.0.proj.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.ff.net_0_proj.bias",
                from_pattern=[f"{prefix_from}.ff.net.0.proj.bias"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.ff.net_2.weight",
                from_pattern=[f"{prefix_from}.ff.net.2.weight"],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.ff.net_2.bias",
                from_pattern=[f"{prefix_from}.ff.net.2.bias"],
            ),
        ])

        return mappings

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        """Get VAE weight mapping - standard SDXL VAE architecture."""
        # Import FLUX VAE mapping since Hunyuan uses standard 4-channel VAE
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping
        return FluxWeightMapping.get_vae_mapping()

    @staticmethod
    def get_clip_encoder_mapping() -> List[WeightTarget]:
        """Get Chinese CLIP encoder weight mapping."""
        mappings = []

        # Embedding layers
        mappings.extend([
            WeightTarget(
                to_pattern="text_model.embeddings.token_embedding.weight",
                from_pattern=["text_model.embeddings.token_embedding.weight"],
            ),
            WeightTarget(
                to_pattern="text_model.embeddings.position_embedding.weight",
                from_pattern=["text_model.embeddings.position_embedding.weight"],
            ),
        ])

        # Encoder layers (12 layers for CLIP-L)
        for layer_idx in range(12):
            prefix = f"text_model.encoder.layers.{layer_idx}"
            mappings.extend([
                # Self-attention
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.q_proj.weight",
                    from_pattern=[f"{prefix}.self_attn.q_proj.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.q_proj.bias",
                    from_pattern=[f"{prefix}.self_attn.q_proj.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.k_proj.weight",
                    from_pattern=[f"{prefix}.self_attn.k_proj.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.k_proj.bias",
                    from_pattern=[f"{prefix}.self_attn.k_proj.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.v_proj.weight",
                    from_pattern=[f"{prefix}.self_attn.v_proj.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.v_proj.bias",
                    from_pattern=[f"{prefix}.self_attn.v_proj.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.out_proj.weight",
                    from_pattern=[f"{prefix}.self_attn.out_proj.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.out_proj.bias",
                    from_pattern=[f"{prefix}.self_attn.out_proj.bias"],
                ),
                # Layer norms
                WeightTarget(
                    to_pattern=f"{prefix}.layer_norm1.weight",
                    from_pattern=[f"{prefix}.layer_norm1.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.layer_norm1.bias",
                    from_pattern=[f"{prefix}.layer_norm1.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.layer_norm2.weight",
                    from_pattern=[f"{prefix}.layer_norm2.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.layer_norm2.bias",
                    from_pattern=[f"{prefix}.layer_norm2.bias"],
                ),
                # MLP
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.fc1.weight",
                    from_pattern=[f"{prefix}.mlp.fc1.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.fc1.bias",
                    from_pattern=[f"{prefix}.mlp.fc1.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.fc2.weight",
                    from_pattern=[f"{prefix}.mlp.fc2.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.fc2.bias",
                    from_pattern=[f"{prefix}.mlp.fc2.bias"],
                ),
            ])

        # Final layer norm
        mappings.extend([
            WeightTarget(
                to_pattern="text_model.final_layer_norm.weight",
                from_pattern=["text_model.final_layer_norm.weight"],
            ),
            WeightTarget(
                to_pattern="text_model.final_layer_norm.bias",
                from_pattern=["text_model.final_layer_norm.bias"],
            ),
        ])

        return mappings

    @staticmethod
    def get_t5_encoder_mapping() -> List[WeightTarget]:
        """Get mT5 encoder weight mapping - reuse FLUX T5 mapping."""
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping
        return FluxWeightMapping.get_t5_encoder_mapping()
