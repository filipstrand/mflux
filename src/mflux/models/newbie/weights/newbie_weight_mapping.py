"""Weight mapping for NewBie-image model.

Maps HuggingFace weight names to MFLUX model structure.

NewBie-image architecture:
- 36 NextDiT blocks with GQA attention
- Dual text encoders (Gemma3 + Jina CLIP)
- 16-channel VAE (FLUX.1-dev)
- AdaLN-Single conditioning
"""

from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


class NewBieWeightMapping(WeightMapping):
    """Weight mapping for NewBie-image model."""

    NUM_BLOCKS = 36

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        """
        Mapping for NextDiT transformer weights.

        NextDiT structure:
        - patch_embed: image patch embedding
        - time_embed: timestep embedding MLP
        - guidance_embed: guidance scale embedding
        - text_proj: dual encoder projection
        - blocks.0-35: NextDiT transformer blocks with GQA
        - norm_out: final RMSNorm
        - final_adaLN: final modulation
        - proj_out: output projection
        """
        mappings = []

        # Patch embedding
        mappings.extend([
            WeightTarget(
                to_pattern="patch_embed.proj.weight",
                from_pattern=["x_embedder.proj.weight", "patch_embed.proj.weight"],
            ),
            WeightTarget(
                to_pattern="patch_embed.proj.bias",
                from_pattern=["x_embedder.proj.bias", "patch_embed.proj.bias"],
            ),
        ])

        # Time embedding
        mappings.extend([
            WeightTarget(
                to_pattern="time_embed.mlp.layers.0.weight",
                from_pattern=["t_embedder.mlp.0.weight", "time_embed.mlp.0.weight"],
            ),
            WeightTarget(
                to_pattern="time_embed.mlp.layers.0.bias",
                from_pattern=["t_embedder.mlp.0.bias", "time_embed.mlp.0.bias"],
            ),
            WeightTarget(
                to_pattern="time_embed.mlp.layers.2.weight",
                from_pattern=["t_embedder.mlp.2.weight", "time_embed.mlp.2.weight"],
            ),
            WeightTarget(
                to_pattern="time_embed.mlp.layers.2.bias",
                from_pattern=["t_embedder.mlp.2.bias", "time_embed.mlp.2.bias"],
            ),
        ])

        # Guidance embedding (similar to time)
        mappings.extend([
            WeightTarget(
                to_pattern="guidance_embed.mlp.layers.0.weight",
                from_pattern=["guidance_embedder.mlp.0.weight", "guidance_embed.mlp.0.weight"],
            ),
            WeightTarget(
                to_pattern="guidance_embed.mlp.layers.0.bias",
                from_pattern=["guidance_embedder.mlp.0.bias", "guidance_embed.mlp.0.bias"],
            ),
            WeightTarget(
                to_pattern="guidance_embed.mlp.layers.2.weight",
                from_pattern=["guidance_embedder.mlp.2.weight", "guidance_embed.mlp.2.weight"],
            ),
            WeightTarget(
                to_pattern="guidance_embed.mlp.layers.2.bias",
                from_pattern=["guidance_embedder.mlp.2.bias", "guidance_embed.mlp.2.bias"],
            ),
        ])

        # Text projector
        mappings.extend([
            WeightTarget(
                to_pattern="text_proj.gemma_proj.weight",
                from_pattern=["context_embedder.weight", "text_proj.gemma_proj.weight"],
            ),
            WeightTarget(
                to_pattern="text_proj.gemma_proj.bias",
                from_pattern=["context_embedder.bias", "text_proj.gemma_proj.bias"],
            ),
            WeightTarget(
                to_pattern="text_proj.clip_proj.weight",
                from_pattern=["pooled_text_embedder.weight", "text_proj.clip_proj.weight"],
            ),
            WeightTarget(
                to_pattern="text_proj.clip_proj.bias",
                from_pattern=["pooled_text_embedder.bias", "text_proj.clip_proj.bias"],
            ),
            WeightTarget(
                to_pattern="text_proj.fusion.weight",
                from_pattern=["text_proj.fusion.weight"],
            ),
            WeightTarget(
                to_pattern="text_proj.fusion.bias",
                from_pattern=["text_proj.fusion.bias"],
            ),
        ])

        # NextDiT transformer blocks (36 blocks)
        for block_idx in range(NewBieWeightMapping.NUM_BLOCKS):
            mappings.extend(NewBieWeightMapping._get_nextdit_block_mapping(block_idx))

        # Output layers
        mappings.extend([
            WeightTarget(
                to_pattern="norm_out.weight",
                from_pattern=["final_layer.norm_final.weight", "norm_out.weight"],
            ),
            WeightTarget(
                to_pattern="final_adaLN.weight",
                from_pattern=["final_layer.adaLN_modulation.1.weight", "final_adaLN.weight"],
            ),
            WeightTarget(
                to_pattern="final_adaLN.bias",
                from_pattern=["final_layer.adaLN_modulation.1.bias", "final_adaLN.bias"],
            ),
            WeightTarget(
                to_pattern="proj_out.weight",
                from_pattern=["final_layer.linear.weight", "proj_out.weight"],
            ),
            WeightTarget(
                to_pattern="proj_out.bias",
                from_pattern=["final_layer.linear.bias", "proj_out.bias"],
            ),
        ])

        return mappings

    @staticmethod
    def _get_nextdit_block_mapping(block_idx: int) -> List[WeightTarget]:
        """
        Get weight mapping for a single NextDiT block.

        Each block has:
        1. norm1 + attn1 (self-attention with GQA)
        2. norm2 + attn2 (cross-attention with GQA)
        3. norm3 + ffn (SwiGLU feed-forward)
        4. adaLN_modulation (AdaLN-Single)
        """
        prefix_to = f"blocks.{block_idx}"
        prefix_from_layers = f"layers.{block_idx}"  # Lumina naming
        prefix_from_blocks = f"transformer_blocks.{block_idx}"  # Diffusers naming

        mappings = []

        # Pre-norm for self-attention (norm1)
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.norm1.weight",
                from_pattern=[
                    f"{prefix_from_layers}.attention_norm.weight",
                    f"{prefix_from_blocks}.norm1.weight",
                ],
            ),
        ])

        # Self-attention (attn1) - GQA
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.wq.weight",
                from_pattern=[
                    f"{prefix_from_layers}.attention.wq.weight",
                    f"{prefix_from_blocks}.attn1.to_q.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.wk.weight",
                from_pattern=[
                    f"{prefix_from_layers}.attention.wk.weight",
                    f"{prefix_from_blocks}.attn1.to_k.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.wv.weight",
                from_pattern=[
                    f"{prefix_from_layers}.attention.wv.weight",
                    f"{prefix_from_blocks}.attn1.to_v.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.wo.weight",
                from_pattern=[
                    f"{prefix_from_layers}.attention.wo.weight",
                    f"{prefix_from_blocks}.attn1.to_out.0.weight",
                ],
            ),
            # QK norms
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.q_norm.weight",
                from_pattern=[
                    f"{prefix_from_layers}.attention.q_norm.weight",
                    f"{prefix_from_blocks}.attn1.norm_q.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn1.k_norm.weight",
                from_pattern=[
                    f"{prefix_from_layers}.attention.k_norm.weight",
                    f"{prefix_from_blocks}.attn1.norm_k.weight",
                ],
            ),
        ])

        # Pre-norm for cross-attention (norm2)
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.norm2.weight",
                from_pattern=[
                    f"{prefix_from_layers}.cross_attention_norm.weight",
                    f"{prefix_from_blocks}.norm2.weight",
                ],
            ),
        ])

        # Cross-attention (attn2) - GQA
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.wq.weight",
                from_pattern=[
                    f"{prefix_from_layers}.cross_attention.wq.weight",
                    f"{prefix_from_blocks}.attn2.to_q.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.wk.weight",
                from_pattern=[
                    f"{prefix_from_layers}.cross_attention.wk.weight",
                    f"{prefix_from_blocks}.attn2.to_k.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.wv.weight",
                from_pattern=[
                    f"{prefix_from_layers}.cross_attention.wv.weight",
                    f"{prefix_from_blocks}.attn2.to_v.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.wo.weight",
                from_pattern=[
                    f"{prefix_from_layers}.cross_attention.wo.weight",
                    f"{prefix_from_blocks}.attn2.to_out.0.weight",
                ],
            ),
            # QK norms
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.q_norm.weight",
                from_pattern=[
                    f"{prefix_from_layers}.cross_attention.q_norm.weight",
                    f"{prefix_from_blocks}.attn2.norm_q.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.attn2.k_norm.weight",
                from_pattern=[
                    f"{prefix_from_layers}.cross_attention.k_norm.weight",
                    f"{prefix_from_blocks}.attn2.norm_k.weight",
                ],
            ),
        ])

        # Pre-norm for FFN (norm3)
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.norm3.weight",
                from_pattern=[
                    f"{prefix_from_layers}.ffn_norm.weight",
                    f"{prefix_from_blocks}.norm3.weight",
                ],
            ),
        ])

        # SwiGLU FFN
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.ffn.w1.weight",
                from_pattern=[
                    f"{prefix_from_layers}.feed_forward.w1.weight",
                    f"{prefix_from_blocks}.ff.net.0.proj.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.ffn.w2.weight",
                from_pattern=[
                    f"{prefix_from_layers}.feed_forward.w2.weight",
                    f"{prefix_from_blocks}.ff.net.2.weight",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.ffn.w3.weight",
                from_pattern=[
                    f"{prefix_from_layers}.feed_forward.w3.weight",
                    f"{prefix_from_blocks}.ff.net.0.gate.weight",
                ],
            ),
        ])

        # AdaLN modulation
        mappings.extend([
            WeightTarget(
                to_pattern=f"{prefix_to}.adaLN_modulation.linear.weight",
                from_pattern=[
                    f"{prefix_from_layers}.adaLN_modulation.1.weight",
                    f"{prefix_from_blocks}.scale_shift_table",
                ],
            ),
            WeightTarget(
                to_pattern=f"{prefix_to}.adaLN_modulation.linear.bias",
                from_pattern=[
                    f"{prefix_from_layers}.adaLN_modulation.1.bias",
                    f"{prefix_from_blocks}.adaLN_modulation.bias",
                ],
            ),
        ])

        return mappings

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        """Get VAE weight mapping - use FLUX VAE (16-channel)."""
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping
        return FluxWeightMapping.get_vae_mapping()

    @staticmethod
    def get_gemma3_encoder_mapping() -> List[WeightTarget]:
        """Get Gemma3 encoder weight mapping."""
        mappings = []

        # Token embeddings
        mappings.append(
            WeightTarget(
                to_pattern="embed_tokens.weight",
                from_pattern=["model.embed_tokens.weight"],
            )
        )

        # Transformer layers (36 layers)
        for layer_idx in range(36):
            prefix = f"layers.{layer_idx}"
            prefix_from = f"model.layers.{layer_idx}"

            mappings.extend([
                # Input LayerNorm
                WeightTarget(
                    to_pattern=f"{prefix}.input_layernorm.weight",
                    from_pattern=[f"{prefix_from}.input_layernorm.weight"],
                ),
                # Self-attention
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.q_proj.weight",
                    from_pattern=[f"{prefix_from}.self_attn.q_proj.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.k_proj.weight",
                    from_pattern=[f"{prefix_from}.self_attn.k_proj.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.v_proj.weight",
                    from_pattern=[f"{prefix_from}.self_attn.v_proj.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.o_proj.weight",
                    from_pattern=[f"{prefix_from}.self_attn.o_proj.weight"],
                ),
                # Post-attention LayerNorm
                WeightTarget(
                    to_pattern=f"{prefix}.post_attention_layernorm.weight",
                    from_pattern=[f"{prefix_from}.post_attention_layernorm.weight"],
                ),
                # MLP
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.gate_proj.weight",
                    from_pattern=[f"{prefix_from}.mlp.gate_proj.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.up_proj.weight",
                    from_pattern=[f"{prefix_from}.mlp.up_proj.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.down_proj.weight",
                    from_pattern=[f"{prefix_from}.mlp.down_proj.weight"],
                ),
            ])

        # Final LayerNorm
        mappings.append(
            WeightTarget(
                to_pattern="norm.weight",
                from_pattern=["model.norm.weight"],
            )
        )

        return mappings

    @staticmethod
    def get_jina_clip_encoder_mapping() -> List[WeightTarget]:
        """Get Jina CLIP encoder weight mapping."""
        mappings = []

        # Embeddings
        mappings.extend([
            WeightTarget(
                to_pattern="word_embeddings.weight",
                from_pattern=["embeddings.word_embeddings.weight"],
            ),
            WeightTarget(
                to_pattern="position_embeddings.weight",
                from_pattern=["embeddings.position_embeddings.weight"],
            ),
            WeightTarget(
                to_pattern="embeddings_layer_norm.weight",
                from_pattern=["embeddings.LayerNorm.weight"],
            ),
            WeightTarget(
                to_pattern="embeddings_layer_norm.bias",
                from_pattern=["embeddings.LayerNorm.bias"],
            ),
        ])

        # Encoder layers (24 layers)
        for layer_idx in range(24):
            prefix = f"layers.{layer_idx}"
            prefix_from = f"encoder.layer.{layer_idx}"

            mappings.extend([
                # Self-attention
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.q_proj.weight",
                    from_pattern=[f"{prefix_from}.attention.self.query.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.q_proj.bias",
                    from_pattern=[f"{prefix_from}.attention.self.query.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.k_proj.weight",
                    from_pattern=[f"{prefix_from}.attention.self.key.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.k_proj.bias",
                    from_pattern=[f"{prefix_from}.attention.self.key.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.v_proj.weight",
                    from_pattern=[f"{prefix_from}.attention.self.value.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.v_proj.bias",
                    from_pattern=[f"{prefix_from}.attention.self.value.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.out_proj.weight",
                    from_pattern=[f"{prefix_from}.attention.output.dense.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.self_attn.out_proj.bias",
                    from_pattern=[f"{prefix_from}.attention.output.dense.bias"],
                ),
                # Layer norms
                WeightTarget(
                    to_pattern=f"{prefix}.layer_norm1.weight",
                    from_pattern=[f"{prefix_from}.attention.output.LayerNorm.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.layer_norm1.bias",
                    from_pattern=[f"{prefix_from}.attention.output.LayerNorm.bias"],
                ),
                # MLP
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.fc1.weight",
                    from_pattern=[f"{prefix_from}.intermediate.dense.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.fc1.bias",
                    from_pattern=[f"{prefix_from}.intermediate.dense.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.fc2.weight",
                    from_pattern=[f"{prefix_from}.output.dense.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.mlp.fc2.bias",
                    from_pattern=[f"{prefix_from}.output.dense.bias"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.layer_norm2.weight",
                    from_pattern=[f"{prefix_from}.output.LayerNorm.weight"],
                ),
                WeightTarget(
                    to_pattern=f"{prefix}.layer_norm2.bias",
                    from_pattern=[f"{prefix_from}.output.LayerNorm.bias"],
                ),
            ])

        # Note: Standard BERT models don't have a final LayerNorm after all encoder layers.
        # Each encoder layer has internal LayerNorms. If Jina CLIP v2 has a final LayerNorm,
        # it would be at "encoder.final_layer_norm" or similar, NOT "pooler.dense" which
        # is a Linear layer. The current implementation may need architectural verification.
        # For now, mapping from potential source locations for a final encoder LayerNorm.
        mappings.extend([
            WeightTarget(
                to_pattern="final_layer_norm.weight",
                from_pattern=[
                    "text_model.encoder.final_layer_norm.weight",
                    "encoder.final_layer_norm.weight",
                    "text_model.final_layer_norm.weight",
                ],
            ),
            WeightTarget(
                to_pattern="final_layer_norm.bias",
                from_pattern=[
                    "text_model.encoder.final_layer_norm.bias",
                    "encoder.final_layer_norm.bias",
                    "text_model.final_layer_norm.bias",
                ],
            ),
        ])

        return mappings
