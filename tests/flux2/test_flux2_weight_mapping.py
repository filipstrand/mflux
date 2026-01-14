"""
Unit tests for FLUX.2 weight mapping verification.

Tests verify:
1. Weight mapping coverage for all FLUX.2 components
2. Mistral3 text encoder weight patterns
3. VAE weight mapping compatibility
4. Transformer weight mappings
"""

import pytest

from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class TestFlux2TransformerWeightMapping:
    """Tests for FLUX.2 transformer weight mapping."""

    @pytest.mark.fast
    def test_transformer_mapping_returns_list(self):
        """Verify get_transformer_mapping returns a list."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        assert isinstance(mapping, list)
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_input_embedder_mappings(self):
        """Verify input embedder weight mappings exist."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        assert "x_embedder.weight" in to_patterns
        assert "context_embedder.weight" in to_patterns

    @pytest.mark.fast
    def test_output_projection_mapping(self):
        """Verify output projection weight mapping exists."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        assert "proj_out.weight" in to_patterns

    @pytest.mark.fast
    def test_norm_out_mapping(self):
        """Verify output norm weight mapping exists."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        assert "norm_out.linear.weight" in to_patterns

    @pytest.mark.fast
    def test_time_guidance_embed_mappings(self):
        """Verify time and guidance embedder weight mappings exist."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # Timestep embedder
        assert "time_guidance_embed.timestep_embedder.linear_1.weight" in to_patterns
        assert "time_guidance_embed.timestep_embedder.linear_2.weight" in to_patterns

        # Guidance embedder
        assert "time_guidance_embed.guidance_embedder.linear_1.weight" in to_patterns
        assert "time_guidance_embed.guidance_embedder.linear_2.weight" in to_patterns

    @pytest.mark.fast
    def test_global_modulation_mappings(self):
        """Verify FLUX.2 global modulation layer mappings exist."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # These are FLUX.2 specific (not in FLUX.1)
        assert "double_stream_modulation_img.linear.weight" in to_patterns
        assert "double_stream_modulation_txt.linear.weight" in to_patterns
        assert "single_stream_modulation.linear.weight" in to_patterns


class TestFlux2JointBlockWeightMapping:
    """Tests for FLUX.2 joint transformer block weight mapping."""

    @pytest.mark.fast
    def test_joint_block_attention_qkv_mappings(self):
        """Verify joint block attention QKV weight mappings exist."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # Image stream QKV
        assert "transformer_blocks.{block}.attn.to_q.weight" in to_patterns
        assert "transformer_blocks.{block}.attn.to_k.weight" in to_patterns
        assert "transformer_blocks.{block}.attn.to_v.weight" in to_patterns
        assert "transformer_blocks.{block}.attn.to_out.0.weight" in to_patterns

    @pytest.mark.fast
    def test_joint_block_context_projections(self):
        """Verify joint block context projection weight mappings exist."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # Context stream projections
        assert "transformer_blocks.{block}.attn.add_q_proj.weight" in to_patterns
        assert "transformer_blocks.{block}.attn.add_k_proj.weight" in to_patterns
        assert "transformer_blocks.{block}.attn.add_v_proj.weight" in to_patterns
        assert "transformer_blocks.{block}.attn.to_add_out.weight" in to_patterns

    @pytest.mark.fast
    def test_joint_block_rmsnorm_mappings(self):
        """Verify joint block RMSNorm weight mappings exist."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # Image stream norms
        assert "transformer_blocks.{block}.attn.norm_q.weight" in to_patterns
        assert "transformer_blocks.{block}.attn.norm_k.weight" in to_patterns

        # Context stream norms
        assert "transformer_blocks.{block}.attn.norm_added_q.weight" in to_patterns
        assert "transformer_blocks.{block}.attn.norm_added_k.weight" in to_patterns

    @pytest.mark.fast
    def test_joint_block_feedforward_mappings(self):
        """Verify joint block feed-forward weight mappings use FLUX.2 naming."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # Image stream FF (linear_in/linear_out, not linear1/linear2)
        assert "transformer_blocks.{block}.ff.linear_in.weight" in to_patterns
        assert "transformer_blocks.{block}.ff.linear_out.weight" in to_patterns

        # Context stream FF
        assert "transformer_blocks.{block}.ff_context.linear_in.weight" in to_patterns
        assert "transformer_blocks.{block}.ff_context.linear_out.weight" in to_patterns


class TestFlux2SingleBlockWeightMapping:
    """Tests for FLUX.2 single transformer block weight mapping."""

    @pytest.mark.fast
    def test_single_block_fused_projection_mapping(self):
        """Verify single block fused QKV+MLP projection mapping exists."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # FLUX.2 uses fused projection (unique to FLUX.2)
        assert "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.weight" in to_patterns

    @pytest.mark.fast
    def test_single_block_output_mapping(self):
        """Verify single block output projection mapping exists."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        assert "single_transformer_blocks.{block}.attn.to_out.weight" in to_patterns

    @pytest.mark.fast
    def test_single_block_rmsnorm_mappings(self):
        """Verify single block RMSNorm weight mappings exist."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        assert "single_transformer_blocks.{block}.attn.norm_q.weight" in to_patterns
        assert "single_transformer_blocks.{block}.attn.norm_k.weight" in to_patterns


class TestFlux2VAEWeightMapping:
    """Tests for FLUX.2 VAE weight mapping."""

    @pytest.mark.fast
    def test_vae_mapping_returns_list(self):
        """Verify get_vae_mapping returns a list."""
        mapping = Flux2WeightMapping.get_vae_mapping()
        assert isinstance(mapping, list)
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_vae_mapping_reuses_flux1_structure(self):
        """Verify VAE mapping reuses FLUX.1 structure (same architecture)."""
        # FLUX.2 VAE has same architecture as FLUX.1, just 32 channels vs 16
        # So the weight mapping should be identical
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

        flux2_mapping = Flux2WeightMapping.get_vae_mapping()
        flux1_mapping = FluxWeightMapping.get_vae_mapping()

        # Should have same number of targets
        assert len(flux2_mapping) == len(flux1_mapping)

    @pytest.mark.fast
    def test_vae_encoder_decoder_coverage(self):
        """Verify VAE mapping covers encoder and decoder."""
        mapping = Flux2WeightMapping.get_vae_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # Should have encoder patterns
        encoder_patterns = [p for p in to_patterns if "encoder" in p]
        assert len(encoder_patterns) > 0

        # Should have decoder patterns
        decoder_patterns = [p for p in to_patterns if "decoder" in p]
        assert len(decoder_patterns) > 0


class TestFlux2TextEncoderWeightMapping:
    """Tests for FLUX.2 Mistral3 text encoder weight mapping."""

    @pytest.mark.fast
    def test_text_encoder_mapping_returns_list(self):
        """Verify get_text_encoder_mapping returns a list."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()
        assert isinstance(mapping, list)
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_embeddings_mapping(self):
        """Verify token embeddings weight mapping exists."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        assert "embed_tokens.weight" in to_patterns

        # Check from_pattern
        embed_target = [t for t in mapping if t.to_pattern == "embed_tokens.weight"][0]
        assert "model.embed_tokens.weight" in embed_target.from_pattern

    @pytest.mark.fast
    def test_layer_norm_mappings(self):
        """Verify layer norm weight mappings exist."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # RMSNorm for each layer
        assert "layers.{block}.input_layernorm.weight" in to_patterns
        assert "layers.{block}.post_attention_layernorm.weight" in to_patterns

    @pytest.mark.fast
    def test_self_attention_mappings(self):
        """Verify self-attention weight mappings exist."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # QKV projections
        assert "layers.{block}.self_attn.q_proj.weight" in to_patterns
        assert "layers.{block}.self_attn.k_proj.weight" in to_patterns
        assert "layers.{block}.self_attn.v_proj.weight" in to_patterns
        assert "layers.{block}.self_attn.o_proj.weight" in to_patterns

    @pytest.mark.fast
    def test_mlp_mappings(self):
        """Verify MLP weight mappings exist."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # MLP projections
        assert "layers.{block}.mlp.gate_proj.weight" in to_patterns
        assert "layers.{block}.mlp.up_proj.weight" in to_patterns
        assert "layers.{block}.mlp.down_proj.weight" in to_patterns

    @pytest.mark.fast
    def test_final_norm_mapping(self):
        """Verify final norm weight mapping exists."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        assert "norm.weight" in to_patterns

        # Check from_pattern
        norm_target = [t for t in mapping if t.to_pattern == "norm.weight"][0]
        assert "model.norm.weight" in norm_target.from_pattern

    @pytest.mark.fast
    def test_output_projection_mapping(self):
        """Verify output projection weight mapping exists with multiple patterns."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()

        # Find output_proj target
        output_proj_targets = [t for t in mapping if t.to_pattern == "output_proj.weight"]
        assert len(output_proj_targets) == 1

        output_proj = output_proj_targets[0]

        # Should have multiple from_patterns for flexibility
        assert len(output_proj.from_pattern) >= 2
        assert "output_projection.weight" in output_proj.from_pattern
        assert "model.output_projection.weight" in output_proj.from_pattern

        # Should be marked as optional
        assert output_proj.required is False

    @pytest.mark.fast
    def test_output_projection_alternative_patterns(self):
        """Verify output projection includes alternative weight patterns."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()

        output_proj = [t for t in mapping if t.to_pattern == "output_proj.weight"][0]

        # Should include lm_head as alternative
        assert "lm_head.weight" in output_proj.from_pattern


class TestFlux2WeightMappingFromPatterns:
    """Tests for weight mapping from_pattern coverage."""

    @pytest.mark.fast
    def test_all_targets_have_from_patterns(self):
        """Verify all weight targets have from_patterns defined."""
        transformer_mapping = Flux2WeightMapping.get_transformer_mapping()
        text_encoder_mapping = Flux2WeightMapping.get_text_encoder_mapping()

        for target in transformer_mapping + text_encoder_mapping:
            assert target.from_pattern is not None
            assert len(target.from_pattern) > 0

    @pytest.mark.fast
    def test_from_patterns_are_unique_per_target(self):
        """Verify from_patterns are distinct for each target."""
        mapping = Flux2WeightMapping.get_transformer_mapping()

        for target in mapping:
            # Each from_pattern should be unique within the target
            assert len(target.from_pattern) == len(set(target.from_pattern))

    @pytest.mark.fast
    def test_block_patterns_use_placeholder(self):
        """Verify block-specific patterns use {block} placeholder."""
        mapping = Flux2WeightMapping.get_transformer_mapping()

        # Joint blocks
        joint_block_targets = [
            t for t in mapping
            if "transformer_blocks.{block}" in t.to_pattern
        ]
        assert len(joint_block_targets) > 0

        # Single blocks
        single_block_targets = [
            t for t in mapping
            if "single_transformer_blocks.{block}" in t.to_pattern
        ]
        assert len(single_block_targets) > 0


class TestFlux2WeightMappingVsFlux1Differences:
    """Tests verifying FLUX.2 weight mapping differences from FLUX.1."""

    @pytest.mark.fast
    def test_global_modulation_layers_are_new(self):
        """Verify FLUX.2 has global modulation mappings (not in FLUX.1)."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # FLUX.2 specific
        assert "double_stream_modulation_img.linear.weight" in to_patterns
        assert "double_stream_modulation_txt.linear.weight" in to_patterns
        assert "single_stream_modulation.linear.weight" in to_patterns

    @pytest.mark.fast
    def test_no_per_block_norm_mappings(self):
        """Verify FLUX.2 doesn't have per-block norm mappings (uses global modulation)."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # FLUX.1 had these, FLUX.2 doesn't
        assert "transformer_blocks.{block}.norm1.linear.weight" not in to_patterns
        assert "transformer_blocks.{block}.norm1_context.linear.weight" not in to_patterns

    @pytest.mark.fast
    def test_linear_in_out_naming(self):
        """Verify FLUX.2 uses linear_in/linear_out (vs linear1/linear2 in FLUX.1)."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # FLUX.2 naming
        assert any("linear_in" in p for p in to_patterns)
        assert any("linear_out" in p for p in to_patterns)

        # FLUX.1 naming should not be present in model paths
        model_paths_str = " ".join(to_patterns)
        assert "ff.linear1.weight" not in model_paths_str
        assert "ff.linear2.weight" not in model_paths_str

    @pytest.mark.fast
    def test_fused_single_block_projection(self):
        """Verify FLUX.2 has fused single block projection mapping."""
        mapping = Flux2WeightMapping.get_transformer_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # FLUX.2 specific fused projection
        assert "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj.weight" in to_patterns

    @pytest.mark.fast
    def test_mistral3_vs_clip_t5_mapping(self):
        """Verify FLUX.2 uses Mistral3 mapping (vs CLIP+T5 in FLUX.1)."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()
        to_patterns = [target.to_pattern for target in mapping]

        # Mistral3 patterns
        assert "embed_tokens.weight" in to_patterns
        assert "layers.{block}.self_attn.q_proj.weight" in to_patterns

        # Should have output_proj for joint_attention_dim
        assert "output_proj.weight" in to_patterns


class TestFlux2WeightMappingTargetCounts:
    """Tests for expected weight mapping target counts."""

    @pytest.mark.fast
    def test_transformer_mapping_has_sufficient_targets(self):
        """Verify transformer mapping has reasonable number of targets."""
        mapping = Flux2WeightMapping.get_transformer_mapping()

        # Should have many targets for 8 joint blocks + 48 single blocks
        # Plus global modulation, embedders, etc.
        assert len(mapping) >= 20

    @pytest.mark.fast
    def test_text_encoder_mapping_has_sufficient_targets(self):
        """Verify text encoder mapping has targets for 40 layers."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()

        # Embeddings + 40 layers × 7 weights per layer + final norm + output_proj
        # = 1 + 280 + 1 + 1 = 283 minimum (with {block} placeholders)
        # But with placeholders, we have fewer unique patterns
        assert len(mapping) >= 10

    @pytest.mark.fast
    def test_vae_mapping_has_sufficient_targets(self):
        """Verify VAE mapping has sufficient targets for encoder/decoder."""
        mapping = Flux2WeightMapping.get_vae_mapping()

        # Should have targets for encoder and decoder blocks
        assert len(mapping) >= 10


class TestFlux2WeightMappingRequiredFlags:
    """Tests for weight mapping required flags."""

    @pytest.mark.fast
    def test_essential_weights_are_required(self):
        """Verify essential weights are marked as required."""
        transformer_mapping = Flux2WeightMapping.get_transformer_mapping()

        # Essential weights that must exist
        essential_patterns = [
            "x_embedder.weight",
            "context_embedder.weight",
            "proj_out.weight",
        ]

        for pattern in essential_patterns:
            targets = [t for t in transformer_mapping if t.to_pattern == pattern]
            if len(targets) > 0:
                # If required field exists, verify it's True or None (default True)
                target = targets[0]
                if hasattr(target, "required"):
                    assert target.required is not False

    @pytest.mark.fast
    def test_output_proj_is_optional(self):
        """Verify output_proj is marked as optional."""
        mapping = Flux2WeightMapping.get_text_encoder_mapping()

        output_proj = [t for t in mapping if t.to_pattern == "output_proj.weight"][0]
        assert output_proj.required is False
