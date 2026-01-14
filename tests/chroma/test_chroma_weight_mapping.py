"""
Tests for Chroma weight mapping.

These are fast unit tests that verify mapping completeness without loading models.
"""

import pytest

from mflux.models.chroma.weights.chroma_weight_mapping import ChromaWeightMapping


class TestChromaWeightMapping:
    """Tests for ChromaWeightMapping class."""

    @pytest.mark.fast
    def test_get_transformer_mapping_returns_list(self):
        """Verify get_transformer_mapping returns a list."""
        mappings = ChromaWeightMapping.get_transformer_mapping()
        assert isinstance(mappings, list)
        assert len(mappings) > 0

    @pytest.mark.fast
    def test_distilled_guidance_layer_mappings_exist(self):
        """Verify DistilledGuidanceLayer weights are mapped."""
        mappings = ChromaWeightMapping.get_transformer_mapping()
        patterns = [m.to_pattern for m in mappings]

        # Check in_proj
        assert any("distilled_guidance_layer.in_proj.weight" in p for p in patterns)
        assert any("distilled_guidance_layer.in_proj.bias" in p for p in patterns)

        # Check out_proj
        assert any("distilled_guidance_layer.out_proj.weight" in p for p in patterns)
        assert any("distilled_guidance_layer.out_proj.bias" in p for p in patterns)

        # Check layers (5 layers, each with linear_1 and linear_2)
        assert any("distilled_guidance_layer.layers" in p for p in patterns)
        assert any("linear_1.weight" in p for p in patterns)
        assert any("linear_2.weight" in p for p in patterns)

        # Check norms (5 RMSNorms)
        assert any("distilled_guidance_layer.norms" in p for p in patterns)

    @pytest.mark.fast
    def test_no_time_text_embed_mapping(self):
        """Verify time_text_embed is NOT mapped (Chroma doesn't have it)."""
        mappings = ChromaWeightMapping.get_transformer_mapping()
        patterns = [m.to_pattern for m in mappings]

        # FLUX has time_text_embed, Chroma should NOT
        assert not any("time_text_embed" in p for p in patterns)

    @pytest.mark.fast
    def test_transformer_blocks_mapped(self):
        """Verify joint transformer block weights are mapped."""
        mappings = ChromaWeightMapping.get_transformer_mapping()
        patterns = [m.to_pattern for m in mappings]

        # Check attention projections
        assert any("transformer_blocks.{block}.attn.to_q" in p for p in patterns)
        assert any("transformer_blocks.{block}.attn.to_k" in p for p in patterns)
        assert any("transformer_blocks.{block}.attn.to_v" in p for p in patterns)
        assert any("transformer_blocks.{block}.attn.to_out" in p for p in patterns)

        # Check context projections (joint attention)
        assert any("transformer_blocks.{block}.attn.add_q_proj" in p for p in patterns)
        assert any("transformer_blocks.{block}.attn.add_k_proj" in p for p in patterns)
        assert any("transformer_blocks.{block}.attn.add_v_proj" in p for p in patterns)

        # Check feed-forward
        assert any("transformer_blocks.{block}.ff.linear1" in p for p in patterns)
        assert any("transformer_blocks.{block}.ff.linear2" in p for p in patterns)
        assert any("transformer_blocks.{block}.ff_context.linear1" in p for p in patterns)
        assert any("transformer_blocks.{block}.ff_context.linear2" in p for p in patterns)

    @pytest.mark.fast
    def test_single_transformer_blocks_mapped(self):
        """Verify single transformer block weights are mapped."""
        mappings = ChromaWeightMapping.get_transformer_mapping()
        patterns = [m.to_pattern for m in mappings]

        # Check attention projections
        assert any("single_transformer_blocks.{block}.attn.to_q" in p for p in patterns)
        assert any("single_transformer_blocks.{block}.attn.to_k" in p for p in patterns)
        assert any("single_transformer_blocks.{block}.attn.to_v" in p for p in patterns)

        # Check QK norms
        assert any("single_transformer_blocks.{block}.attn.norm_q" in p for p in patterns)
        assert any("single_transformer_blocks.{block}.attn.norm_k" in p for p in patterns)

        # Check MLP projections
        assert any("single_transformer_blocks.{block}.proj_mlp" in p for p in patterns)
        assert any("single_transformer_blocks.{block}.proj_out" in p for p in patterns)

    @pytest.mark.fast
    def test_no_norm1_linear_mapping(self):
        """Verify norm1.linear and norm1_context.linear are NOT mapped (Chroma doesn't have them)."""
        mappings = ChromaWeightMapping.get_transformer_mapping()
        patterns = [m.to_pattern for m in mappings]

        # FLUX has norm1.linear, Chroma should NOT (modulations are pre-computed)
        assert not any("norm1.linear" in p for p in patterns)
        assert not any("norm1_context.linear" in p for p in patterns)

    @pytest.mark.fast
    def test_no_single_block_norm_linear_mapping(self):
        """Verify single block norm.linear is NOT mapped."""
        mappings = ChromaWeightMapping.get_transformer_mapping()
        patterns = [m.to_pattern for m in mappings]

        # Chroma single blocks don't have norm.linear
        assert not any("single_transformer_blocks.{block}.norm.linear" in p for p in patterns)

    @pytest.mark.fast
    def test_embedder_mappings_exist(self):
        """Verify x_embedder and context_embedder are mapped."""
        mappings = ChromaWeightMapping.get_transformer_mapping()
        patterns = [m.to_pattern for m in mappings]

        assert any("x_embedder.weight" in p for p in patterns)
        assert any("x_embedder.bias" in p for p in patterns)
        assert any("context_embedder.weight" in p for p in patterns)
        assert any("context_embedder.bias" in p for p in patterns)

    @pytest.mark.fast
    def test_proj_out_mapping_exists(self):
        """Verify proj_out is mapped."""
        mappings = ChromaWeightMapping.get_transformer_mapping()
        patterns = [m.to_pattern for m in mappings]

        assert any("proj_out.weight" in p for p in patterns)
        assert any("proj_out.bias" in p for p in patterns)

    @pytest.mark.fast
    def test_block_counts(self):
        """Verify correct number of blocks in mappings."""
        mappings = ChromaWeightMapping.get_transformer_mapping()

        # Find mappings with max_blocks attribute
        single_block_mappings = [
            m
            for m in mappings
            if hasattr(m, "max_blocks") and m.max_blocks is not None and "single_transformer_blocks" in m.to_pattern
        ]

        distilled_layer_mappings = [
            m
            for m in mappings
            if hasattr(m, "max_blocks")
            and m.max_blocks is not None
            and "distilled_guidance_layer.layers" in m.to_pattern
        ]

        distilled_norm_mappings = [
            m
            for m in mappings
            if hasattr(m, "max_blocks")
            and m.max_blocks is not None
            and "distilled_guidance_layer.norms" in m.to_pattern
        ]

        # Single blocks should have max_blocks=38
        for m in single_block_mappings:
            assert m.max_blocks == 38

        # DistilledGuidanceLayer layers should have max_blocks=5
        for m in distilled_layer_mappings:
            assert m.max_blocks == 5

        # DistilledGuidanceLayer norms should have max_blocks=5
        for m in distilled_norm_mappings:
            assert m.max_blocks == 5


class TestChromaVAEMapping:
    """Tests for VAE weight mapping (reuses FLUX)."""

    @pytest.mark.fast
    def test_vae_mapping_reuses_flux(self):
        """Verify VAE mapping reuses FLUX VAE mapping."""
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

        chroma_vae = ChromaWeightMapping.get_vae_mapping()
        flux_vae = FluxWeightMapping.get_vae_mapping()

        # Should be identical
        assert len(chroma_vae) == len(flux_vae)


class TestChromaT5Mapping:
    """Tests for T5 encoder weight mapping (reuses FLUX)."""

    @pytest.mark.fast
    def test_t5_mapping_reuses_flux(self):
        """Verify T5 mapping reuses FLUX T5 mapping."""
        from mflux.models.flux.weights.flux_weight_mapping import FluxWeightMapping

        chroma_t5 = ChromaWeightMapping.get_t5_encoder_mapping()
        flux_t5 = FluxWeightMapping.get_t5_encoder_mapping()

        # Should be identical
        assert len(chroma_t5) == len(flux_t5)


class TestChromaWeightMappingCompleteness:
    """Tests for mapping completeness."""

    @pytest.mark.fast
    def test_distilled_guidance_layer_weight_count(self):
        """
        Verify DistilledGuidanceLayer has expected number of weight mappings.

        Expected:
        - in_proj: 2 (weight, bias)
        - layers.{0-4}: 5 * 4 = 20 (linear_1.weight/bias, linear_2.weight/bias)
        - norms.{0-4}: 5 (weight only, RMSNorm has no bias)
        - out_proj: 2 (weight, bias)

        Total: 2 + 20 + 5 + 2 = 29
        """
        mappings = ChromaWeightMapping.get_transformer_mapping()

        dgl_mappings = [m for m in mappings if "distilled_guidance_layer" in m.to_pattern]

        # Count unique pattern types (some have max_blocks)
        unique_patterns = set(m.to_pattern for m in dgl_mappings)

        # Should have:
        # - in_proj.weight, in_proj.bias
        # - layers.{block}.linear_1.weight, layers.{block}.linear_1.bias
        # - layers.{block}.linear_2.weight, layers.{block}.linear_2.bias
        # - norms.{block}.weight
        # - out_proj.weight, out_proj.bias
        expected_pattern_types = 9  # 2 + 4 + 1 + 2

        assert len(unique_patterns) == expected_pattern_types

    @pytest.mark.fast
    def test_no_clip_encoder_references(self):
        """Verify no CLIP encoder references (Chroma uses T5 only)."""
        # ChromaWeightMapping doesn't have get_clip_mapping method
        assert not hasattr(ChromaWeightMapping, "get_clip_mapping")

    @pytest.mark.fast
    def test_mapping_patterns_are_valid(self):
        """Verify all mapping patterns have valid format."""
        mappings = ChromaWeightMapping.get_transformer_mapping()

        for m in mappings:
            # to_pattern should be a string
            assert isinstance(m.to_pattern, str)

            # from_pattern should be a list
            assert isinstance(m.from_pattern, list)

            # Pattern should not be empty
            assert len(m.to_pattern) > 0
            assert len(m.from_pattern) > 0
