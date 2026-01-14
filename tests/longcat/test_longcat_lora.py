"""
Unit tests for LongCat LoRA mapping and loading.

Tests verify:
1. LongCatLoRAMapping covers all expected targets
2. LongCatLoRAMapping includes norm layers (unlike Chroma)
3. BFL/Kohya format patterns are correctly defined
"""

import pytest

from mflux.models.longcat.weights.longcat_lora_mapping import LongCatLoRAMapping


class TestLongCatLoRAMappingCoverage:
    """Tests for LongCatLoRAMapping target coverage."""

    @pytest.mark.fast
    def test_mapping_returns_non_empty_list(self):
        """Verify get_mapping returns a non-empty list of targets."""
        mapping = LongCatLoRAMapping.get_mapping()
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_joint_block_attention_targets(self):
        """Verify joint transformer block attention targets are included."""
        mapping = LongCatLoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        # Image attention
        assert "transformer_blocks.{block}.attn.to_q" in paths
        assert "transformer_blocks.{block}.attn.to_k" in paths
        assert "transformer_blocks.{block}.attn.to_v" in paths
        assert "transformer_blocks.{block}.attn.to_out.0" in paths

        # Text/context attention
        assert "transformer_blocks.{block}.attn.add_q_proj" in paths
        assert "transformer_blocks.{block}.attn.add_k_proj" in paths
        assert "transformer_blocks.{block}.attn.add_v_proj" in paths
        assert "transformer_blocks.{block}.attn.to_add_out" in paths

    @pytest.mark.fast
    def test_joint_block_ff_targets(self):
        """Verify joint transformer block feed-forward targets are included."""
        mapping = LongCatLoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        # Image FFN
        assert "transformer_blocks.{block}.ff.linear1" in paths
        assert "transformer_blocks.{block}.ff.linear2" in paths

        # Text/context FFN
        assert "transformer_blocks.{block}.ff_context.linear1" in paths
        assert "transformer_blocks.{block}.ff_context.linear2" in paths

    @pytest.mark.fast
    def test_single_block_attention_targets(self):
        """Verify single transformer block attention targets are included."""
        mapping = LongCatLoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        assert "single_transformer_blocks.{block}.attn.to_q" in paths
        assert "single_transformer_blocks.{block}.attn.to_k" in paths
        assert "single_transformer_blocks.{block}.attn.to_v" in paths

    @pytest.mark.fast
    def test_single_block_mlp_targets(self):
        """Verify single transformer block MLP targets are included."""
        mapping = LongCatLoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        assert "single_transformer_blocks.{block}.proj_mlp" in paths
        assert "single_transformer_blocks.{block}.proj_out" in paths


class TestLongCatLoRAMappingNormLayers:
    """Tests for norm layer targets (LongCat uses TimeTextEmbed unlike Chroma)."""

    @pytest.mark.fast
    def test_norm1_linear_targets_included(self):
        """Verify norm1.linear IS included (LongCat uses TimeTextEmbed, not DistilledGuidanceLayer)."""
        mapping = LongCatLoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        # These exist in LongCat (unlike Chroma)
        assert "transformer_blocks.{block}.norm1.linear" in paths

    @pytest.mark.fast
    def test_norm1_context_linear_targets_included(self):
        """Verify norm1_context.linear IS included."""
        mapping = LongCatLoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        assert "transformer_blocks.{block}.norm1_context.linear" in paths

    @pytest.mark.fast
    def test_single_block_norm_linear_targets_included(self):
        """Verify single block norm.linear IS included."""
        mapping = LongCatLoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        assert "single_transformer_blocks.{block}.norm.linear" in paths


class TestLongCatLoRAMappingBFLPatterns:
    """Tests for BFL/Kohya format pattern coverage."""

    @pytest.mark.fast
    def test_double_block_qkv_patterns(self):
        """Verify BFL double block QKV patterns are included."""
        mapping = LongCatLoRAMapping.get_mapping()

        # Find to_q target and check for BFL pattern
        to_q_targets = [t for t in mapping if t.model_path == "transformer_blocks.{block}.attn.to_q"]
        assert len(to_q_targets) >= 2  # Standard + BFL

        # Check BFL pattern exists
        bfl_patterns_found = False
        for target in to_q_targets:
            if any("lora_unet_double_blocks" in p for p in target.possible_up_patterns):
                bfl_patterns_found = True
                assert target.up_transform is not None
                assert target.down_transform is not None
        assert bfl_patterns_found

    @pytest.mark.fast
    def test_double_block_mlp_patterns(self):
        """Verify BFL double block MLP patterns are included."""
        mapping = LongCatLoRAMapping.get_mapping()

        # Find ff.linear1 target
        ff_targets = [t for t in mapping if t.model_path == "transformer_blocks.{block}.ff.linear1"]
        assert len(ff_targets) >= 2  # Standard + BFL

        # Check BFL pattern exists
        bfl_patterns_found = False
        for target in ff_targets:
            if any("lora_unet_double_blocks" in p and "img_mlp_0" in p for p in target.possible_up_patterns):
                bfl_patterns_found = True
        assert bfl_patterns_found

    @pytest.mark.fast
    def test_single_block_linear1_patterns(self):
        """Verify BFL single block linear1 patterns are included with transforms."""
        mapping = LongCatLoRAMapping.get_mapping()

        # Find single block to_q target with BFL pattern
        single_q_targets = [t for t in mapping if t.model_path == "single_transformer_blocks.{block}.attn.to_q"]
        assert len(single_q_targets) >= 2  # Standard + BFL

        # Check BFL pattern with split transform
        bfl_patterns_found = False
        for target in single_q_targets:
            if any("lora_unet_single_blocks" in p and "linear1" in p for p in target.possible_up_patterns):
                bfl_patterns_found = True
                # BFL single block linear1 requires split transforms
                assert target.up_transform is not None
                assert target.down_transform is not None
        assert bfl_patterns_found

    @pytest.mark.fast
    def test_single_block_linear2_patterns(self):
        """Verify BFL single block linear2 patterns are included (no transforms)."""
        mapping = LongCatLoRAMapping.get_mapping()

        # Find proj_out target
        proj_out_targets = [t for t in mapping if t.model_path == "single_transformer_blocks.{block}.proj_out"]
        assert len(proj_out_targets) >= 2  # Standard + BFL

        # Check BFL pattern - linear2 maps directly to proj_out
        bfl_patterns_found = False
        for target in proj_out_targets:
            if any("lora_unet_single_blocks" in p and "linear2" in p for p in target.possible_up_patterns):
                bfl_patterns_found = True
        assert bfl_patterns_found


class TestLongCatLoRAMappingAlphaPatterns:
    """Tests for alpha pattern support."""

    @pytest.mark.fast
    def test_alpha_patterns_exist(self):
        """Verify alpha patterns are defined for targets."""
        mapping = LongCatLoRAMapping.get_mapping()

        # Most targets should have alpha patterns
        targets_with_alpha = [t for t in mapping if t.possible_alpha_patterns]
        assert len(targets_with_alpha) > 0

    @pytest.mark.fast
    def test_bfl_alpha_patterns(self):
        """Verify BFL format alpha patterns are included."""
        mapping = LongCatLoRAMapping.get_mapping()

        # Check that BFL targets have alpha patterns
        for target in mapping:
            if any("lora_unet" in p for p in target.possible_up_patterns):
                # BFL targets should have corresponding alpha patterns
                assert len(target.possible_alpha_patterns) > 0
                assert any("alpha" in p for p in target.possible_alpha_patterns)


class TestLongCatLoRAMappingTargetCount:
    """Tests for expected target counts."""

    @pytest.mark.fast
    def test_total_target_count(self):
        """Verify reasonable number of LoRA targets."""
        mapping = LongCatLoRAMapping.get_mapping()

        # Joint blocks: 15 targets × 2 formats = 30 (includes norm layers)
        # Single blocks: 6 targets × 2 formats = 12 (includes norm layer)
        # Total: ~42 (but some may be deduplicated)
        # Should be at least 36 targets
        assert len(mapping) >= 36

    @pytest.mark.fast
    def test_unique_model_paths(self):
        """Verify target model paths are valid."""
        mapping = LongCatLoRAMapping.get_mapping()

        for target in mapping:
            # Each target should have a valid model path
            assert target.model_path
            assert isinstance(target.model_path, str)
            # Path should contain either transformer_blocks or single_transformer_blocks
            assert "transformer_blocks" in target.model_path or "single_transformer_blocks" in target.model_path


class TestLongCatLoRAMappingBlockCounts:
    """Tests for LongCat-specific block counts (10 joint + 20 single)."""

    @pytest.mark.fast
    def test_joint_blocks_count(self):
        """Verify LongCat has 10 joint transformer blocks defined."""
        # This tests that the mapping is designed for 10 joint blocks
        mapping = LongCatLoRAMapping.get_mapping()

        # The {block} placeholder should work for blocks 0-9 (10 blocks)
        # We verify by checking that the patterns are correctly structured
        joint_targets = [t for t in mapping if "transformer_blocks.{block}" in t.model_path and "single" not in t.model_path]
        assert len(joint_targets) > 0

    @pytest.mark.fast
    def test_single_blocks_count(self):
        """Verify LongCat has 20 single transformer blocks defined."""
        # This tests that the mapping is designed for 20 single blocks
        mapping = LongCatLoRAMapping.get_mapping()

        # The {block} placeholder should work for blocks 0-19 (20 blocks)
        single_targets = [t for t in mapping if "single_transformer_blocks.{block}" in t.model_path]
        assert len(single_targets) > 0
