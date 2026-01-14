"""
Unit tests for FLUX.2 LoRA mapping and loading.

Tests verify:
1. Flux2LoRAMapping covers all expected targets
2. Global modulation layer targets are included
3. Fused QKV+MLP projection targets in single blocks
4. BFL/Kohya format patterns are correctly defined
"""

import pytest

from mflux.models.flux2.weights.flux2_lora_mapping import Flux2LoRAMapping


class TestFlux2LoRAMappingCoverage:
    """Tests for Flux2LoRAMapping target coverage."""

    @pytest.mark.fast
    def test_mapping_returns_non_empty_list(self):
        """Verify get_mapping returns a non-empty list of targets."""
        mapping = Flux2LoRAMapping.get_mapping()
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_joint_block_attention_targets(self):
        """Verify joint transformer block attention targets are included (8 blocks)."""
        mapping = Flux2LoRAMapping.get_mapping()
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
        """Verify joint transformer block feed-forward targets use FLUX.2 naming."""
        mapping = Flux2LoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        # FLUX.2 uses linear_in/linear_out instead of linear1/linear2
        assert "transformer_blocks.{block}.ff.linear_in" in paths
        assert "transformer_blocks.{block}.ff.linear_out" in paths

        # Text/context FFN
        assert "transformer_blocks.{block}.ff_context.linear_in" in paths
        assert "transformer_blocks.{block}.ff_context.linear_out" in paths


class TestFlux2SingleBlockTargets:
    """Tests for FLUX.2's fused single block attention."""

    @pytest.mark.fast
    def test_fused_qkv_mlp_projection(self):
        """Verify single blocks use fused to_qkv_mlp_proj (FLUX.2 specific)."""
        mapping = Flux2LoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        # FLUX.2 uses fused QKV+MLP projection instead of separate Q, K, V, MLP
        assert "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj" in paths

    @pytest.mark.fast
    def test_single_block_output_targets(self):
        """Verify single block output projection target exists."""
        mapping = Flux2LoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        # FLUX.2 single blocks use to_out instead of proj_out
        assert "single_transformer_blocks.{block}.attn.to_out" in paths


class TestFlux2GlobalModulationTargets:
    """Tests for FLUX.2's global modulation layer targets (unique to FLUX.2)."""

    @pytest.mark.fast
    def test_double_stream_img_modulation(self):
        """Verify double stream image modulation is included."""
        mapping = Flux2LoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        assert "double_stream_modulation_img.linear" in paths

    @pytest.mark.fast
    def test_double_stream_txt_modulation(self):
        """Verify double stream text modulation is included."""
        mapping = Flux2LoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        assert "double_stream_modulation_txt.linear" in paths

    @pytest.mark.fast
    def test_single_stream_modulation(self):
        """Verify single stream modulation is included."""
        mapping = Flux2LoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        assert "single_stream_modulation.linear" in paths


class TestFlux2LoRAMappingBFLPatterns:
    """Tests for BFL/Kohya format pattern coverage."""

    @pytest.mark.fast
    def test_double_block_qkv_patterns(self):
        """Verify BFL double block QKV patterns are included."""
        mapping = Flux2LoRAMapping.get_mapping()

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
        mapping = Flux2LoRAMapping.get_mapping()

        # Find ff.linear_in target (FLUX.2 naming)
        ff_targets = [t for t in mapping if t.model_path == "transformer_blocks.{block}.ff.linear_in"]
        assert len(ff_targets) >= 2  # Standard + BFL

        # Check BFL pattern exists
        bfl_patterns_found = False
        for target in ff_targets:
            if any("lora_unet_double_blocks" in p and "img_mlp_0" in p for p in target.possible_up_patterns):
                bfl_patterns_found = True
        assert bfl_patterns_found

    @pytest.mark.fast
    def test_single_block_linear1_patterns(self):
        """Verify BFL single block linear1 patterns map to fused projection."""
        mapping = Flux2LoRAMapping.get_mapping()

        # Find fused QKV+MLP projection target with BFL pattern
        fused_targets = [t for t in mapping if t.model_path == "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj"]
        assert len(fused_targets) >= 2  # Standard + BFL

        # Check BFL pattern exists
        bfl_patterns_found = False
        for target in fused_targets:
            if any("lora_unet_single_blocks" in p and "linear1" in p for p in target.possible_up_patterns):
                bfl_patterns_found = True
        assert bfl_patterns_found

    @pytest.mark.fast
    def test_single_block_linear2_patterns(self):
        """Verify BFL single block linear2 patterns map to to_out."""
        mapping = Flux2LoRAMapping.get_mapping()

        # Find to_out target
        to_out_targets = [t for t in mapping if t.model_path == "single_transformer_blocks.{block}.attn.to_out"]
        assert len(to_out_targets) >= 2  # Standard + BFL

        # Check BFL pattern exists
        bfl_patterns_found = False
        for target in to_out_targets:
            if any("lora_unet_single_blocks" in p and "linear2" in p for p in target.possible_up_patterns):
                bfl_patterns_found = True
        assert bfl_patterns_found


class TestFlux2LoRAMappingAlphaPatterns:
    """Tests for alpha pattern support."""

    @pytest.mark.fast
    def test_alpha_patterns_exist(self):
        """Verify alpha patterns are defined for targets."""
        mapping = Flux2LoRAMapping.get_mapping()

        # Most targets should have alpha patterns
        targets_with_alpha = [t for t in mapping if t.possible_alpha_patterns]
        assert len(targets_with_alpha) > 0

    @pytest.mark.fast
    def test_bfl_alpha_patterns(self):
        """Verify BFL format alpha patterns are included."""
        mapping = Flux2LoRAMapping.get_mapping()

        # Check that BFL targets have alpha patterns
        for target in mapping:
            if any("lora_unet" in p for p in target.possible_up_patterns):
                # BFL targets should have corresponding alpha patterns
                assert len(target.possible_alpha_patterns) > 0
                assert any("alpha" in p for p in target.possible_alpha_patterns)


class TestFlux2LoRAMappingTargetCount:
    """Tests for expected target counts."""

    @pytest.mark.fast
    def test_total_target_count(self):
        """Verify reasonable number of LoRA targets for FLUX.2."""
        mapping = Flux2LoRAMapping.get_mapping()

        # Joint blocks: 12 targets × 2 formats = 24 (FLUX.2 doesn't have norm layer targets)
        # Single blocks: 2 targets × 2 formats = 4 (fused QKV+MLP + to_out)
        # Global modulation: 3 targets
        # Total: ~31 minimum
        assert len(mapping) >= 25

    @pytest.mark.fast
    def test_unique_model_paths(self):
        """Verify target model paths are valid."""
        mapping = Flux2LoRAMapping.get_mapping()

        for target in mapping:
            # Each target should have a valid model path
            assert target.model_path
            assert isinstance(target.model_path, str)
            # Path should be for transformer blocks, single blocks, or global modulation
            valid_prefixes = [
                "transformer_blocks",
                "single_transformer_blocks",
                "double_stream_modulation",
                "single_stream_modulation",
            ]
            assert any(prefix in target.model_path for prefix in valid_prefixes)


class TestFlux2LoRAMappingBlockCounts:
    """Tests for FLUX.2-specific block counts (8 joint + 48 single)."""

    @pytest.mark.fast
    def test_joint_blocks_targets(self):
        """Verify joint transformer block targets exist for 8 blocks."""
        mapping = Flux2LoRAMapping.get_mapping()

        # The {block} placeholder should work for blocks 0-7 (8 blocks)
        joint_targets = [
            t for t in mapping
            if "transformer_blocks.{block}" in t.model_path
            and "single" not in t.model_path
        ]
        assert len(joint_targets) > 0

    @pytest.mark.fast
    def test_single_blocks_targets(self):
        """Verify single transformer block targets exist for 48 blocks."""
        mapping = Flux2LoRAMapping.get_mapping()

        # The {block} placeholder should work for blocks 0-47 (48 blocks)
        single_targets = [t for t in mapping if "single_transformer_blocks.{block}" in t.model_path]
        assert len(single_targets) > 0


class TestFlux2VsFLUX1Differences:
    """Tests that verify FLUX.2 differences from FLUX.1."""

    @pytest.mark.fast
    def test_no_norm1_linear_targets(self):
        """Verify FLUX.2 does NOT have per-block norm layers (uses global modulation)."""
        mapping = Flux2LoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        # FLUX.2 uses global modulation instead of per-block norm layers
        assert "transformer_blocks.{block}.norm1.linear" not in paths
        assert "transformer_blocks.{block}.norm1_context.linear" not in paths
        assert "single_transformer_blocks.{block}.norm.linear" not in paths

    @pytest.mark.fast
    def test_linear_in_out_naming(self):
        """Verify FLUX.2 uses linear_in/linear_out instead of linear1/linear2."""
        mapping = Flux2LoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        # Should use linear_in/linear_out
        assert any("linear_in" in p for p in paths)
        assert any("linear_out" in p for p in paths)

        # Should NOT use linear1/linear2 in model paths (only in BFL patterns)
        model_path_str = " ".join(paths)
        assert "ff.linear1" not in model_path_str
        assert "ff.linear2" not in model_path_str

    @pytest.mark.fast
    def test_fused_single_block_projection(self):
        """Verify FLUX.2 uses fused to_qkv_mlp_proj in single blocks."""
        mapping = Flux2LoRAMapping.get_mapping()
        paths = [t.model_path for t in mapping]

        # Should have fused projection
        assert "single_transformer_blocks.{block}.attn.to_qkv_mlp_proj" in paths

        # Should NOT have separate to_q, to_k, to_v, proj_mlp for single blocks
        # (only the fused version)
        single_block_paths = [p for p in paths if "single_transformer_blocks" in p]
        assert "single_transformer_blocks.{block}.attn.to_q" not in single_block_paths
        assert "single_transformer_blocks.{block}.proj_mlp" not in single_block_paths
