"""
Unit tests for Hunyuan-DiT LoRA mapping and loading.

Tests verify:
1. HunyuanLoRAMapping covers all expected targets
2. Self-attention (attn1) targets are included
3. Cross-attention (attn2) targets are included
4. Feed-forward targets are included
5. All 28 blocks have targets
"""

import pytest

from mflux.models.hunyuan.weights.hunyuan_lora_mapping import HunyuanLoRAMapping


class TestHunyuanLoRAMappingCoverage:
    """Tests for HunyuanLoRAMapping target coverage."""

    @pytest.mark.fast
    def test_mapping_returns_non_empty_list(self):
        """Verify get_mapping returns a non-empty list of targets."""
        mapping = HunyuanLoRAMapping.get_mapping()
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_correct_number_of_blocks(self):
        """Verify mapping targets 28 DiT blocks."""
        assert HunyuanLoRAMapping.NUM_BLOCKS == 28


class TestHunyuanSelfAttentionTargets:
    """Tests for Hunyuan self-attention (attn1) targets."""

    @pytest.mark.fast
    def test_self_attention_to_q_targets(self):
        """Verify self-attention to_q targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        to_q_targets = [t for t in mapping if "attn1.to_q" in t.model_path]
        assert len(to_q_targets) == HunyuanLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_self_attention_to_k_targets(self):
        """Verify self-attention to_k targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        to_k_targets = [t for t in mapping if "attn1.to_k" in t.model_path]
        assert len(to_k_targets) == HunyuanLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_self_attention_to_v_targets(self):
        """Verify self-attention to_v targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        to_v_targets = [t for t in mapping if "attn1.to_v" in t.model_path]
        assert len(to_v_targets) == HunyuanLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_self_attention_to_out_targets(self):
        """Verify self-attention to_out targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        to_out_targets = [t for t in mapping if "attn1.to_out" in t.model_path]
        assert len(to_out_targets) == HunyuanLoRAMapping.NUM_BLOCKS


class TestHunyuanCrossAttentionTargets:
    """Tests for Hunyuan cross-attention (attn2) targets."""

    @pytest.mark.fast
    def test_cross_attention_to_q_targets(self):
        """Verify cross-attention to_q targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        to_q_targets = [t for t in mapping if "attn2.to_q" in t.model_path]
        assert len(to_q_targets) == HunyuanLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_cross_attention_to_k_targets(self):
        """Verify cross-attention to_k targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        to_k_targets = [t for t in mapping if "attn2.to_k" in t.model_path]
        assert len(to_k_targets) == HunyuanLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_cross_attention_to_v_targets(self):
        """Verify cross-attention to_v targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        to_v_targets = [t for t in mapping if "attn2.to_v" in t.model_path]
        assert len(to_v_targets) == HunyuanLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_cross_attention_to_out_targets(self):
        """Verify cross-attention to_out targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        to_out_targets = [t for t in mapping if "attn2.to_out" in t.model_path]
        assert len(to_out_targets) == HunyuanLoRAMapping.NUM_BLOCKS


class TestHunyuanFeedForwardTargets:
    """Tests for Hunyuan feed-forward (ff) targets."""

    @pytest.mark.fast
    def test_ff_net_0_proj_targets(self):
        """Verify FFN gate projection targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        ff_targets = [t for t in mapping if "ff.net_0_proj" in t.model_path]
        assert len(ff_targets) == HunyuanLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_ff_net_2_targets(self):
        """Verify FFN down projection targets exist for all blocks."""
        mapping = HunyuanLoRAMapping.get_mapping()
        ff_targets = [t for t in mapping if "ff.net_2" in t.model_path]
        assert len(ff_targets) == HunyuanLoRAMapping.NUM_BLOCKS


class TestHunyuanLoRAPatterns:
    """Tests for LoRA pattern definitions."""

    @pytest.mark.fast
    def test_all_targets_have_up_patterns(self):
        """Verify all targets have at least one up pattern."""
        mapping = HunyuanLoRAMapping.get_mapping()
        for target in mapping:
            assert len(target.possible_up_patterns) > 0, f"Target {target.model_path} has no up patterns"

    @pytest.mark.fast
    def test_all_targets_have_down_patterns(self):
        """Verify all targets have at least one down pattern."""
        mapping = HunyuanLoRAMapping.get_mapping()
        for target in mapping:
            assert len(target.possible_down_patterns) > 0, f"Target {target.model_path} has no down patterns"

    @pytest.mark.fast
    def test_all_targets_have_alpha_patterns(self):
        """Verify all targets have at least one alpha pattern."""
        mapping = HunyuanLoRAMapping.get_mapping()
        for target in mapping:
            assert len(target.possible_alpha_patterns) > 0, f"Target {target.model_path} has no alpha patterns"


class TestHunyuanLoRAMappingTargetCount:
    """Tests for expected target counts."""

    @pytest.mark.fast
    def test_total_target_count(self):
        """Verify total number of LoRA targets for Hunyuan-DiT.

        Expected targets per block:
        - Self-attention (attn1): 4 (to_q, to_k, to_v, to_out)
        - Cross-attention (attn2): 4 (to_q, to_k, to_v, to_out)
        - Feed-forward: 2 (net_0_proj, net_2)
        Total per block: 10
        Total for 28 blocks: 280
        """
        mapping = HunyuanLoRAMapping.get_mapping()
        expected_per_block = 10
        expected_total = expected_per_block * HunyuanLoRAMapping.NUM_BLOCKS
        assert len(mapping) == expected_total

    @pytest.mark.fast
    def test_unique_model_paths(self):
        """Verify all model paths are unique."""
        mapping = HunyuanLoRAMapping.get_mapping()
        model_paths = [t.model_path for t in mapping]
        assert len(model_paths) == len(set(model_paths))


class TestHunyuanDiTSpecificTargets:
    """Tests for DiT-specific architecture targets."""

    @pytest.mark.fast
    def test_block_indexing(self):
        """Verify block indices are correct (0-27)."""
        mapping = HunyuanLoRAMapping.get_mapping()

        for block_idx in range(HunyuanLoRAMapping.NUM_BLOCKS):
            block_targets = [t for t in mapping if f"blocks.{block_idx}." in t.model_path]
            # Each block should have 10 targets
            assert len(block_targets) == 10, f"Block {block_idx} has {len(block_targets)} targets, expected 10"

    @pytest.mark.fast
    def test_no_extra_blocks(self):
        """Verify no targets for blocks beyond 27."""
        mapping = HunyuanLoRAMapping.get_mapping()

        for target in mapping:
            # Check that block index doesn't exceed 27
            import re
            match = re.search(r"blocks\.(\d+)\.", target.model_path)
            if match:
                block_idx = int(match.group(1))
                assert block_idx < HunyuanLoRAMapping.NUM_BLOCKS, f"Found target for block {block_idx}"
