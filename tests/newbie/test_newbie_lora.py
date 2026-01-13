"""
Unit tests for NewBie-image LoRA mapping and loading.

Tests verify:
1. NewBieLoRAMapping covers all expected targets
2. Self-attention (attn1) targets are included with GQA naming
3. Cross-attention (attn2) targets are included with GQA naming
4. SwiGLU feed-forward targets are included
5. All 36 blocks have targets
"""

import pytest

from mflux.models.newbie.weights.newbie_lora_mapping import NewBieLoRAMapping


class TestNewBieLoRAMappingCoverage:
    """Tests for NewBieLoRAMapping target coverage."""

    @pytest.mark.fast
    def test_mapping_returns_non_empty_list(self):
        """Verify get_mapping returns a non-empty list of targets."""
        mapping = NewBieLoRAMapping.get_mapping()
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_correct_number_of_blocks(self):
        """Verify mapping targets 36 NextDiT blocks."""
        assert NewBieLoRAMapping.NUM_BLOCKS == 36


class TestNewBieSelfAttentionTargets:
    """Tests for NewBie self-attention (attn1) targets with GQA."""

    @pytest.mark.fast
    def test_self_attention_wq_targets(self):
        """Verify self-attention wq targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        wq_targets = [t for t in mapping if "attn1.wq" in t.model_path]
        assert len(wq_targets) == NewBieLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_self_attention_wk_targets(self):
        """Verify self-attention wk targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        wk_targets = [t for t in mapping if "attn1.wk" in t.model_path]
        assert len(wk_targets) == NewBieLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_self_attention_wv_targets(self):
        """Verify self-attention wv targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        wv_targets = [t for t in mapping if "attn1.wv" in t.model_path]
        assert len(wv_targets) == NewBieLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_self_attention_wo_targets(self):
        """Verify self-attention wo targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        wo_targets = [t for t in mapping if "attn1.wo" in t.model_path]
        assert len(wo_targets) == NewBieLoRAMapping.NUM_BLOCKS


class TestNewBieCrossAttentionTargets:
    """Tests for NewBie cross-attention (attn2) targets with GQA."""

    @pytest.mark.fast
    def test_cross_attention_wq_targets(self):
        """Verify cross-attention wq targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        wq_targets = [t for t in mapping if "attn2.wq" in t.model_path]
        assert len(wq_targets) == NewBieLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_cross_attention_wk_targets(self):
        """Verify cross-attention wk targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        wk_targets = [t for t in mapping if "attn2.wk" in t.model_path]
        assert len(wk_targets) == NewBieLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_cross_attention_wv_targets(self):
        """Verify cross-attention wv targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        wv_targets = [t for t in mapping if "attn2.wv" in t.model_path]
        assert len(wv_targets) == NewBieLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_cross_attention_wo_targets(self):
        """Verify cross-attention wo targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        wo_targets = [t for t in mapping if "attn2.wo" in t.model_path]
        assert len(wo_targets) == NewBieLoRAMapping.NUM_BLOCKS


class TestNewBieFeedForwardTargets:
    """Tests for NewBie SwiGLU feed-forward (ffn) targets."""

    @pytest.mark.fast
    def test_ffn_w1_targets(self):
        """Verify FFN w1 (gate) targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        w1_targets = [t for t in mapping if "ffn.w1" in t.model_path]
        assert len(w1_targets) == NewBieLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_ffn_w2_targets(self):
        """Verify FFN w2 (down) targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        w2_targets = [t for t in mapping if "ffn.w2" in t.model_path]
        assert len(w2_targets) == NewBieLoRAMapping.NUM_BLOCKS

    @pytest.mark.fast
    def test_ffn_w3_targets(self):
        """Verify FFN w3 (up) targets exist for all blocks."""
        mapping = NewBieLoRAMapping.get_mapping()
        w3_targets = [t for t in mapping if "ffn.w3" in t.model_path]
        assert len(w3_targets) == NewBieLoRAMapping.NUM_BLOCKS


class TestNewBieLoRAPatterns:
    """Tests for LoRA pattern definitions."""

    @pytest.mark.fast
    def test_all_targets_have_up_patterns(self):
        """Verify all targets have at least one up pattern."""
        mapping = NewBieLoRAMapping.get_mapping()
        for target in mapping:
            assert len(target.possible_up_patterns) > 0, f"Target {target.model_path} has no up patterns"

    @pytest.mark.fast
    def test_all_targets_have_down_patterns(self):
        """Verify all targets have at least one down pattern."""
        mapping = NewBieLoRAMapping.get_mapping()
        for target in mapping:
            assert len(target.possible_down_patterns) > 0, f"Target {target.model_path} has no down patterns"

    @pytest.mark.fast
    def test_all_targets_have_alpha_patterns(self):
        """Verify all targets have at least one alpha pattern."""
        mapping = NewBieLoRAMapping.get_mapping()
        for target in mapping:
            assert len(target.possible_alpha_patterns) > 0, f"Target {target.model_path} has no alpha patterns"


class TestNewBieLoRAMappingTargetCount:
    """Tests for expected target counts."""

    @pytest.mark.fast
    def test_total_target_count(self):
        """Verify total number of LoRA targets for NewBie-image.

        Expected targets per block:
        - Self-attention (attn1): 4 (wq, wk, wv, wo)
        - Cross-attention (attn2): 4 (wq, wk, wv, wo)
        - SwiGLU Feed-forward: 3 (w1, w2, w3)
        Total per block: 11
        Total for 36 blocks: 396
        """
        mapping = NewBieLoRAMapping.get_mapping()
        expected_per_block = 11
        expected_total = expected_per_block * NewBieLoRAMapping.NUM_BLOCKS
        assert len(mapping) == expected_total

    @pytest.mark.fast
    def test_unique_model_paths(self):
        """Verify all model paths are unique."""
        mapping = NewBieLoRAMapping.get_mapping()
        model_paths = [t.model_path for t in mapping]
        assert len(model_paths) == len(set(model_paths))


class TestNewBieNextDiTSpecificTargets:
    """Tests for NextDiT-specific architecture targets."""

    @pytest.mark.fast
    def test_block_indexing(self):
        """Verify block indices are correct (0-35)."""
        mapping = NewBieLoRAMapping.get_mapping()

        for block_idx in range(NewBieLoRAMapping.NUM_BLOCKS):
            block_targets = [t for t in mapping if f"blocks.{block_idx}." in t.model_path]
            # Each block should have 11 targets
            assert len(block_targets) == 11, f"Block {block_idx} has {len(block_targets)} targets, expected 11"

    @pytest.mark.fast
    def test_no_extra_blocks(self):
        """Verify no targets for blocks beyond 35."""
        mapping = NewBieLoRAMapping.get_mapping()

        for target in mapping:
            # Check that block index doesn't exceed 35
            import re
            match = re.search(r"blocks\.(\d+)\.", target.model_path)
            if match:
                block_idx = int(match.group(1))
                assert block_idx < NewBieLoRAMapping.NUM_BLOCKS, f"Found target for block {block_idx}"

    @pytest.mark.fast
    def test_gqa_naming_convention(self):
        """Verify GQA naming convention (wq, wk, wv, wo instead of to_q, etc.)."""
        mapping = NewBieLoRAMapping.get_mapping()

        attention_targets = [t for t in mapping if "attn" in t.model_path]
        for target in attention_targets:
            # Should use wq/wk/wv/wo naming
            assert any(proj in target.model_path for proj in ["wq", "wk", "wv", "wo"]), \
                f"Target {target.model_path} doesn't use GQA naming convention"

    @pytest.mark.fast
    def test_swiglu_naming_convention(self):
        """Verify SwiGLU naming convention (w1, w2, w3)."""
        mapping = NewBieLoRAMapping.get_mapping()

        ffn_targets = [t for t in mapping if "ffn" in t.model_path]
        for target in ffn_targets:
            # Should use w1/w2/w3 naming
            assert any(proj in target.model_path for proj in ["w1", "w2", "w3"]), \
                f"Target {target.model_path} doesn't use SwiGLU naming convention"
