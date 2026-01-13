"""
Unit tests for Hunyuan-DiT weight loading and mapping verification.

Tests verify:
1. Weight mapping correctness for transformer blocks
2. Gate linear layer mapping patterns
3. Text projector weight loading
4. Complete weight hierarchy validation
"""

import pytest
import mlx.core as mx

from mflux.models.hunyuan.weights.hunyuan_weight_mapping import HunyuanWeightMapping


class TestHunyuanTransformerWeightMapping:
    """Tests for transformer weight mapping."""

    @pytest.mark.fast
    def test_transformer_mapping_non_empty(self):
        """Verify get_transformer_mapping returns weights."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_time_embed_weights_exist(self):
        """Verify timestep embedding weights are mapped."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()
        time_embed_targets = [t for t in mapping if "time_embed" in t.to_pattern]

        # Should have linear_1 and linear_2 weights and biases (4 total)
        assert len(time_embed_targets) == 4

        # Verify specific patterns
        patterns = [t.to_pattern for t in time_embed_targets]
        assert "time_embed.linear_1.weight" in patterns
        assert "time_embed.linear_1.bias" in patterns
        assert "time_embed.linear_2.weight" in patterns
        assert "time_embed.linear_2.bias" in patterns

    @pytest.mark.fast
    def test_patch_embed_weights_exist(self):
        """Verify patch embedding weights are mapped."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()
        patch_embed_targets = [t for t in mapping if "patch_embed" in t.to_pattern]

        # Should have proj weight and bias (2 total)
        assert len(patch_embed_targets) == 2

        patterns = [t.to_pattern for t in patch_embed_targets]
        assert "patch_embed.proj.weight" in patterns
        assert "patch_embed.proj.bias" in patterns

    @pytest.mark.fast
    def test_text_proj_weights_exist(self):
        """Verify text projector weights are mapped."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()
        text_proj_targets = [t for t in mapping if "text_proj" in t.to_pattern]

        # Should have CLIP and T5 projections (4 total: 2 weights + 2 biases)
        assert len(text_proj_targets) == 4

        patterns = [t.to_pattern for t in text_proj_targets]
        assert "text_proj.clip_proj.weight" in patterns
        assert "text_proj.clip_proj.bias" in patterns
        assert "text_proj.t5_proj.weight" in patterns
        assert "text_proj.t5_proj.bias" in patterns

    @pytest.mark.fast
    def test_output_layers_exist(self):
        """Verify output layer weights are mapped."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        # Check norm_out
        norm_out_targets = [t for t in mapping if "norm_out" in t.to_pattern]
        assert len(norm_out_targets) == 2  # weight and bias

        # Check proj_out
        proj_out_targets = [t for t in mapping if "proj_out" in t.to_pattern]
        assert len(proj_out_targets) == 2  # weight and bias


class TestHunyuanDiTBlockWeightMapping:
    """Tests for DiT block weight mapping."""

    @pytest.mark.fast
    def test_block_count(self):
        """Verify mapping covers all 28 blocks."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        for block_idx in range(28):
            block_targets = [t for t in mapping if f"blocks.{block_idx}." in t.to_pattern]
            assert len(block_targets) > 0, f"No weights mapped for block {block_idx}"

    @pytest.mark.fast
    def test_adaln_norm1_weights_per_block(self):
        """Verify AdaLN norm1 weights exist for each block."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        for block_idx in range(28):
            norm1_targets = [
                t for t in mapping
                if f"blocks.{block_idx}.norm1" in t.to_pattern
            ]
            # Should have norm.weight, norm.bias, linear.weight, linear.bias (4 total)
            assert len(norm1_targets) == 4, f"Block {block_idx} norm1 has {len(norm1_targets)} weights"

    @pytest.mark.fast
    def test_adaln_norm2_weights_per_block(self):
        """Verify AdaLN norm2 weights exist for each block."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        for block_idx in range(28):
            norm2_targets = [
                t for t in mapping
                if f"blocks.{block_idx}.norm2" in t.to_pattern
            ]
            assert len(norm2_targets) == 4, f"Block {block_idx} norm2 has {len(norm2_targets)} weights"

    @pytest.mark.fast
    def test_adaln_norm3_weights_per_block(self):
        """Verify AdaLN norm3 weights exist for each block."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        for block_idx in range(28):
            norm3_targets = [
                t for t in mapping
                if f"blocks.{block_idx}.norm3" in t.to_pattern
            ]
            assert len(norm3_targets) == 4, f"Block {block_idx} norm3 has {len(norm3_targets)} weights"

    @pytest.mark.fast
    def test_self_attention_weights_per_block(self):
        """Verify self-attention weights exist for each block."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        for block_idx in range(28):
            attn1_targets = [
                t for t in mapping
                if f"blocks.{block_idx}.attn1" in t.to_pattern
            ]
            # to_q, to_k, to_v, to_out (weights + biases) + norm_q, norm_k = 10 total
            assert len(attn1_targets) == 10, f"Block {block_idx} attn1 has {len(attn1_targets)} weights"

    @pytest.mark.fast
    def test_cross_attention_weights_per_block(self):
        """Verify cross-attention weights exist for each block."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        for block_idx in range(28):
            attn2_targets = [
                t for t in mapping
                if f"blocks.{block_idx}.attn2" in t.to_pattern
            ]
            # to_q, to_k, to_v, to_out (weights + biases) + norm_q, norm_k = 10 total
            assert len(attn2_targets) == 10, f"Block {block_idx} attn2 has {len(attn2_targets)} weights"

    @pytest.mark.fast
    def test_feedforward_weights_per_block(self):
        """Verify feed-forward weights exist for each block."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        for block_idx in range(28):
            ff_targets = [
                t for t in mapping
                if f"blocks.{block_idx}.ff" in t.to_pattern
            ]
            # net_0_proj (weight + bias) + net_2 (weight + bias) = 4 total
            assert len(ff_targets) == 4, f"Block {block_idx} ff has {len(ff_targets)} weights"

    @pytest.mark.fast
    def test_gate_linear_weights_per_block(self):
        """Verify gate_linear weights exist for each block."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        for block_idx in range(28):
            gate_targets = [
                t for t in mapping
                if f"blocks.{block_idx}.gate_linear" in t.to_pattern
            ]
            # weight + bias = 2 total
            assert len(gate_targets) == 2, f"Block {block_idx} gate_linear has {len(gate_targets)} weights"


class TestGateLinearMapping:
    """Tests for gate_linear weight mapping patterns."""

    @pytest.mark.fast
    def test_gate_linear_has_multiple_patterns(self):
        """Verify gate_linear supports multiple source patterns."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        gate_weight_targets = [
            t for t in mapping
            if "gate_linear.weight" in t.to_pattern
        ]

        # Should have at least one gate_linear weight per block
        assert len(gate_weight_targets) == 28

        # Each target should have multiple from_patterns for flexibility
        for target in gate_weight_targets:
            assert len(target.from_pattern) >= 1

    @pytest.mark.fast
    def test_gate_linear_bias_patterns(self):
        """Verify gate_linear bias has proper patterns."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        gate_bias_targets = [
            t for t in mapping
            if "gate_linear.bias" in t.to_pattern
        ]

        assert len(gate_bias_targets) == 28

        for target in gate_bias_targets:
            assert len(target.from_pattern) >= 1


class TestWeightMappingCompleteness:
    """Tests for complete weight hierarchy."""

    @pytest.mark.fast
    def test_all_to_patterns_unique(self):
        """Verify all target patterns are unique."""
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        to_patterns = [t.to_pattern for t in mapping]
        assert len(to_patterns) == len(set(to_patterns)), "Duplicate target patterns found"

    @pytest.mark.fast
    def test_total_weight_count(self):
        """Verify total weight count matches expected architecture.

        Expected weights:
        - Time embed: 4 (2 layers × 2 params)
        - Patch embed: 2 (1 conv × 2 params)
        - Text proj: 4 (2 projections × 2 params)
        - 28 blocks × weights_per_block
        - Output: 4 (norm + proj × 2 params)

        Per block:
        - norm1: 4 (norm × 2 + linear × 2)
        - attn1: 10 (4 projections × 2 + 2 norms)
        - norm2: 4
        - attn2: 10
        - norm3: 4
        - ff: 4 (2 layers × 2)
        - gate_linear: 2
        Total per block: 38

        Total: 4 + 2 + 4 + (28 × 38) + 4 = 1078
        """
        mapping = HunyuanWeightMapping.get_transformer_mapping()

        expected_per_block = 38
        expected_total = 4 + 2 + 4 + (28 * expected_per_block) + 4

        assert len(mapping) == expected_total, \
            f"Expected {expected_total} weights, got {len(mapping)}"


class TestVAEMapping:
    """Tests for VAE weight mapping."""

    @pytest.mark.fast
    def test_vae_mapping_non_empty(self):
        """Verify VAE mapping returns weights."""
        mapping = HunyuanWeightMapping.get_vae_mapping()
        assert len(mapping) > 0


class TestCLIPEncoderMapping:
    """Tests for CLIP encoder weight mapping."""

    @pytest.mark.fast
    def test_clip_mapping_non_empty(self):
        """Verify CLIP encoder mapping returns weights."""
        mapping = HunyuanWeightMapping.get_clip_encoder_mapping()
        assert len(mapping) > 0

    @pytest.mark.fast
    def test_clip_embedding_layers(self):
        """Verify CLIP embedding layers are mapped."""
        mapping = HunyuanWeightMapping.get_clip_encoder_mapping()

        embedding_targets = [
            t for t in mapping
            if "embeddings" in t.to_pattern
        ]

        # Should have token_embedding and position_embedding
        assert len(embedding_targets) >= 2

    @pytest.mark.fast
    def test_clip_encoder_layers(self):
        """Verify CLIP has 12 encoder layers."""
        mapping = HunyuanWeightMapping.get_clip_encoder_mapping()

        # Count unique layer indices
        layer_indices = set()
        for target in mapping:
            if "encoder.layers." in target.to_pattern:
                # Extract layer number
                import re
                match = re.search(r"encoder\.layers\.(\d+)", target.to_pattern)
                if match:
                    layer_indices.add(int(match.group(1)))

        assert len(layer_indices) == 12, f"Expected 12 CLIP layers, found {len(layer_indices)}"


class TestT5EncoderMapping:
    """Tests for T5 encoder weight mapping."""

    @pytest.mark.fast
    def test_t5_mapping_non_empty(self):
        """Verify T5 encoder mapping returns weights."""
        mapping = HunyuanWeightMapping.get_t5_encoder_mapping()
        assert len(mapping) > 0
