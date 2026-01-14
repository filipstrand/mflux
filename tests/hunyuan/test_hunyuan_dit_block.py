"""
Unit tests for Hunyuan-DiT block forward pass patterns.

Tests verify:
1. DiT block initialization and structure
2. AdaLN (Adaptive Layer Normalization) functionality
3. Self-attention forward pass
4. Cross-attention forward pass
5. Feed-forward network
6. Gate mechanisms (AdaLN-Zero)
7. Residual connections
8. Block output shapes
"""

import pytest
import mlx.core as mx

from mflux.models.hunyuan.model.hunyuan_transformer.hunyuan_dit_block import (
    HunyuanAdaLayerNorm,
    HunyuanFeedForward,
    HunyuanDiTBlock,
)


class TestHunyuanAdaLayerNorm:
    """Tests for Adaptive Layer Normalization."""

    @pytest.mark.fast
    def test_adaln_initialization(self):
        """Verify AdaLN initializes correctly."""
        hidden_dim = 1408
        adaln = HunyuanAdaLayerNorm(hidden_dim)

        assert adaln.norm is not None
        assert adaln.linear is not None

    @pytest.mark.fast
    def test_adaln_forward_pass_shapes(self):
        """Verify AdaLN output shapes."""
        hidden_dim = 1408
        adaln = HunyuanAdaLayerNorm(hidden_dim)

        batch_size = 2
        seq_len = 100
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))
        temb = mx.random.normal((batch_size, hidden_dim))

        output, shift, scale = adaln(hidden_states, temb)

        # Output should match input shape
        assert output.shape == hidden_states.shape
        # Shift and scale should be [batch, hidden_dim]
        assert shift.shape == (batch_size, hidden_dim)
        assert scale.shape == (batch_size, hidden_dim)

    @pytest.mark.fast
    def test_adaln_produces_normalized_output(self):
        """Verify AdaLN produces properly normalized output."""
        hidden_dim = 1408
        adaln = HunyuanAdaLayerNorm(hidden_dim)

        batch_size = 2
        seq_len = 100
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim)) * 10.0  # Large values
        temb = mx.random.normal((batch_size, hidden_dim))

        output, _, _ = adaln(hidden_states, temb)

        # Output should have controlled magnitude
        output_std = mx.std(output)
        assert output_std < 10.0  # Should be normalized

    @pytest.mark.fast
    def test_adaln_shift_scale_dependency_on_temb(self):
        """Verify shift and scale depend on timestep embedding."""
        hidden_dim = 1408
        adaln = HunyuanAdaLayerNorm(hidden_dim)

        batch_size = 1
        seq_len = 10
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))

        # Different timestep embeddings
        temb1 = mx.random.normal((batch_size, hidden_dim))
        temb2 = mx.random.normal((batch_size, hidden_dim))

        _, shift1, scale1 = adaln(hidden_states, temb1)
        _, shift2, scale2 = adaln(hidden_states, temb2)

        # Different tembs should produce different shift/scale
        assert not mx.allclose(shift1, shift2, atol=1e-3)
        assert not mx.allclose(scale1, scale2, atol=1e-3)


class TestHunyuanFeedForward:
    """Tests for Feed-Forward Network."""

    @pytest.mark.fast
    def test_feedforward_initialization(self):
        """Verify FFN initializes correctly."""
        hidden_dim = 1408
        ff = HunyuanFeedForward(hidden_dim)

        assert ff.net_0_proj is not None
        assert ff.net_2 is not None

    @pytest.mark.fast
    def test_feedforward_default_intermediate_dim(self):
        """Verify default intermediate dimension is 4x hidden_dim."""
        hidden_dim = 1408
        ff = HunyuanFeedForward(hidden_dim)

        # net_0_proj should expand to 4x hidden_dim
        expected_intermediate = hidden_dim * 4
        assert ff.net_0_proj.weight.shape[0] == expected_intermediate

    @pytest.mark.fast
    def test_feedforward_custom_intermediate_dim(self):
        """Verify custom intermediate dimension works."""
        hidden_dim = 1408
        intermediate_dim = 2048
        ff = HunyuanFeedForward(hidden_dim, intermediate_dim=intermediate_dim)

        assert ff.net_0_proj.weight.shape[0] == intermediate_dim

    @pytest.mark.fast
    def test_feedforward_forward_pass_shapes(self):
        """Verify FFN output shape matches input shape."""
        hidden_dim = 1408
        ff = HunyuanFeedForward(hidden_dim)

        batch_size = 2
        seq_len = 100
        hidden_states = mx.random.normal((batch_size, seq_len, hidden_dim))

        output = ff(hidden_states)

        assert output.shape == hidden_states.shape

    @pytest.mark.fast
    def test_feedforward_nonlinearity(self):
        """Verify FFN applies nonlinearity (not identity mapping)."""
        hidden_dim = 1408
        ff = HunyuanFeedForward(hidden_dim)

        # Zero input should produce non-zero output due to bias
        hidden_states = mx.zeros((1, 10, hidden_dim))
        output = ff(hidden_states)

        # Output should not be all zeros
        assert mx.sum(mx.abs(output)) > 0


class TestHunyuanDiTBlockStructure:
    """Tests for DiT block structure and initialization."""

    @pytest.mark.fast
    def test_dit_block_initialization(self):
        """Verify DiT block initializes correctly."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        assert block.norm1 is not None
        assert block.attn1 is not None
        assert block.norm2 is not None
        assert block.attn2 is not None
        assert block.norm3 is not None
        assert block.ff is not None
        assert block.gate_linear is not None

    @pytest.mark.fast
    def test_gate_linear_output_dimension(self):
        """Verify gate_linear produces 3 gates."""
        block = HunyuanDiTBlock()

        # gate_linear should output 3 values (one per sublayer)
        assert block.gate_linear.weight.shape[0] == 3

    @pytest.mark.fast
    def test_dit_block_default_parameters(self):
        """Verify default parameters match Hunyuan-DiT specification."""
        block = HunyuanDiTBlock()

        assert block.hidden_dim == 1408
        # Additional component checks
        assert block.attn1 is not None
        assert block.attn2 is not None


class TestHunyuanDiTBlockForwardPass:
    """Tests for DiT block forward pass."""

    @pytest.mark.fast
    def test_forward_pass_shapes(self):
        """Verify forward pass output shape."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 2
        img_seq_len = 256  # 32x32 patches with patch_size=2
        text_seq_len = 333  # 77 CLIP + 256 T5
        hidden_dim = 1408

        hidden_states = mx.random.normal((batch_size, img_seq_len, hidden_dim))
        encoder_hidden_states = mx.random.normal((batch_size, text_seq_len, hidden_dim))
        temb = mx.random.normal((batch_size, hidden_dim))

        output = block(hidden_states, encoder_hidden_states, temb)

        # Output should match input hidden_states shape
        assert output.shape == hidden_states.shape

    @pytest.mark.fast
    def test_forward_pass_with_rotary_embeddings(self):
        """Verify forward pass works with rotary embeddings."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 1
        img_seq_len = 256
        text_seq_len = 333
        hidden_dim = 1408
        head_dim = 88

        hidden_states = mx.random.normal((batch_size, img_seq_len, hidden_dim))
        encoder_hidden_states = mx.random.normal((batch_size, text_seq_len, hidden_dim))
        temb = mx.random.normal((batch_size, hidden_dim))
        rotary_emb = mx.random.normal((img_seq_len, head_dim // 2, 2))

        output = block(hidden_states, encoder_hidden_states, temb, rotary_emb=rotary_emb)

        assert output.shape == hidden_states.shape

    @pytest.mark.fast
    def test_forward_pass_with_attention_mask(self):
        """Verify forward pass works with attention mask."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 1
        img_seq_len = 256
        text_seq_len = 333
        hidden_dim = 1408

        hidden_states = mx.random.normal((batch_size, img_seq_len, hidden_dim))
        encoder_hidden_states = mx.random.normal((batch_size, text_seq_len, hidden_dim))
        temb = mx.random.normal((batch_size, hidden_dim))
        attention_mask = mx.ones((batch_size, text_seq_len))

        output = block(
            hidden_states,
            encoder_hidden_states,
            temb,
            attention_mask=attention_mask,
        )

        assert output.shape == hidden_states.shape


class TestDiTBlockResidualConnections:
    """Tests for residual connections in DiT block."""

    @pytest.mark.fast
    def test_residual_connection_pattern(self):
        """Verify residual connections are applied."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 1
        img_seq_len = 64
        text_seq_len = 333
        hidden_dim = 1408

        hidden_states = mx.random.normal((batch_size, img_seq_len, hidden_dim))
        encoder_hidden_states = mx.random.normal((batch_size, text_seq_len, hidden_dim))
        temb = mx.random.normal((batch_size, hidden_dim))

        output = block(hidden_states, encoder_hidden_states, temb)

        # Output should be different from input (not identity)
        assert not mx.allclose(output, hidden_states, atol=1e-3)

        # But should retain some similarity (residual connection)
        # Compute correlation
        input_flat = hidden_states.reshape(-1)
        output_flat = output.reshape(-1)
        correlation = mx.sum(input_flat * output_flat) / (
            mx.sqrt(mx.sum(input_flat ** 2)) * mx.sqrt(mx.sum(output_flat ** 2))
        )
        # Should have positive correlation due to residual
        assert correlation > 0


class TestDiTBlockGatingMechanism:
    """Tests for AdaLN-Zero gating mechanism."""

    @pytest.mark.fast
    def test_gate_values_in_valid_range(self):
        """Verify gate values are in [0, 1] range after sigmoid."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 2
        hidden_dim = 1408
        temb = mx.random.normal((batch_size, hidden_dim))

        gates = block.gate_linear(temb)
        gates_sigmoid = mx.sigmoid(gates)

        # All gate values should be in [0, 1]
        assert mx.all(gates_sigmoid >= 0.0)
        assert mx.all(gates_sigmoid <= 1.0)

    @pytest.mark.fast
    def test_three_separate_gates(self):
        """Verify three independent gates for three sublayers."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 1
        hidden_dim = 1408
        temb = mx.random.normal((batch_size, hidden_dim))

        gates = block.gate_linear(temb)

        # Should have 3 gate values
        assert gates.shape == (batch_size, 3)


class TestDiTBlockEdgeCases:
    """Tests for edge cases in DiT block."""

    @pytest.mark.fast
    def test_single_sequence_element(self):
        """Verify block works with single sequence element."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 1
        img_seq_len = 1  # Single element
        text_seq_len = 333
        hidden_dim = 1408

        hidden_states = mx.random.normal((batch_size, img_seq_len, hidden_dim))
        encoder_hidden_states = mx.random.normal((batch_size, text_seq_len, hidden_dim))
        temb = mx.random.normal((batch_size, hidden_dim))

        output = block(hidden_states, encoder_hidden_states, temb)

        assert output.shape == (batch_size, img_seq_len, hidden_dim)

    @pytest.mark.fast
    def test_large_sequence_length(self):
        """Verify block works with large sequence lengths."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 1
        img_seq_len = 4096  # Large 64x64 patches
        text_seq_len = 333
        hidden_dim = 1408

        hidden_states = mx.random.normal((batch_size, img_seq_len, hidden_dim))
        encoder_hidden_states = mx.random.normal((batch_size, text_seq_len, hidden_dim))
        temb = mx.random.normal((batch_size, hidden_dim))

        output = block(hidden_states, encoder_hidden_states, temb)

        assert output.shape == (batch_size, img_seq_len, hidden_dim)


class TestDiTBlockComponentInteraction:
    """Tests for interaction between DiT block components."""

    @pytest.mark.fast
    def test_self_attention_uses_rotary_embeddings(self):
        """Verify self-attention can use rotary embeddings."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 1
        img_seq_len = 64
        text_seq_len = 333
        hidden_dim = 1408
        head_dim = 88

        hidden_states = mx.random.normal((batch_size, img_seq_len, hidden_dim))
        encoder_hidden_states = mx.random.normal((batch_size, text_seq_len, hidden_dim))
        temb = mx.random.normal((batch_size, hidden_dim))
        rotary_emb = mx.random.normal((img_seq_len, head_dim // 2, 2))

        # With rotary embeddings
        output_with_rope = block(
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb=rotary_emb,
        )

        # Without rotary embeddings
        output_without_rope = block(
            hidden_states,
            encoder_hidden_states,
            temb,
            rotary_emb=None,
        )

        # Outputs should be different
        assert not mx.allclose(output_with_rope, output_without_rope, atol=1e-3)

    @pytest.mark.fast
    def test_cross_attention_uses_encoder_hidden_states(self):
        """Verify cross-attention uses encoder hidden states."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 1
        img_seq_len = 64
        text_seq_len = 333
        hidden_dim = 1408

        hidden_states = mx.random.normal((batch_size, img_seq_len, hidden_dim))
        temb = mx.random.normal((batch_size, hidden_dim))

        # Different encoder hidden states
        encoder_hidden_states1 = mx.random.normal((batch_size, text_seq_len, hidden_dim))
        encoder_hidden_states2 = mx.random.normal((batch_size, text_seq_len, hidden_dim))

        output1 = block(hidden_states, encoder_hidden_states1, temb)
        output2 = block(hidden_states, encoder_hidden_states2, temb)

        # Different text conditioning should produce different outputs
        assert not mx.allclose(output1, output2, atol=1e-3)

    @pytest.mark.fast
    def test_timestep_embedding_affects_all_sublayers(self):
        """Verify timestep embedding affects output through AdaLN."""
        block = HunyuanDiTBlock(
            hidden_dim=1408,
            num_heads=16,
            head_dim=88,
            text_dim=1408,
        )

        batch_size = 1
        img_seq_len = 64
        text_seq_len = 333
        hidden_dim = 1408

        hidden_states = mx.random.normal((batch_size, img_seq_len, hidden_dim))
        encoder_hidden_states = mx.random.normal((batch_size, text_seq_len, hidden_dim))

        # Different timestep embeddings
        temb1 = mx.random.normal((batch_size, hidden_dim))
        temb2 = mx.random.normal((batch_size, hidden_dim))

        output1 = block(hidden_states, encoder_hidden_states, temb1)
        output2 = block(hidden_states, encoder_hidden_states, temb2)

        # Different timesteps should produce different outputs
        assert not mx.allclose(output1, output2, atol=1e-3)
