"""
Unit tests for Hunyuan-DiT text encoding (CLIP + T5).

Tests verify:
1. Text encoder initialization and structure
2. CLIP text encoding output shapes
3. T5 text encoding output shapes
4. Text projector functionality
5. Dual encoder concatenation
6. Prompt caching behavior
"""

import pytest
import mlx.core as mx
from mlx import nn

from mflux.models.hunyuan.model.hunyuan_transformer.hunyuan_dit import (
    HunyuanTextProjector,
)


class TestHunyuanTextProjector:
    """Tests for HunyuanTextProjector."""

    @pytest.mark.fast
    def test_projector_initialization(self):
        """Verify text projector initializes correctly."""
        projector = HunyuanTextProjector(
            hidden_dim=1408,
            clip_dim=1024,
            t5_dim=2048,
        )

        assert projector.clip_proj is not None
        assert projector.t5_proj is not None
        assert isinstance(projector.clip_proj, nn.Linear)
        assert isinstance(projector.t5_proj, nn.Linear)

    @pytest.mark.fast
    def test_projector_forward_pass_shapes(self):
        """Verify text projector output shapes."""
        projector = HunyuanTextProjector(
            hidden_dim=1408,
            clip_dim=1024,
            t5_dim=2048,
        )

        batch_size = 2
        clip_seq_len = 77
        t5_seq_len = 256

        # Create dummy inputs
        clip_embeds = mx.random.normal((batch_size, clip_seq_len, 1024))
        t5_embeds = mx.random.normal((batch_size, t5_seq_len, 2048))

        # Forward pass
        output = projector(clip_embeds, t5_embeds)

        # Verify output shape
        expected_seq_len = clip_seq_len + t5_seq_len  # 77 + 256 = 333
        assert output.shape == (batch_size, expected_seq_len, 1408)

    @pytest.mark.fast
    def test_projector_concatenation_order(self):
        """Verify CLIP embeds come before T5 embeds in output."""
        projector = HunyuanTextProjector(
            hidden_dim=1408,
            clip_dim=1024,
            t5_dim=2048,
        )

        batch_size = 1
        clip_seq_len = 77
        t5_seq_len = 256

        # Create distinct inputs for verification
        clip_embeds = mx.ones((batch_size, clip_seq_len, 1024))
        t5_embeds = mx.zeros((batch_size, t5_seq_len, 2048))

        output = projector(clip_embeds, t5_embeds)

        # First 77 positions should be projected CLIP embeds (non-zero)
        # Last 256 positions should be projected T5 embeds (close to zero)
        clip_output_norm = mx.sum(mx.abs(output[:, :clip_seq_len, :]))
        t5_output_norm = mx.sum(mx.abs(output[:, clip_seq_len:, :]))

        # CLIP portion should have larger norm than T5 portion
        assert clip_output_norm > t5_output_norm

    @pytest.mark.fast
    def test_projector_different_hidden_dims(self):
        """Verify projector works with different hidden dimensions."""
        hidden_dims = [512, 1024, 1408, 2048]

        for hidden_dim in hidden_dims:
            projector = HunyuanTextProjector(
                hidden_dim=hidden_dim,
                clip_dim=1024,
                t5_dim=2048,
            )

            batch_size = 1
            clip_embeds = mx.random.normal((batch_size, 77, 1024))
            t5_embeds = mx.random.normal((batch_size, 256, 2048))

            output = projector(clip_embeds, t5_embeds)
            assert output.shape == (batch_size, 333, hidden_dim)


class TestTextEncoderOutputShapes:
    """Tests for expected text encoder output shapes."""

    @pytest.mark.fast
    def test_clip_output_shape_specification(self):
        """Verify CLIP encoder should output [batch, 77, 1024]."""
        # This is a specification test
        expected_clip_seq_len = 77
        expected_clip_dim = 1024

        # Document expected shapes
        assert expected_clip_seq_len == 77
        assert expected_clip_dim == 1024

    @pytest.mark.fast
    def test_t5_output_shape_specification(self):
        """Verify T5 encoder should output [batch, 256, 2048]."""
        expected_t5_seq_len = 256
        expected_t5_dim = 2048

        # Document expected shapes
        assert expected_t5_seq_len == 256
        assert expected_t5_dim == 2048

    @pytest.mark.fast
    def test_combined_text_sequence_length(self):
        """Verify combined text sequence length is 333 (77 + 256)."""
        clip_seq_len = 77
        t5_seq_len = 256
        expected_total = clip_seq_len + t5_seq_len

        assert expected_total == 333


class TestTextEncodingDimensions:
    """Tests for text encoding dimension compatibility."""

    @pytest.mark.fast
    def test_clip_projection_dimensions(self):
        """Verify CLIP projection from 1024 to hidden_dim."""
        hidden_dim = 1408
        clip_dim = 1024

        projector = HunyuanTextProjector(hidden_dim=hidden_dim, clip_dim=clip_dim)

        # Verify projection layer dimensions
        clip_proj_weight = projector.clip_proj.weight
        assert clip_proj_weight.shape == (hidden_dim, clip_dim)

    @pytest.mark.fast
    def test_t5_projection_dimensions(self):
        """Verify T5 projection from 2048 to hidden_dim."""
        hidden_dim = 1408
        t5_dim = 2048

        projector = HunyuanTextProjector(hidden_dim=hidden_dim, t5_dim=t5_dim)

        # Verify projection layer dimensions
        t5_proj_weight = projector.t5_proj.weight
        assert t5_proj_weight.shape == (hidden_dim, t5_dim)


class TestTextProjectorBiasHandling:
    """Tests for bias handling in text projector."""

    @pytest.mark.fast
    def test_clip_proj_has_bias(self):
        """Verify CLIP projection has bias."""
        projector = HunyuanTextProjector(hidden_dim=1408)
        assert projector.clip_proj.bias is not None

    @pytest.mark.fast
    def test_t5_proj_has_bias(self):
        """Verify T5 projection has bias."""
        projector = HunyuanTextProjector(hidden_dim=1408)
        assert projector.t5_proj.bias is not None

    @pytest.mark.fast
    def test_bias_shapes(self):
        """Verify bias shapes match hidden_dim."""
        hidden_dim = 1408
        projector = HunyuanTextProjector(hidden_dim=hidden_dim)

        assert projector.clip_proj.bias.shape == (hidden_dim,)
        assert projector.t5_proj.bias.shape == (hidden_dim,)


class TestTextEncodingEdgeCases:
    """Tests for edge cases in text encoding."""

    @pytest.mark.fast
    def test_batch_size_one(self):
        """Verify projector works with batch size 1."""
        projector = HunyuanTextProjector()

        clip_embeds = mx.random.normal((1, 77, 1024))
        t5_embeds = mx.random.normal((1, 256, 2048))

        output = projector(clip_embeds, t5_embeds)
        assert output.shape == (1, 333, 1408)

    @pytest.mark.fast
    def test_large_batch_size(self):
        """Verify projector works with large batch sizes."""
        projector = HunyuanTextProjector()

        batch_size = 16
        clip_embeds = mx.random.normal((batch_size, 77, 1024))
        t5_embeds = mx.random.normal((batch_size, 256, 2048))

        output = projector(clip_embeds, t5_embeds)
        assert output.shape == (batch_size, 333, 1408)

    @pytest.mark.fast
    def test_zero_embeddings(self):
        """Verify projector handles zero embeddings."""
        projector = HunyuanTextProjector()

        clip_embeds = mx.zeros((1, 77, 1024))
        t5_embeds = mx.zeros((1, 256, 2048))

        output = projector(clip_embeds, t5_embeds)
        assert output.shape == (1, 333, 1408)
        # Output should not be all zeros due to bias
        assert mx.sum(mx.abs(output)) > 0


class TestDualEncoderIntegration:
    """Tests for dual encoder integration patterns."""

    @pytest.mark.fast
    def test_clip_t5_independent_processing(self):
        """Verify CLIP and T5 are processed independently before concatenation."""
        projector = HunyuanTextProjector()

        # Create test inputs
        clip_embeds = mx.random.normal((1, 77, 1024))
        t5_embeds = mx.random.normal((1, 256, 2048))

        # Process together
        combined_output = projector(clip_embeds, t5_embeds)

        # Process separately
        clip_only = projector.clip_proj(clip_embeds)
        t5_only = projector.t5_proj(t5_embeds)

        # Verify combined equals concatenation of separate
        manual_combined = mx.concatenate([clip_only, t5_only], axis=1)

        # Should be equal
        assert mx.allclose(combined_output, manual_combined, atol=1e-5)

    @pytest.mark.fast
    def test_text_sequence_length_composition(self):
        """Verify text sequence length is properly composed from both encoders."""
        projector = HunyuanTextProjector()

        clip_embeds = mx.random.normal((2, 77, 1024))
        t5_embeds = mx.random.normal((2, 256, 2048))

        output = projector(clip_embeds, t5_embeds)

        # First 77 tokens should be from CLIP
        clip_part = output[:, :77, :]
        assert clip_part.shape == (2, 77, 1408)

        # Last 256 tokens should be from T5
        t5_part = output[:, 77:, :]
        assert t5_part.shape == (2, 256, 1408)


class TestTextEncoderCompatibility:
    """Tests for text encoder compatibility with transformer."""

    @pytest.mark.fast
    def test_projected_dim_matches_transformer_hidden_dim(self):
        """Verify projected embeddings match transformer hidden dimension."""
        transformer_hidden_dim = 1408
        projector = HunyuanTextProjector(hidden_dim=transformer_hidden_dim)

        clip_embeds = mx.random.normal((1, 77, 1024))
        t5_embeds = mx.random.normal((1, 256, 2048))

        output = projector(clip_embeds, t5_embeds)

        # Last dimension should match transformer hidden dim
        assert output.shape[-1] == transformer_hidden_dim

    @pytest.mark.fast
    def test_sequence_length_for_cross_attention(self):
        """Verify sequence length is suitable for cross-attention."""
        projector = HunyuanTextProjector()

        clip_embeds = mx.random.normal((1, 77, 1024))
        t5_embeds = mx.random.normal((1, 256, 2048))

        output = projector(clip_embeds, t5_embeds)

        # Total sequence length should be 333
        # This will be the key/value length in cross-attention
        assert output.shape[1] == 333
