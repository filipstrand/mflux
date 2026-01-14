"""
Tests for Chroma Adaptive Layer Normalization modules.

These are fast unit tests that don't require model loading.
"""

import mlx.core as mx
import pytest

from mflux.models.chroma.model.chroma_transformer.chroma_ada_layer_norm import (
    ChromaAdaLayerNormContinuousPruned,
    ChromaAdaLayerNormZeroPruned,
    ChromaAdaLayerNormZeroSinglePruned,
)


class TestChromaAdaLayerNormZeroPruned:
    """Tests for joint block adaptive layer norm (6 modulations)."""

    @pytest.mark.fast
    def test_output_shape_preserved(self):
        """Verify hidden states shape is preserved."""
        dim = 3072
        norm = ChromaAdaLayerNormZeroPruned(dim=dim)

        batch_size = 2
        seq_len = 100
        hidden_states = mx.random.normal((batch_size, seq_len, dim))
        modulations = mx.random.normal((batch_size, 6, dim))

        output, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm(hidden_states, modulations)

        assert output.shape == hidden_states.shape

    @pytest.mark.fast
    def test_returns_five_values(self):
        """Verify returns (normalized_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp)."""
        dim = 3072
        norm = ChromaAdaLayerNormZeroPruned(dim=dim)

        hidden_states = mx.random.normal((1, 50, dim))
        modulations = mx.random.normal((1, 6, dim))

        result = norm(hidden_states, modulations)

        assert len(result) == 5

    @pytest.mark.fast
    def test_modulation_shapes(self):
        """Verify modulation outputs have correct shapes."""
        dim = 3072
        batch_size = 2
        norm = ChromaAdaLayerNormZeroPruned(dim=dim)

        hidden_states = mx.random.normal((batch_size, 50, dim))
        modulations = mx.random.normal((batch_size, 6, dim))

        _, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm(hidden_states, modulations)

        # Each modulation should be [batch, dim]
        assert gate_msa.shape == (batch_size, dim)
        assert shift_mlp.shape == (batch_size, dim)
        assert scale_mlp.shape == (batch_size, dim)
        assert gate_mlp.shape == (batch_size, dim)

    @pytest.mark.fast
    def test_has_no_linear_layer(self):
        """Verify ChromaAdaLayerNormZeroPruned has NO linear projection (unlike FLUX)."""
        norm = ChromaAdaLayerNormZeroPruned(dim=3072)

        # Should only have norm, not a linear layer
        assert hasattr(norm, "norm")
        assert not hasattr(norm, "linear")

    @pytest.mark.fast
    def test_custom_dim(self):
        """Verify works with custom dimension."""
        dim = 1024
        norm = ChromaAdaLayerNormZeroPruned(dim=dim)

        hidden_states = mx.random.normal((1, 20, dim))
        modulations = mx.random.normal((1, 6, dim))

        output, gate_msa, shift_mlp, scale_mlp, gate_mlp = norm(hidden_states, modulations)

        assert output.shape == (1, 20, dim)
        assert gate_msa.shape == (1, dim)


class TestChromaAdaLayerNormZeroSinglePruned:
    """Tests for single block adaptive layer norm (3 modulations)."""

    @pytest.mark.fast
    def test_output_shape_preserved(self):
        """Verify hidden states shape is preserved."""
        dim = 3072
        norm = ChromaAdaLayerNormZeroSinglePruned(dim=dim)

        batch_size = 2
        seq_len = 100
        hidden_states = mx.random.normal((batch_size, seq_len, dim))
        modulations = mx.random.normal((batch_size, 3, dim))

        output, gate_msa = norm(hidden_states, modulations)

        assert output.shape == hidden_states.shape

    @pytest.mark.fast
    def test_returns_two_values(self):
        """Verify returns (normalized_hidden_states, gate_msa)."""
        dim = 3072
        norm = ChromaAdaLayerNormZeroSinglePruned(dim=dim)

        hidden_states = mx.random.normal((1, 50, dim))
        modulations = mx.random.normal((1, 3, dim))

        result = norm(hidden_states, modulations)

        assert len(result) == 2

    @pytest.mark.fast
    def test_gate_shape(self):
        """Verify gate output has correct shape."""
        dim = 3072
        batch_size = 2
        norm = ChromaAdaLayerNormZeroSinglePruned(dim=dim)

        hidden_states = mx.random.normal((batch_size, 50, dim))
        modulations = mx.random.normal((batch_size, 3, dim))

        _, gate_msa = norm(hidden_states, modulations)

        assert gate_msa.shape == (batch_size, dim)

    @pytest.mark.fast
    def test_has_no_linear_layer(self):
        """Verify ChromaAdaLayerNormZeroSinglePruned has NO linear projection (unlike FLUX)."""
        norm = ChromaAdaLayerNormZeroSinglePruned(dim=3072)

        # Should only have norm, not a linear layer
        assert hasattr(norm, "norm")
        assert not hasattr(norm, "linear")

    @pytest.mark.fast
    def test_custom_dim(self):
        """Verify works with custom dimension."""
        dim = 1024
        norm = ChromaAdaLayerNormZeroSinglePruned(dim=dim)

        hidden_states = mx.random.normal((1, 20, dim))
        modulations = mx.random.normal((1, 3, dim))

        output, gate_msa = norm(hidden_states, modulations)

        assert output.shape == (1, 20, dim)
        assert gate_msa.shape == (1, dim)


class TestChromaAdaLayerNormContinuousPruned:
    """Tests for final norm adaptive layer norm (2 modulations)."""

    @pytest.mark.fast
    def test_output_shape_preserved(self):
        """Verify hidden states shape is preserved."""
        dim = 3072
        norm = ChromaAdaLayerNormContinuousPruned(dim=dim)

        batch_size = 2
        seq_len = 100
        hidden_states = mx.random.normal((batch_size, seq_len, dim))
        modulations = mx.random.normal((batch_size, 2, dim))

        output = norm(hidden_states, modulations)

        assert output.shape == hidden_states.shape

    @pytest.mark.fast
    def test_returns_single_tensor(self):
        """Verify returns just the normalized hidden states."""
        dim = 3072
        norm = ChromaAdaLayerNormContinuousPruned(dim=dim)

        hidden_states = mx.random.normal((1, 50, dim))
        modulations = mx.random.normal((1, 2, dim))

        result = norm(hidden_states, modulations)

        # Should return a single tensor, not a tuple
        assert isinstance(result, mx.array)

    @pytest.mark.fast
    def test_has_no_linear_layer(self):
        """Verify ChromaAdaLayerNormContinuousPruned has NO linear projection (unlike FLUX)."""
        norm = ChromaAdaLayerNormContinuousPruned(dim=3072)

        # Should only have norm, not a linear layer
        assert hasattr(norm, "norm")
        assert not hasattr(norm, "linear")

    @pytest.mark.fast
    def test_custom_dim(self):
        """Verify works with custom dimension."""
        dim = 1024
        norm = ChromaAdaLayerNormContinuousPruned(dim=dim)

        hidden_states = mx.random.normal((1, 20, dim))
        modulations = mx.random.normal((1, 2, dim))

        output = norm(hidden_states, modulations)

        assert output.shape == (1, 20, dim)

    @pytest.mark.fast
    def test_normalization_effect(self):
        """Verify normalization is actually applied."""
        dim = 3072
        norm = ChromaAdaLayerNormContinuousPruned(dim=dim)

        # Create input with non-zero mean and non-unit variance
        hidden_states = mx.random.normal((1, 50, dim)) * 10 + 5
        modulations = mx.zeros((1, 2, dim))  # Zero shift/scale

        output = norm(hidden_states, modulations)

        # With zero shift/scale, output should be roughly normalized
        # (mean ~0, std ~1 per feature)
        # Note: Using 1 + scale, so scale=0 means multiply by 1
        assert not mx.allclose(output, hidden_states)


class TestChromaVsFluxAdaLayerNorm:
    """Tests comparing Chroma's pruned norms to FLUX's (conceptual)."""

    @pytest.mark.fast
    def test_no_linear_layer_difference(self):
        """
        Document the key difference: Chroma's AdaLayerNorm has NO linear layer.

        In FLUX:
        - AdaLayerNormZero has: norm + linear(dim, 6*dim)
        - AdaLayerNormZeroSingle has: norm + linear(dim, 3*dim)
        - AdaLayerNormContinuous has: norm + linear(dim, 2*dim)

        In Chroma:
        - ChromaAdaLayerNormZeroPruned has: norm only (modulations pre-computed)
        - ChromaAdaLayerNormZeroSinglePruned has: norm only
        - ChromaAdaLayerNormContinuousPruned has: norm only

        The modulations come from DistilledGuidanceLayer instead of being
        computed per-block from text embeddings.
        """
        joint_norm = ChromaAdaLayerNormZeroPruned(dim=3072)
        single_norm = ChromaAdaLayerNormZeroSinglePruned(dim=3072)
        final_norm = ChromaAdaLayerNormContinuousPruned(dim=3072)

        # All should have norm but no linear
        for norm in [joint_norm, single_norm, final_norm]:
            assert hasattr(norm, "norm")
            assert not hasattr(norm, "linear")

    @pytest.mark.fast
    def test_modulation_input_format(self):
        """
        Document the expected modulation input format.

        Modulations come from DistilledGuidanceLayer as [batch, N, dim] where:
        - N=6 for joint blocks (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
        - N=3 for single blocks (shift_msa, scale_msa, gate_msa)
        - N=2 for final norm (shift, scale)
        """
        dim = 3072
        batch = 2

        # Joint block expects [batch, 6, dim]
        joint_mods = mx.random.normal((batch, 6, dim))
        assert joint_mods.shape == (batch, 6, dim)

        # Single block expects [batch, 3, dim]
        single_mods = mx.random.normal((batch, 3, dim))
        assert single_mods.shape == (batch, 3, dim)

        # Final norm expects [batch, 2, dim]
        final_mods = mx.random.normal((batch, 2, dim))
        assert final_mods.shape == (batch, 2, dim)
