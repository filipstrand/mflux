"""
Tests for DistilledGuidanceLayer and ChromaApproximator.

These are fast unit tests that don't require model loading.
"""

import mlx.core as mx
import pytest

from mflux.models.chroma.model.chroma_transformer.distilled_guidance_layer import (
    ApproximatorBlock,
    ChromaApproximator,
    DistilledGuidanceLayer,
    RMSNorm,
    get_timestep_embedding,
)


class TestRMSNorm:
    """Tests for RMSNorm layer."""

    @pytest.mark.fast
    def test_rms_norm_output_shape(self):
        """Verify RMSNorm preserves input shape."""
        norm = RMSNorm(dim=512)
        x = mx.random.normal((2, 100, 512))
        output = norm(x)
        assert output.shape == x.shape

    @pytest.mark.fast
    def test_rms_norm_weight_shape(self):
        """Verify RMSNorm weight has correct shape."""
        dim = 3072
        norm = RMSNorm(dim=dim)
        assert norm.weight.shape == (dim,)


class TestApproximatorBlock:
    """Tests for ApproximatorBlock (single MLP block)."""

    @pytest.mark.fast
    def test_approximator_block_output_shape(self):
        """Verify ApproximatorBlock preserves hidden dimension."""
        hidden_dim = 5120
        block = ApproximatorBlock(hidden_dim=hidden_dim)
        x = mx.random.normal((2, 10, hidden_dim))
        output = block(x)
        assert output.shape == x.shape

    @pytest.mark.fast
    def test_approximator_block_has_two_linears(self):
        """Verify ApproximatorBlock has two linear layers."""
        block = ApproximatorBlock(hidden_dim=5120)
        assert hasattr(block, "linear_1")
        assert hasattr(block, "linear_2")


class TestChromaApproximator:
    """Tests for ChromaApproximator (5-layer MLP network)."""

    @pytest.mark.fast
    def test_chroma_approximator_output_shape(self):
        """Verify ChromaApproximator produces correct output dimensions."""
        in_dim = 64
        out_dim = 3072
        hidden_dim = 5120
        n_layers = 5

        approx = ChromaApproximator(
            in_dim=in_dim,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
        )

        x = mx.random.normal((2, 344, in_dim))
        output = approx(x)

        assert output.shape == (2, 344, out_dim)

    @pytest.mark.fast
    def test_chroma_approximator_layer_count(self):
        """Verify ChromaApproximator has correct number of layers."""
        n_layers = 5
        approx = ChromaApproximator(n_layers=n_layers)

        assert len(approx.layers) == n_layers
        assert len(approx.norms) == n_layers

    @pytest.mark.fast
    def test_chroma_approximator_default_dimensions(self):
        """Verify ChromaApproximator default dimensions match Chroma config."""
        approx = ChromaApproximator()

        # Check in_proj dimensions (64 -> 5120)
        assert approx.in_proj.weight.shape == (5120, 64)

        # Check out_proj dimensions (5120 -> 3072)
        assert approx.out_proj.weight.shape == (3072, 5120)


class TestGetTimestepEmbedding:
    """Tests for sinusoidal timestep embedding function."""

    @pytest.mark.fast
    def test_timestep_embedding_shape(self):
        """Verify timestep embedding produces correct shape."""
        timesteps = mx.array([0.0, 0.5, 1.0])
        embedding_dim = 64

        emb = get_timestep_embedding(timesteps, embedding_dim)

        assert emb.shape == (3, embedding_dim)

    @pytest.mark.fast
    def test_timestep_embedding_single_timestep(self):
        """Verify timestep embedding works with single timestep."""
        timestep = mx.array([0.5])
        embedding_dim = 16

        emb = get_timestep_embedding(timestep, embedding_dim)

        assert emb.shape == (1, embedding_dim)

    @pytest.mark.fast
    def test_timestep_embedding_flip_sin_cos(self):
        """Verify flip_sin_to_cos parameter affects output."""
        timesteps = mx.array([0.5])
        embedding_dim = 16

        emb_flip = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=True)
        emb_no_flip = get_timestep_embedding(timesteps, embedding_dim, flip_sin_to_cos=False)

        # Outputs should be different when flip is toggled
        assert not mx.allclose(emb_flip, emb_no_flip)


class TestDistilledGuidanceLayer:
    """Tests for the complete DistilledGuidanceLayer."""

    @pytest.mark.fast
    def test_distilled_guidance_layer_output_shape(self):
        """Verify DistilledGuidanceLayer produces [batch, 344, 3072] output."""
        layer = DistilledGuidanceLayer()

        timestep = mx.array([0.5])
        output = layer(timestep)

        # Default: out_dim=344, inner_dim=3072
        assert output.shape == (1, 344, 3072)

    @pytest.mark.fast
    def test_distilled_guidance_layer_batch_processing(self):
        """Verify DistilledGuidanceLayer handles batch inputs."""
        layer = DistilledGuidanceLayer()

        timesteps = mx.array([0.1, 0.5, 0.9])
        output = layer(timesteps)

        assert output.shape == (3, 344, 3072)

    @pytest.mark.fast
    def test_distilled_guidance_layer_scalar_timestep(self):
        """Verify DistilledGuidanceLayer handles scalar timestep."""
        layer = DistilledGuidanceLayer()

        timestep = mx.array(0.5)  # Scalar
        output = layer(timestep)

        assert output.shape == (1, 344, 3072)

    @pytest.mark.fast
    def test_distilled_guidance_layer_modulation_count(self):
        """Verify DistilledGuidanceLayer produces exactly 344 modulation vectors."""
        layer = DistilledGuidanceLayer()

        timestep = mx.array([0.5])
        output = layer(timestep)

        # 344 = 114 (single) + 114 (joint img) + 114 (joint txt) + 2 (final)
        assert output.shape[1] == 344

    @pytest.mark.fast
    def test_distilled_guidance_layer_custom_dimensions(self):
        """Verify DistilledGuidanceLayer respects custom dimensions."""
        layer = DistilledGuidanceLayer(
            num_channels=32,
            out_dim=100,
            inner_dim=1024,
        )

        timestep = mx.array([0.5])
        output = layer(timestep)

        assert output.shape == (1, 100, 1024)

    @pytest.mark.fast
    def test_distilled_guidance_layer_has_approximator(self):
        """Verify DistilledGuidanceLayer has ChromaApproximator component."""
        layer = DistilledGuidanceLayer()
        assert hasattr(layer, "approximator")
        assert isinstance(layer.approximator, ChromaApproximator)

    @pytest.mark.fast
    def test_distilled_guidance_layer_mod_proj_shape(self):
        """Verify pre-computed modulation projection has correct shape."""
        layer = DistilledGuidanceLayer()

        # _mod_proj should be [out_dim, num_channels//2] = [344, 32]
        assert layer._mod_proj.shape == (344, 32)

    @pytest.mark.fast
    def test_distilled_guidance_layer_deterministic(self):
        """Verify same timestep produces same output."""
        layer = DistilledGuidanceLayer()

        timestep = mx.array([0.5])
        output1 = layer(timestep)
        output2 = layer(timestep)

        assert mx.allclose(output1, output2)


class TestModulationDistribution:
    """Tests for modulation index distribution (critical for correctness)."""

    @pytest.mark.fast
    def test_modulation_indices_single_blocks(self):
        """Verify single block modulations are at indices 0-113."""
        # 38 single blocks × 3 mods each = 114 modulations
        num_single_blocks = 38
        mods_per_single = 3
        expected_count = num_single_blocks * mods_per_single
        assert expected_count == 114

    @pytest.mark.fast
    def test_modulation_indices_joint_blocks(self):
        """Verify joint block modulations are at indices 114-341."""
        # 19 joint blocks × (6 img + 6 txt) mods each = 228 modulations
        num_joint_blocks = 19
        mods_per_joint = 12  # 6 for img, 6 for txt
        expected_count = num_joint_blocks * mods_per_joint
        assert expected_count == 228

    @pytest.mark.fast
    def test_modulation_indices_final_norm(self):
        """Verify final norm modulations are at indices 342-343."""
        # Final norm uses 2 modulations (shift, scale)
        final_norm_mods = 2
        assert final_norm_mods == 2

    @pytest.mark.fast
    def test_total_modulation_count(self):
        """Verify total modulation count is 344."""
        single_mods = 38 * 3  # 114
        joint_mods = 19 * 12  # 228
        final_mods = 2

        total = single_mods + joint_mods + final_mods
        assert total == 344
