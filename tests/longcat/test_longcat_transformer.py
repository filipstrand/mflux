"""
Unit tests for LongCat transformer forward pass.

Tests verify:
1. Transformer architecture (10 joint + 20 single blocks)
2. Forward pass shape validation
3. Dimension correctness (hidden=3072, context=3584, x_embedder=64)
4. Component initialization
"""

import mlx.core as mx
import pytest

from mflux.models.common.config import Config, ModelConfig
from mflux.models.longcat.model.longcat_transformer.longcat_transformer import LongCatTransformer


class TestLongCatTransformerArchitecture:
    """Tests for LongCat transformer architecture constants."""

    @pytest.mark.fast
    def test_transformer_block_counts(self):
        """Verify LongCat has 10 joint + 20 single blocks."""
        transformer = LongCatTransformer()

        # Check block counts
        assert len(transformer.transformer_blocks) == 10
        assert len(transformer.single_transformer_blocks) == 20

    @pytest.mark.fast
    def test_transformer_dimensions(self):
        """Verify LongCat dimension constants."""
        assert LongCatTransformer.NUM_JOINT_BLOCKS == 10
        assert LongCatTransformer.NUM_SINGLE_BLOCKS == 20
        assert LongCatTransformer.HIDDEN_DIM == 3072
        assert LongCatTransformer.CONTEXT_DIM == 3584  # Qwen2.5-VL
        assert LongCatTransformer.X_EMBEDDER_DIM == 64  # 16 VAE channels * 4

    @pytest.mark.fast
    def test_x_embedder_dim_calculation(self):
        """Verify X_EMBEDDER_DIM is correctly calculated from VAE channels."""
        # 16 latent channels * 4 for patch packing = 64
        expected_dim = 16 * 4
        assert LongCatTransformer.X_EMBEDDER_DIM == expected_dim


class TestLongCatTransformerInitialization:
    """Tests for LongCat transformer component initialization."""

    @pytest.mark.fast
    def test_transformer_components_initialized(self):
        """Verify all transformer components are initialized."""
        transformer = LongCatTransformer()

        # Check core components exist
        assert hasattr(transformer, "pos_embed")
        assert hasattr(transformer, "x_embedder")
        assert hasattr(transformer, "time_text_embed")
        assert hasattr(transformer, "context_embedder")
        assert hasattr(transformer, "norm_out")
        assert hasattr(transformer, "proj_out")

    @pytest.mark.fast
    def test_x_embedder_shape(self):
        """Verify x_embedder has correct input/output dimensions."""
        transformer = LongCatTransformer()

        # x_embedder should map 64 -> 3072
        assert transformer.x_embedder.weight.shape == (3072, 64)

    @pytest.mark.fast
    def test_context_embedder_shape(self):
        """Verify context_embedder has correct input/output dimensions."""
        transformer = LongCatTransformer()

        # context_embedder should map 3584 (Qwen2.5-VL) -> 3072
        assert transformer.context_embedder.weight.shape == (3072, 3584)

    @pytest.mark.fast
    def test_proj_out_shape(self):
        """Verify proj_out has correct input/output dimensions."""
        transformer = LongCatTransformer()

        # proj_out should map 3072 -> 64
        assert transformer.proj_out.weight.shape == (64, 3072)


class TestLongCatTransformerForwardPass:
    """Tests for LongCat transformer forward pass."""

    @pytest.fixture
    def transformer(self):
        """Create a LongCat transformer for testing."""
        return LongCatTransformer()

    @pytest.fixture
    def config(self):
        """Create a test config."""
        model_config = ModelConfig.longcat()
        return Config(
            width=512,
            height=512,
            guidance=4.0,
            num_inference_steps=4,
            model_config=model_config,
        )

    @pytest.mark.fast
    def test_forward_pass_output_shape(self, transformer, config):
        """Test that forward pass produces correct output shape."""
        batch_size = 1
        height, width = 512, 512
        latent_h, latent_w = height // 16, width // 16
        num_latent_patches = latent_h * latent_w

        # Create mock inputs
        hidden_states = mx.random.normal((batch_size, num_latent_patches, 64))
        prompt_embeds = mx.random.normal((batch_size, 128, 3584))  # Qwen2.5-VL embeddings
        pooled_prompt_embeds = mx.random.normal((batch_size, 3584))

        # Forward pass
        output = transformer(
            t=0,
            config=config,
            hidden_states=hidden_states,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        # Verify output shape matches input latent shape
        assert output.shape == (batch_size, num_latent_patches, 64)

    @pytest.mark.fast
    def test_forward_pass_with_different_resolutions(self, transformer):
        """Test forward pass works with different resolutions."""
        test_cases = [
            (256, 256),  # Small
            (512, 512),  # Medium
            (1024, 1024),  # Large
            (512, 1024),  # Rectangular
        ]

        batch_size = 1
        model_config = ModelConfig.longcat()

        for height, width in test_cases:
            latent_h, latent_w = height // 16, width // 16
            num_patches = latent_h * latent_w

            # Create new config for this resolution
            test_config = Config(
                width=width,
                height=height,
                guidance=4.0,
                num_inference_steps=4,
                model_config=model_config,
            )

            # Create inputs for this resolution
            hidden_states = mx.random.normal((batch_size, num_patches, 64))
            prompt_embeds = mx.random.normal((batch_size, 128, 3584))
            pooled_prompt_embeds = mx.random.normal((batch_size, 3584))

            # Forward pass should work
            output = transformer(
                t=0,
                config=test_config,
                hidden_states=hidden_states,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )

            # Verify output shape
            assert output.shape == (batch_size, num_patches, 64)

    @pytest.mark.fast
    def test_forward_pass_with_different_sequence_lengths(self, transformer, config):
        """Test forward pass works with different text sequence lengths."""
        batch_size = 1
        height, width = 512, 512
        latent_h, latent_w = height // 16, width // 16
        num_patches = latent_h * latent_w

        hidden_states = mx.random.normal((batch_size, num_patches, 64))

        # Test different sequence lengths
        for seq_len in [32, 64, 128, 256, 512]:
            prompt_embeds = mx.random.normal((batch_size, seq_len, 3584))
            pooled_prompt_embeds = mx.random.normal((batch_size, 3584))

            output = transformer(
                t=0,
                config=config,
                hidden_states=hidden_states,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
            )

            assert output.shape == (batch_size, num_patches, 64)

    @pytest.mark.fast
    def test_forward_pass_deterministic(self, transformer, config):
        """Test that forward pass is deterministic with same inputs."""
        batch_size = 1
        height, width = 512, 512
        latent_h, latent_w = height // 16, width // 16
        num_patches = latent_h * latent_w

        # Create fixed inputs
        mx.random.seed(42)
        hidden_states = mx.random.normal((batch_size, num_patches, 64))
        prompt_embeds = mx.random.normal((batch_size, 128, 3584))
        pooled_prompt_embeds = mx.random.normal((batch_size, 3584))

        # Run twice
        output1 = transformer(
            t=0,
            config=config,
            hidden_states=hidden_states,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        output2 = transformer(
            t=0,
            config=config,
            hidden_states=hidden_states,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
        )

        # Should be identical
        assert mx.allclose(output1, output2)


class TestLongCatTransformerHelperMethods:
    """Tests for LongCat transformer helper methods."""

    @pytest.mark.fast
    def test_prepare_latent_image_ids_shape(self):
        """Test _prepare_latent_image_ids produces correct shape."""
        height, width = 512, 512
        latent_h, latent_w = height // 16, width // 16

        ids = LongCatTransformer._prepare_latent_image_ids(height, width)

        # Should be [1, num_patches, 3]
        assert ids.shape == (1, latent_h * latent_w, 3)

    @pytest.mark.fast
    def test_prepare_text_ids_shape(self):
        """Test _prepare_text_ids produces correct shape."""
        seq_len = 128

        ids = LongCatTransformer._prepare_text_ids(seq_len)

        # Should be [1, seq_len, 3]
        assert ids.shape == (1, seq_len, 3)

    @pytest.mark.fast
    def test_prepare_text_ids_all_zeros(self):
        """Test _prepare_text_ids returns all zeros (no position info for text)."""
        seq_len = 128

        ids = LongCatTransformer._prepare_text_ids(seq_len)

        # Should be all zeros
        assert mx.all(ids == 0)

    @pytest.mark.fast
    def test_prepare_latent_image_ids_position_encoding(self):
        """Test _prepare_latent_image_ids encodes 2D positions."""
        height, width = 32, 32  # Small for testing
        latent_h, latent_w = height // 16, width // 16  # 2x2

        ids = LongCatTransformer._prepare_latent_image_ids(height, width)

        # First dimension should be 0 (batch/temporal)
        assert mx.all(ids[:, :, 0] == 0)

        # Second dimension should encode row position (0, 0, 1, 1)
        expected_rows = mx.array([0, 0, 1, 1]).reshape(1, 4)
        assert mx.allclose(ids[:, :, 1], expected_rows)

        # Third dimension should encode column position (0, 1, 0, 1)
        expected_cols = mx.array([0, 1, 0, 1]).reshape(1, 4)
        assert mx.allclose(ids[:, :, 2], expected_cols)


class TestLongCatTransformerVsFlux:
    """Tests comparing LongCat transformer to FLUX architecture."""

    @pytest.mark.fast
    def test_longcat_has_fewer_blocks_than_flux(self):
        """Verify LongCat has fewer blocks than FLUX (10 vs 19 joint)."""
        longcat = LongCatTransformer()

        # LongCat: 10 joint blocks (vs FLUX: 19)
        assert len(longcat.transformer_blocks) == 10
        # FLUX typically has 19 joint blocks (documented for comparison)

        # LongCat: 20 single blocks (vs FLUX: 38)
        assert len(longcat.single_transformer_blocks) == 20
        # FLUX typically has 38 single blocks (documented for comparison)

    @pytest.mark.fast
    def test_longcat_uses_different_context_dim(self):
        """Verify LongCat uses Qwen2.5-VL context dim (3584 vs FLUX 4096)."""
        longcat = LongCatTransformer()

        # LongCat uses Qwen2.5-VL: 3584
        assert longcat.context_embedder.weight.shape[1] == 3584

        # FLUX uses T5: 4096
        # (We just verify LongCat is different)

    @pytest.mark.fast
    def test_longcat_uses_same_vae_as_flux(self):
        """Verify LongCat uses same VAE channels as FLUX (16)."""
        # Both use 16 channels * 4 = 64 for x_embedder input
        assert LongCatTransformer.X_EMBEDDER_DIM == 64
