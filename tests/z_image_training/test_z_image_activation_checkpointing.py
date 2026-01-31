"""Tests for Z-Image activation checkpointing functionality.

Activation checkpointing trades compute for memory by recomputing
activations during the backward pass instead of storing them.
"""

from unittest.mock import MagicMock

from mlx import nn

from mflux.models.z_image.variants.training.optimization.activation_checkpointing import (
    ActivationCheckpointer,
    apply_gradient_checkpointing,
)
from mflux.models.z_image.variants.training.optimization.memory_optimizer import (
    MemoryOptimizer,
)


class MockLayer(nn.Module):
    """Mock transformer layer for testing."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def __call__(self, x):
        return self.linear(x)


class MockTransformer(nn.Module):
    """Mock transformer for testing checkpointing."""

    def __init__(self, num_layers: int = 4, num_refiner_layers: int = 2):
        super().__init__()
        self.layers = [MockLayer() for _ in range(num_layers)]
        self.noise_refiner = [MockLayer() for _ in range(num_refiner_layers)]
        self.context_refiner = [MockLayer() for _ in range(num_refiner_layers)]


class TestActivationCheckpointer:
    """Tests for ActivationCheckpointer class."""

    def test_constants_defined(self):
        """Test that constants are defined."""
        assert hasattr(ActivationCheckpointer, "CHECKPOINTING_MEMORY_FACTOR")
        assert hasattr(ActivationCheckpointer, "COMPUTE_OVERHEAD_FACTOR")

        assert 0 < ActivationCheckpointer.CHECKPOINTING_MEMORY_FACTOR < 1
        assert ActivationCheckpointer.COMPUTE_OVERHEAD_FACTOR > 1

    def test_apply_to_all_layers(self):
        """Test applying checkpointing to all layers."""
        transformer = MockTransformer(num_layers=4)

        count = ActivationCheckpointer.apply_to_model(
            transformer,
            checkpoint_every_n_layers=1,  # Every layer
        )

        # Should wrap all layers: 4 + 2 + 2 = 8
        assert count == 8

        # Verify layers are marked as checkpointed
        for layer in transformer.layers:
            assert getattr(layer, "_checkpointed", False) is True

    def test_apply_every_2and_layer(self):
        """Test applying checkpointing every 2nd layer."""
        transformer = MockTransformer(num_layers=4, num_refiner_layers=2)

        count = ActivationCheckpointer.apply_to_model(
            transformer,
            checkpoint_every_n_layers=2,
        )

        # With every 2nd layer: layers[0,2] + noise_refiner[0] + context_refiner[0] = 4
        assert count == 4

        # Check which layers are checkpointed
        assert transformer.layers[0]._checkpointed is True
        assert not hasattr(transformer.layers[1], "_checkpointed")
        assert transformer.layers[2]._checkpointed is True
        assert not hasattr(transformer.layers[3], "_checkpointed")

    def test_apply_specific_layer_names(self):
        """Test applying checkpointing to specific layer names only."""
        transformer = MockTransformer(num_layers=4, num_refiner_layers=2)

        count = ActivationCheckpointer.apply_to_model(
            transformer,
            checkpoint_every_n_layers=1,
            layer_names=["layers"],  # Only main layers
        )

        # Should only wrap main layers
        assert count == 4

        # noise_refiner should not be checkpointed
        for layer in transformer.noise_refiner:
            assert not hasattr(layer, "_checkpointed")

    def test_remove_from_model(self):
        """Test removing checkpointing from model."""
        transformer = MockTransformer(num_layers=4)

        # First apply
        ActivationCheckpointer.apply_to_model(transformer, checkpoint_every_n_layers=1)

        # Verify checkpointed
        assert transformer.layers[0]._checkpointed is True

        # Remove
        count = ActivationCheckpointer.remove_from_model(transformer)

        # Verify removed
        assert count > 0
        assert not hasattr(transformer.layers[0], "_checkpointed")

    def test_wrap_preserves_functionality(self):
        """Test that wrapped layers still work correctly."""
        import mlx.core as mx

        transformer = MockTransformer(num_layers=2)

        # Get output before checkpointing
        x = mx.ones((1, 128))
        original_output = transformer.layers[0](x)

        # Apply checkpointing
        ActivationCheckpointer.apply_to_model(transformer, checkpoint_every_n_layers=1)

        # Get output after checkpointing
        checkpointed_output = transformer.layers[0](x)

        # Outputs should be identical (checkpointing only affects backward)
        mx.eval(original_output, checkpointed_output)
        assert mx.allclose(original_output, checkpointed_output)


class TestMemorySavingsEstimation:
    """Tests for memory savings estimation."""

    def test_estimate_memory_savings_basic(self):
        """Test basic memory savings estimation."""
        estimate = ActivationCheckpointer.estimate_memory_savings(
            num_layers=30,
            checkpoint_every_n=1,  # All layers
            batch_size=1,
            base_activation_gb=15.0,
        )

        assert "original_activation_gb" in estimate
        assert "checkpointed_activation_gb" in estimate
        assert "memory_saved_gb" in estimate
        assert "reduction_factor" in estimate

        # With all layers checkpointed, should save significant memory
        assert estimate["memory_saved_gb"] > 10.0
        assert estimate["checkpointed_activation_gb"] < estimate["original_activation_gb"]

    def test_estimate_memory_savings_every_2and(self):
        """Test memory savings with every 2nd layer."""
        estimate = ActivationCheckpointer.estimate_memory_savings(
            num_layers=30,
            checkpoint_every_n=2,
            batch_size=1,
        )

        # Should save less than checkpointing all layers
        estimate_all = ActivationCheckpointer.estimate_memory_savings(
            num_layers=30,
            checkpoint_every_n=1,
            batch_size=1,
        )

        assert estimate["memory_saved_gb"] < estimate_all["memory_saved_gb"]

    def test_estimate_scales_with_batch_size(self):
        """Test that memory estimate scales with batch size."""
        estimate_bs1 = ActivationCheckpointer.estimate_memory_savings(
            num_layers=30,
            checkpoint_every_n=1,
            batch_size=1,
        )

        estimate_bs4 = ActivationCheckpointer.estimate_memory_savings(
            num_layers=30,
            checkpoint_every_n=1,
            batch_size=4,
        )

        # Memory should scale linearly with batch size
        ratio = estimate_bs4["original_activation_gb"] / estimate_bs1["original_activation_gb"]
        assert 3.9 < ratio < 4.1


class TestMemoryOptimizerIntegration:
    """Tests for integration with MemoryOptimizer."""

    def test_checkpointing_factor_defined(self):
        """Test that checkpointing factor is defined in MemoryOptimizer."""
        assert hasattr(MemoryOptimizer, "ACTIVATION_CHECKPOINTING_FACTOR")
        factor = MemoryOptimizer.ACTIVATION_CHECKPOINTING_FACTOR

        # Should be a reduction factor between 0 and 1
        assert 0 < factor < 1


class TestApplyGradientCheckpointing:
    """Tests for convenience function."""

    def test_apply_enabled(self):
        """Test applying checkpointing when enabled."""
        model = MagicMock()
        model.transformer = MockTransformer(num_layers=4)

        result = apply_gradient_checkpointing(model, enabled=True)

        assert result is True
        assert model.transformer.layers[0]._checkpointed is True

    def test_apply_disabled(self):
        """Test no-op when disabled."""
        model = MagicMock()
        model.transformer = MockTransformer(num_layers=4)

        result = apply_gradient_checkpointing(model, enabled=False)

        assert result is False

    def test_apply_no_transformer(self):
        """Test handling model without transformer."""
        model = MagicMock(spec=[])  # No transformer attribute

        result = apply_gradient_checkpointing(model, enabled=True)

        assert result is False


class TestFullFinetuneSpecIntegration:
    """Tests for FullFinetuneSpec integration."""

    def test_gradient_checkpointing_field(self):
        """Test that FullFinetuneSpec has gradient_checkpointing field."""
        from mflux.models.z_image.variants.training.state.training_spec import (
            FullFinetuneSpec,
        )

        spec = FullFinetuneSpec(gradient_checkpointing=True)
        assert spec.gradient_checkpointing is True

        spec = FullFinetuneSpec(gradient_checkpointing=False)
        assert spec.gradient_checkpointing is False

    def test_default_is_disabled(self):
        """Test that gradient checkpointing is disabled by default."""
        from mflux.models.z_image.variants.training.state.training_spec import (
            FullFinetuneSpec,
        )

        spec = FullFinetuneSpec()
        assert spec.gradient_checkpointing is False
