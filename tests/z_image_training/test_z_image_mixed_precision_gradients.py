"""Tests for Z-Image mixed precision gradient computation.

Tests MixedPrecisionGradientOptimizer and GradientAccumulator for
memory-efficient training with FP16 gradients.
"""

from unittest.mock import MagicMock

import mlx.core as mx
import mlx.optimizers as optim
import pytest
from mlx import nn


class TestMixedPrecisionGradientOptimizer:
    """Tests for MixedPrecisionGradientOptimizer class."""

    def test_init_default(self):
        """Test default initialization."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        base = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(base)

        assert optimizer.gradient_dtype == mx.float16
        assert optimizer.accumulation_dtype == mx.float32
        assert optimizer.loss_scale == 1.0

    def test_init_custom_dtype(self):
        """Test initialization with custom gradient dtype."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        base = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(
            base,
            gradient_dtype=mx.bfloat16,
        )

        assert optimizer.gradient_dtype == mx.bfloat16

    def test_invalid_gradient_dtype_raises(self):
        """Test that invalid gradient dtype raises error."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        base = optim.AdamW(learning_rate=1e-4)

        with pytest.raises(TypeError, match="gradient_dtype"):
            MixedPrecisionGradientOptimizer(base, gradient_dtype=mx.int32)

    def test_invalid_loss_scale_raises(self):
        """Test that non-positive loss scale raises error."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        base = optim.AdamW(learning_rate=1e-4)

        with pytest.raises(ValueError, match="loss_scale"):
            MixedPrecisionGradientOptimizer(base, loss_scale=0.0)

        with pytest.raises(ValueError, match="loss_scale"):
            MixedPrecisionGradientOptimizer(base, loss_scale=-1.0)

    def test_cast_gradients_changes_dtype(self):
        """Test that gradients are cast to target dtype."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        base = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(
            base,
            gradient_dtype=mx.float16,
        )

        gradients = {
            "layer1": {"weight": mx.ones((10, 10), dtype=mx.float32)},
            "layer2": {"weight": mx.ones((5, 5), dtype=mx.float32)},
        }

        cast_grads = optimizer._cast_gradients(gradients)

        assert cast_grads["layer1"]["weight"].dtype == mx.float16
        assert cast_grads["layer2"]["weight"].dtype == mx.float16

    def test_has_inf_or_nan_detects_nan(self):
        """Test that NaN detection works."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        base = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(base)

        # Normal gradients
        normal_grads = {"layer": {"weight": mx.ones((10, 10))}}
        assert optimizer._has_inf_or_nan(normal_grads) is False

        # Gradients with NaN
        nan_grads = {"layer": {"weight": mx.array([[float("nan")]])}}
        assert optimizer._has_inf_or_nan(nan_grads) is True

    def test_has_inf_or_nan_detects_inf(self):
        """Test that inf detection works."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        base = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(base)

        inf_grads = {"layer": {"weight": mx.array([[float("inf")]])}}
        assert optimizer._has_inf_or_nan(inf_grads) is True

    def test_update_returns_true_on_success(self):
        """Test that update returns True on successful update."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        model = nn.Linear(10, 5)
        base = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(base)

        gradients = {"weight": mx.ones((5, 10)) * 0.1, "bias": mx.ones((5,)) * 0.1}

        result = optimizer.update(model, gradients)

        assert result is True

    def test_dynamic_loss_scaling_reduces_on_nan(self):
        """Test that dynamic scaling reduces scale on NaN."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        base = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(
            base,
            loss_scale=1000.0,
            dynamic_loss_scaling=True,
        )

        initial_scale = optimizer.get_current_scale()

        # Simulate NaN gradients
        nan_grads = {"layer": {"weight": mx.array([[float("nan")]])}}
        model = MagicMock()

        optimizer.update(model, nan_grads)

        assert optimizer.get_current_scale() < initial_scale

    def test_get_stats_returns_expected_keys(self):
        """Test that get_stats returns expected statistics."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            MixedPrecisionGradientOptimizer,
        )

        base = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(base)

        stats = optimizer.get_stats()

        assert "total_updates" in stats
        assert "casts_performed" in stats
        assert "inf_nan_count" in stats
        assert "current_scale" in stats
        assert "gradient_dtype" in stats


class TestGradientAccumulator:
    """Tests for GradientAccumulator class."""

    def test_init_default(self):
        """Test default initialization."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator()

        assert accumulator.accumulation_steps == 1
        assert accumulator.accumulation_dtype == mx.float32
        assert accumulator.current_step == 0

    def test_init_custom_steps(self):
        """Test initialization with custom steps."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator(accumulation_steps=4)

        assert accumulator.accumulation_steps == 4

    def test_invalid_steps_raises(self):
        """Test that accumulation_steps < 1 raises error."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        with pytest.raises(ValueError, match="accumulation_steps"):
            GradientAccumulator(accumulation_steps=0)

    def test_accumulate_increments_step(self):
        """Test that accumulate increments current step."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator(accumulation_steps=4)

        gradients = {"layer": {"weight": mx.ones((10, 10))}}

        assert accumulator.current_step == 0
        accumulator.accumulate(gradients)
        assert accumulator.current_step == 1
        accumulator.accumulate(gradients)
        assert accumulator.current_step == 2

    def test_should_step_at_accumulation_steps(self):
        """Test should_step returns True at correct time."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator(accumulation_steps=3)
        gradients = {"layer": {"weight": mx.ones((10, 10))}}

        assert accumulator.should_step() is False
        accumulator.accumulate(gradients)
        assert accumulator.should_step() is False
        accumulator.accumulate(gradients)
        assert accumulator.should_step() is False
        accumulator.accumulate(gradients)
        assert accumulator.should_step() is True

    def test_get_and_reset_returns_averaged_gradients(self):
        """Test that get_and_reset returns averaged gradients."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator(accumulation_steps=2)

        # First batch: all 2s
        grad1 = {"layer": {"weight": mx.ones((10, 10)) * 2.0}}
        # Second batch: all 4s
        grad2 = {"layer": {"weight": mx.ones((10, 10)) * 4.0}}

        accumulator.accumulate(grad1)
        accumulator.accumulate(grad2)

        result = accumulator.get_and_reset()

        # Average: (2 + 4) / 2 = 3
        mx.synchronize()
        assert mx.allclose(result["layer"]["weight"], mx.ones((10, 10)) * 3.0)

    def test_get_and_reset_resets_accumulator(self):
        """Test that get_and_reset resets the accumulator."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator(accumulation_steps=2)
        gradients = {"layer": {"weight": mx.ones((10, 10))}}

        accumulator.accumulate(gradients)
        accumulator.accumulate(gradients)
        accumulator.get_and_reset()

        assert accumulator.current_step == 0
        assert accumulator.should_step() is False

    def test_get_and_reset_raises_if_no_accumulation(self):
        """Test that get_and_reset raises if called before accumulation."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator()

        with pytest.raises(RuntimeError, match="get_and_reset.*before any gradients.*accumulated"):
            accumulator.get_and_reset()

    def test_reset_clears_without_returning(self):
        """Test that reset clears accumulator without returning."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator(accumulation_steps=4)
        gradients = {"layer": {"weight": mx.ones((10, 10))}}

        accumulator.accumulate(gradients)
        accumulator.accumulate(gradients)

        assert accumulator.current_step == 2

        accumulator.reset()

        assert accumulator.current_step == 0

    def test_accumulates_in_fp32(self):
        """Test that accumulation happens in FP32."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator(
            accumulation_steps=2,
            accumulation_dtype=mx.float32,
        )

        # Input FP16 gradients
        grad = {"layer": {"weight": mx.ones((10, 10), dtype=mx.float16)}}

        accumulator.accumulate(grad)
        result = accumulator.get_and_reset()

        # Should be accumulated in FP32
        assert result["layer"]["weight"].dtype == mx.float32


class TestMixedPrecisionIntegration:
    """Integration tests for mixed precision training."""

    def test_optimizer_and_accumulator_together(self):
        """Test using optimizer with accumulator."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
            MixedPrecisionGradientOptimizer,
        )

        model = nn.Linear(10, 5)
        base = optim.AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(base)
        accumulator = GradientAccumulator(accumulation_steps=2)

        # Simulate 4 micro-batches (2 optimizer steps)
        for i in range(4):
            gradients = {
                "weight": mx.random.normal((5, 10)) * 0.01,
                "bias": mx.random.normal((5,)) * 0.01,
            }
            accumulator.accumulate(gradients)

            if accumulator.should_step():
                final_grads = accumulator.get_and_reset()
                optimizer.update(model, final_grads)

        # Should have done 2 optimizer updates
        assert optimizer._total_updates == 2

    def test_bf16_momentum_with_fp16_gradients(self):
        """Test combining BF16 momentum optimizer with FP16 gradients."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            BFloat16AdamW,
            MixedPrecisionGradientOptimizer,
        )

        model = nn.Linear(10, 5)

        # BF16 momentum optimizer wrapped with FP16 gradient optimizer
        base = BFloat16AdamW(learning_rate=1e-4)
        optimizer = MixedPrecisionGradientOptimizer(
            base,
            gradient_dtype=mx.float16,
        )

        gradients = {
            "weight": mx.random.normal((5, 10)) * 0.01,
            "bias": mx.random.normal((5,)) * 0.01,
        }

        # Should work without errors
        result = optimizer.update(model, gradients)
        assert result is True


class TestMemorySavingsEstimation:
    """Tests for memory savings estimation."""

    def test_fp16_gradients_halve_memory(self):
        """Test that FP16 gradients use half the memory of FP32."""
        # FP32: 4 bytes per element
        # FP16: 2 bytes per element
        # Savings: 50%

        fp32_gradient = mx.ones((1000, 1000), dtype=mx.float32)
        fp16_gradient = fp32_gradient.astype(mx.float16)

        fp32_bytes = fp32_gradient.nbytes
        fp16_bytes = fp16_gradient.nbytes

        ratio = fp16_bytes / fp32_bytes
        assert abs(ratio - 0.5) < 0.01  # Should be ~50%

    def test_accumulator_precision_constant(self):
        """Test that accumulator maintains precision regardless of input."""
        from mflux.models.z_image.variants.training.optimization.precision_optimizer import (
            GradientAccumulator,
        )

        accumulator = GradientAccumulator(
            accumulation_steps=3,
            accumulation_dtype=mx.float32,
        )

        # Mix of dtypes
        grad1 = {"layer": {"weight": mx.ones((10, 10), dtype=mx.float16)}}
        grad2 = {"layer": {"weight": mx.ones((10, 10), dtype=mx.bfloat16)}}
        grad3 = {"layer": {"weight": mx.ones((10, 10), dtype=mx.float32)}}

        accumulator.accumulate(grad1)
        accumulator.accumulate(grad2)
        accumulator.accumulate(grad3)

        result = accumulator.get_and_reset()

        # Should always be FP32 regardless of input dtypes
        assert result["layer"]["weight"].dtype == mx.float32
