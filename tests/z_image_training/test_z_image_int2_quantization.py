"""Tests for INT2 quantization support in Z-Image.

INT2 provides extreme compression (8x vs FP16) at the cost of quality.
Useful for memory-constrained inference or quality testing.
"""

from mflux.cli.defaults.defaults import QUANTIZE_CHOICES
from mflux.models.z_image.variants.training.optimization.memory_optimizer import MemoryOptimizer


class TestINT2QuantizationChoice:
    """Tests for INT2 in QUANTIZE_CHOICES."""

    def test_int2_in_quantize_choices(self):
        """INT2 should be a valid quantization choice."""
        assert 2 in QUANTIZE_CHOICES

    def test_quantize_choices_sorted(self):
        """QUANTIZE_CHOICES should include all common bit widths."""
        # Verify all expected bit widths are present
        expected = {2, 3, 4, 5, 6, 8}
        assert set(QUANTIZE_CHOICES) == expected

    def test_int2_is_smallest(self):
        """INT2 should be the smallest quantization level."""
        assert min(QUANTIZE_CHOICES) == 2


class TestINT2MemoryEstimates:
    """Tests for INT2 memory estimation."""

    def test_model_size_2bit_defined(self):
        """MODEL_SIZE_2BIT constant should be defined."""
        assert hasattr(MemoryOptimizer, "MODEL_SIZE_2BIT")
        assert MemoryOptimizer.MODEL_SIZE_2BIT > 0

    def test_model_size_2bit_is_smallest(self):
        """INT2 should have smallest model size."""
        assert MemoryOptimizer.MODEL_SIZE_2BIT < MemoryOptimizer.MODEL_SIZE_4BIT
        assert MemoryOptimizer.MODEL_SIZE_2BIT < MemoryOptimizer.MODEL_SIZE_8BIT
        assert MemoryOptimizer.MODEL_SIZE_2BIT < MemoryOptimizer.MODEL_SIZE_BF16

    def test_model_size_2bit_approximately_half_of_4bit(self):
        """INT2 should be approximately half the size of INT4."""
        ratio = MemoryOptimizer.MODEL_SIZE_2BIT / MemoryOptimizer.MODEL_SIZE_4BIT
        assert 0.4 <= ratio <= 0.6  # Allow some tolerance

    def test_model_size_ratios(self):
        """Verify model size ratios are reasonable."""
        # INT2: ~0.25 bytes per param
        # INT4: ~0.5 bytes per param
        # INT8: ~1 byte per param
        # BF16: ~2 bytes per param
        bf16 = MemoryOptimizer.MODEL_SIZE_BF16
        int8 = MemoryOptimizer.MODEL_SIZE_8BIT
        int4 = MemoryOptimizer.MODEL_SIZE_4BIT
        int2 = MemoryOptimizer.MODEL_SIZE_2BIT

        # Ratios should be approximately 8:4:2:1
        assert 0.45 <= int8 / bf16 <= 0.55  # INT8 is ~50% of BF16
        assert 0.45 <= int4 / int8 <= 0.55  # INT4 is ~50% of INT8
        assert 0.45 <= int2 / int4 <= 0.55  # INT2 is ~50% of INT4


class TestINT2MemoryEstimation:
    """Tests for memory estimation with INT2 quantization."""

    def test_estimate_memory_for_lora_with_int2(self):
        """Test memory estimation handles INT2 quantization."""
        estimate = MemoryOptimizer.estimate_memory_for_lora_training(
            batch_size=1,
            quantize=2,
            width=1024,
            height=1024,
        )

        assert "model" in estimate
        assert estimate["model"] == MemoryOptimizer.MODEL_SIZE_2BIT
        assert estimate["total"] > 0

    def test_int2_reduces_total_memory(self):
        """INT2 should result in lower total memory than INT4."""
        estimate_int2 = MemoryOptimizer.estimate_memory_for_lora_training(batch_size=1, quantize=2)
        estimate_int4 = MemoryOptimizer.estimate_memory_for_lora_training(batch_size=1, quantize=4)

        assert estimate_int2["total"] < estimate_int4["total"]

    def test_int2_allows_larger_batch_size(self):
        """INT2 should allow larger batch sizes in memory."""
        batch_int2, _ = MemoryOptimizer.calculate_optimal_batch_size(
            mode="lora",
            available_memory_gb=32.0,
            quantize=2,
        )
        batch_int4, _ = MemoryOptimizer.calculate_optimal_batch_size(
            mode="lora",
            available_memory_gb=32.0,
            quantize=4,
        )

        # INT2 should allow same or larger batch
        assert batch_int2 >= batch_int4


class TestINT2QualityWarnings:
    """Tests for INT2 quality expectations.

    INT2 is an extreme quantization level that will degrade quality
    significantly. These tests document expected behavior.
    """

    def test_int2_compression_ratio(self):
        """Document INT2 compression ratio vs FP16."""
        bf16_size = MemoryOptimizer.MODEL_SIZE_BF16
        int2_size = MemoryOptimizer.MODEL_SIZE_2BIT

        compression_ratio = bf16_size / int2_size
        # INT2 should provide ~8x compression vs FP16
        assert compression_ratio >= 7.0

    def test_int2_size_estimate(self):
        """Document expected INT2 model size."""
        # 6B params at 2 bits = 6e9 * 0.25 bytes = 1.5GB
        expected_size_gb = 1.5
        assert abs(MemoryOptimizer.MODEL_SIZE_2BIT - expected_size_gb) < 0.1
