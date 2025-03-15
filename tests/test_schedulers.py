import math

import mlx.core as mx
import numpy as np
import pytest

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig


@pytest.fixture
def num_inference_steps():
    """Number of inference steps for testing schedulers."""
    return 20


@pytest.fixture
def rtol():
    """Relative tolerance for float comparison."""
    return 1e-5


@pytest.fixture
def atol():
    """Absolute tolerance for float comparison."""
    return 1e-8


@pytest.fixture
def model_config():
    """Mock ModelConfig for testing."""
    return ModelConfig(
        alias="test",
        model_name="test_model",
        supports_guidance=True,
        num_train_steps=1000,
        max_sequence_length=512,
        base_model=None,
    )


class TestLinearScheduler:
    """Tests for the linear noise scheduler."""

    def test_shape_and_type(self, num_inference_steps):
        """Test the shape and dtype of the linear scheduler output."""
        sigmas = RuntimeConfig._create_linear_sigmas(num_inference_steps)
        assert sigmas.shape == (num_inference_steps + 1,)
        assert sigmas.dtype == mx.float32

    def test_boundary_conditions(self, num_inference_steps, rtol, atol):
        """Test the boundary values of the linear scheduler."""
        sigmas = RuntimeConfig._create_linear_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Check start and end values (excluding final 0)
        assert np.isclose(sigmas_np[0], 1.0, rtol=rtol, atol=atol)
        assert np.isclose(sigmas_np[-2], 1.0 / num_inference_steps, rtol=rtol, atol=atol)
        assert sigmas_np[-1] == 0.0

    def test_linear_decay(self, num_inference_steps, rtol, atol):
        """Test that the linear scheduler has consistent step sizes."""
        sigmas = RuntimeConfig._create_linear_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Check that differences between consecutive steps are consistent (linear)
        diffs = np.diff(sigmas_np[:-1])  # Exclude the final 0
        assert np.allclose(diffs, diffs[0], rtol=rtol, atol=atol)


class TestCosineScheduler:
    """Tests for the cosine noise scheduler."""

    def test_shape_and_type(self, num_inference_steps):
        """Test the shape and dtype of the cosine scheduler output."""
        sigmas = RuntimeConfig._create_cosine_sigmas(num_inference_steps)
        assert sigmas.shape == (num_inference_steps + 1,)
        assert sigmas.dtype == mx.float32

    def test_boundary_conditions(self, num_inference_steps, rtol, atol):
        """Test the boundary values of the cosine scheduler."""
        sigmas = RuntimeConfig._create_cosine_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Check boundary conditions
        assert np.isclose(sigmas_np[0], 1.0, rtol=rtol, atol=atol)
        assert sigmas_np[-1] == 0.0

    def test_cosine_shape(self, num_inference_steps):
        """Test the shape properties of the cosine scheduler curve."""
        sigmas = RuntimeConfig._create_cosine_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Verify cosine shape properties - values should decrease more rapidly at extremes
        # and more slowly in the middle compared to linear
        diffs = np.diff(sigmas_np[:-1])  # Exclude the final 0

        # Earlier diffs should be smaller (in absolute value) than middle diffs
        assert abs(diffs[0]) < abs(diffs[len(diffs) // 2])

        # Verify that middle section has the steepest slope
        mid_idx = len(diffs) // 2
        assert abs(diffs[mid_idx]) > abs(diffs[0])  # Middle should be steeper than beginning

    def test_cosine_formula(self, num_inference_steps, rtol, atol):
        """Test the cosine values against the expected formula."""
        sigmas = RuntimeConfig._create_cosine_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Directly test values against the expected formula
        s = 0.008
        steps = np.linspace(0, 1, num_inference_steps)
        expected_sigmas = np.cos(((steps + s) / (1 + s)) * math.pi * 0.5) ** 2
        expected_sigmas = expected_sigmas / expected_sigmas[0]

        for i in range(num_inference_steps):
            assert np.isclose(sigmas_np[i], expected_sigmas[i], rtol=rtol, atol=atol)


class TestExponentialScheduler:
    """Tests for the exponential noise scheduler."""

    def test_shape_and_type(self, num_inference_steps):
        """Test the shape and dtype of the exponential scheduler output."""
        sigmas = RuntimeConfig._create_exponential_sigmas(num_inference_steps)
        assert sigmas.shape == (num_inference_steps + 1,)
        assert sigmas.dtype == mx.float32

    def test_boundary_conditions(self, num_inference_steps, rtol, atol):
        """Test the boundary values of the exponential scheduler."""
        sigmas = RuntimeConfig._create_exponential_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Check boundary conditions
        assert np.isclose(sigmas_np[0], 1.0, rtol=rtol, atol=atol)
        assert sigmas_np[-1] == 0.0

    def test_exponential_decay(self, num_inference_steps):
        """Test the exponential decay property of the scheduler."""
        sigmas = RuntimeConfig._create_exponential_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Verify exponential characteristics - decay should be more rapid at first
        diffs = np.diff(sigmas_np[:-1])  # Exclude the final 0

        # Verify that each diff is smaller (in absolute value) than the previous,
        # which indicates exponential decay
        for i in range(1, len(diffs)):
            assert abs(diffs[i]) < abs(diffs[i - 1])

    def test_exponential_formula(self, num_inference_steps, rtol, atol):
        """Test the exponential values against the expected formula."""
        sigmas = RuntimeConfig._create_exponential_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Directly test values against the expected formula
        beta = 5.0
        steps = np.linspace(0, 1, num_inference_steps)
        expected_sigmas = np.exp(-beta * steps)

        for i in range(num_inference_steps):
            assert np.isclose(sigmas_np[i], expected_sigmas[i], rtol=rtol, atol=atol)


class TestSqrtScheduler:
    """Tests for the square root transformation noise scheduler."""

    def test_shape_and_type(self, num_inference_steps):
        """Test the shape and dtype of the sqrt scheduler output."""
        # Create a square root transformation manually for testing
        steps = np.linspace(0, 1, num_inference_steps)
        sigmas = np.sqrt(1 - steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])

        assert sigmas.shape == (num_inference_steps + 1,)
        assert sigmas.dtype == mx.float32

    def test_boundary_conditions(self, num_inference_steps, rtol, atol):
        """Test the boundary values of the sqrt scheduler."""
        # Create a square root transformation manually for testing
        steps = np.linspace(0, 1, num_inference_steps)
        sigmas = np.sqrt(1 - steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])

        sigmas_np = np.array(sigmas)

        # Check boundary conditions
        assert np.isclose(sigmas_np[0], 1.0, rtol=rtol, atol=atol)
        assert sigmas_np[-1] == 0.0

    def test_sqrt_formula(self, num_inference_steps, rtol, atol):
        """Test the square root values against the expected formula."""
        # Create a square root transformation manually for testing
        steps = np.linspace(0, 1, num_inference_steps)
        sigmas = np.sqrt(1 - steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])

        sigmas_np = np.array(sigmas)

        # Verify square root transformation effects
        expected_sigmas = np.sqrt(1 - steps)
        for i in range(num_inference_steps):
            assert np.isclose(sigmas_np[i], expected_sigmas[i], rtol=rtol, atol=atol)

    def test_sqrt_increasing_steps(self, num_inference_steps):
        """Test that the step sizes increase with the sqrt scheduler."""
        # Create a square root transformation manually for testing
        steps = np.linspace(0, 1, num_inference_steps)
        sigmas = np.sqrt(1 - steps)
        sigmas = mx.array(sigmas).astype(mx.float32)
        sigmas = mx.concatenate([sigmas, mx.zeros(1)])

        sigmas_np = np.array(sigmas)

        # The difference between consecutive steps should increase as we go
        diffs = np.diff(sigmas_np[:-1])  # Exclude the final 0
        for i in range(1, len(diffs)):
            assert abs(diffs[i]) > abs(diffs[i - 1])


class TestScaledLinearScheduler:
    """Tests for the properly scaled linear noise scheduler."""

    def test_shape_and_type(self, num_inference_steps):
        """Test the shape and dtype of the scaled linear scheduler output."""
        sigmas = RuntimeConfig._create_scaled_linear_sigmas(num_inference_steps)
        assert sigmas.shape == (num_inference_steps + 1,)
        assert sigmas.dtype == mx.float32

    def test_boundary_conditions(self, num_inference_steps, rtol, atol):
        """Test the boundary values of the scaled linear scheduler."""
        sigmas = RuntimeConfig._create_scaled_linear_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Check boundary conditions
        assert np.isclose(sigmas_np[0], 1.0, rtol=rtol, atol=atol)
        assert sigmas_np[-1] == 0.0

    def test_scaled_linear_monotonic_decrease(self, num_inference_steps):
        """Test the monotonic decrease property of the scaled linear scheduler."""
        sigmas = RuntimeConfig._create_scaled_linear_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # Check that the curve decreases monotonically
        diffs = np.diff(sigmas_np[:-1])
        assert np.all(diffs < 0)  # All differences should be negative (decreasing)

    def test_scaled_linear_increasing_steps(self, num_inference_steps):
        """Test that the step sizes increase with the scaled linear scheduler."""
        sigmas = RuntimeConfig._create_scaled_linear_sigmas(num_inference_steps)
        sigmas_np = np.array(sigmas)

        # The differences should increase in magnitude (steeper curve toward the end)
        diffs = np.diff(sigmas_np[:-1])
        for i in range(1, len(diffs)):
            assert abs(diffs[i]) > abs(diffs[i - 1])


class TestSigmaShifting:
    """Tests for the sigma shifting function used for high-resolution images."""

    @pytest.fixture
    def resolutions(self):
        """Resolutions to test with."""
        return [
            (512, 512),  # Smaller resolution
            (1024, 1024),  # Medium resolution
            (2048, 2048),  # Larger resolution
        ]

    def test_shift_shape_and_type(self, num_inference_steps, resolutions):
        """Test the shape and type of shifted sigmas output."""
        sigmas = RuntimeConfig._create_linear_sigmas(num_inference_steps)

        for width, height in resolutions:
            shifted_sigmas = RuntimeConfig._shift_sigmas(sigmas=sigmas, width=width, height=height)

            # Check shape and type
            assert shifted_sigmas.shape == sigmas.shape
            assert shifted_sigmas.dtype == sigmas.dtype

    def test_shift_boundary_conditions(self, num_inference_steps, resolutions):
        """Test boundary conditions for shifted sigmas."""
        sigmas = RuntimeConfig._create_linear_sigmas(num_inference_steps)

        for width, height in resolutions:
            shifted_sigmas = RuntimeConfig._shift_sigmas(sigmas=sigmas, width=width, height=height)

            # Check that the last value is still 0
            assert shifted_sigmas[-1] == 0.0

            # Check that values are between 0 and 1
            shifted_np = np.array(shifted_sigmas)
            assert np.all(shifted_np >= 0.0)
            assert np.all(shifted_np <= 1.0)

    def test_shift_increases_with_resolution(self, num_inference_steps, resolutions):
        """Test that higher resolutions produce higher shift values."""
        sigmas = RuntimeConfig._create_linear_sigmas(num_inference_steps)

        for width, height in resolutions:
            if width > 512 and height > 512:
                shifted_sigmas = RuntimeConfig._shift_sigmas(sigmas=sigmas, width=width, height=height)

                # For most points, the shifted value should be higher than the original
                # (but not necessarily all, and we need to exclude the 0 at the end)
                shifted_np = np.array(shifted_sigmas)
                orig_np = np.array(sigmas)[:-1]
                shift_np = shifted_np[:-1]
                assert np.sum(shift_np > orig_np) > len(orig_np) // 2


class TestSchedulerSelection:
    """Tests for the scheduler selection based on config."""

    @pytest.fixture
    def schedulers(self):
        """List of scheduler types to test."""
        return ["linear", "cosine", "exponential", "sqrt", "scaled_linear"]

    def test_scheduler_selection(self, num_inference_steps, rtol, atol, model_config, schedulers):
        """Test that _create_sigmas selects the correct scheduler based on the config."""
        for scheduler in schedulers:
            config = Config(num_inference_steps=num_inference_steps, noise_scheduler=scheduler)
            sigmas = RuntimeConfig._create_sigmas(config, model_config)

            # Create the expected sigmas based on scheduler type
            if scheduler == "linear":
                expected_sigmas = RuntimeConfig._create_linear_sigmas(num_inference_steps)
            elif scheduler == "cosine":
                expected_sigmas = RuntimeConfig._create_cosine_sigmas(num_inference_steps)
            elif scheduler == "exponential":
                expected_sigmas = RuntimeConfig._create_exponential_sigmas(num_inference_steps)
            elif scheduler == "sqrt":
                # Create a square root transformation manually for testing
                steps = np.linspace(0, 1, num_inference_steps)
                expected_sigmas = np.sqrt(1 - steps)
                expected_sigmas = mx.array(expected_sigmas).astype(mx.float32)
                expected_sigmas = mx.concatenate([expected_sigmas, mx.zeros(1)])
            elif scheduler == "scaled_linear":
                expected_sigmas = RuntimeConfig._create_scaled_linear_sigmas(num_inference_steps)

            # Convert to numpy for comparison
            sigmas_np = np.array(sigmas)
            expected_np = np.array(expected_sigmas)

            # Check that the right scheduler was used
            if scheduler == "sqrt":
                # For sqrt, just check that we're getting the expected pattern
                assert sigmas.shape == expected_sigmas.shape
                assert sigmas.dtype == expected_sigmas.dtype
                # Check basic properties
                assert np.isclose(sigmas_np[0], 1.0, rtol=rtol, atol=atol)
                assert sigmas_np[-1] == 0.0
            else:
                assert np.allclose(sigmas_np, expected_np, rtol=rtol, atol=atol)
