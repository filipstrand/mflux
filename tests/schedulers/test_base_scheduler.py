import mlx.core as mx
import pytest

from mflux.schedulers.base_scheduler import BaseScheduler


def test_base_scheduler_is_abstract():
    """
    Test that BaseScheduler cannot be instantiated directly.
    """
    with pytest.raises(TypeError):
        BaseScheduler()


def test_sigmas_must_be_implemented():
    """
    Test that subclasses of BaseScheduler must implement the sigmas property.
    """

    class IncompleteScheduler(BaseScheduler):
        def scale_model_input(self, latents: mx.array) -> mx.array:
            return latents

    with pytest.raises(NotImplementedError):
        IncompleteScheduler().sigmas


def test_scale_model_input_default_behavior():
    """
    Test the default behavior of scale_model_input.
    """

    class CompleteScheduler(BaseScheduler):
        @property
        def sigmas(self) -> mx.array:
            return mx.array([1.0, 0.5, 0.0])

        def scale_model_input(self, latents: mx.array) -> mx.array:
            return latents

    scheduler = CompleteScheduler()
    test_input = mx.array([1, 2, 3])
    assert mx.array_equal(scheduler.scale_model_input(test_input), test_input)
