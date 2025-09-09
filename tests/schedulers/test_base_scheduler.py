import mlx.core as mx
import pytest

from mflux.schedulers.base_scheduler import BaseScheduler


def test_base_scheduler_is_abstract():
    """
    Test that BaseScheduler cannot be instantiated directly.
    """
    with pytest.raises(TypeError):
        BaseScheduler()
