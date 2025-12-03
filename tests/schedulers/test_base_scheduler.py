import pytest

from mflux.models.common.schedulers.base_scheduler import BaseScheduler


@pytest.mark.fast
def test_base_scheduler_is_abstract():
    with pytest.raises(TypeError):
        BaseScheduler()
