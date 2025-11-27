import pytest

import mflux.models.common.schedulers as schedulers


@pytest.mark.fast
def test_scheduler_by_path():
    schedulers.try_import_external_scheduler("mflux.models.common.schedulers.linear_scheduler.LinearScheduler")


@pytest.mark.fast
def test_scheduler_bad_module():
    with pytest.raises(schedulers.SchedulerModuleNotFound):
        schedulers.try_import_external_scheduler("someone.other.project.BarScheduler")


@pytest.mark.fast
def test_scheduler_bad_classname():
    with pytest.raises(schedulers.SchedulerClassNotFound):
        schedulers.try_import_external_scheduler("mflux.models.common.schedulers.linear_scheduler.FooBarScheduler")
