import pytest

import mflux.schedulers


def test_scheduler_by_path():
    mflux.schedulers.try_import_external_scheduler("mflux.schedulers.linear_scheduler.LinearScheduler")


def test_scheduler_bad_module():
    with pytest.raises(mflux.schedulers.SchedulerModuleNotFound):
        mflux.schedulers.try_import_external_scheduler("someone.other.project.BarScheduler")


def test_scheduler_bad_classname():
    with pytest.raises(mflux.schedulers.SchedulerClassNotFound):
        mflux.schedulers.try_import_external_scheduler("mflux.schedulers.linear_scheduler.FooBarScheduler")
