import pytest

from mflux.callbacks.callback_registry import CallbackRegistry


@pytest.fixture(autouse=True)
def clear_callback_registry():
    CallbackRegistry.clear()
    yield
    CallbackRegistry.clear()
