import io

import mlx.core as mx
import numpy as np
import pytest

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.schedulers import try_import_external_scheduler
from mflux.schedulers.linear_scheduler import LinearScheduler


def test_linear_scheduler_import_by_name():
    assert try_import_external_scheduler("mflux.schedulers.linear_scheduler.LinearScheduler") == LinearScheduler


@pytest.fixture
def test_runtime_config():
    return RuntimeConfig(
        Config(
            # these are the only attributes relevant to schedulers
            num_inference_steps=14,
            width=1024,
            height=1024,
            scheduler_type="linear",
        ),
        ModelConfig.dev(),  # requires_sigma_shift=True
    )


def test_linear_scheduler_initialization(test_runtime_config):
    """
    Test the initialization of the LinearScheduler.
    """
    scheduler = LinearScheduler(runtime_config=test_runtime_config)
    assert scheduler.sigmas is not None
    assert isinstance(scheduler.sigmas, mx.array)
    assert len(scheduler.sigmas) > 0


def test_linear_scheduler_sigmas_property_no_shift(test_runtime_config):
    """
    Test the sigmas property of the LinearScheduler without sigma shift.
    """
    scheduler = LinearScheduler(runtime_config=test_runtime_config)
    expected_sigmas_from_mflux_0_9_0 = mx.array(
        np.load(
            io.BytesIO(
                bytes.fromhex(
                    # see: https://gist.github.com/anthonywu/2832147ff5f5f50c81df4d13152d2bed
                    "934e554d505901003e007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a202831352c20297d20202020200a0000803fb7e9793fd92a733f7aa66b3fab38633f30b4593f59df4e3f4f6f423f4b01343f1b10233fc3e30e3f5cedec3e0a90b03e4025483e00000000"
                )
            )
        )
    )

    assert mx.allclose(scheduler.sigmas, expected_sigmas_from_mflux_0_9_0)
    assert scheduler.sigmas.shape == (test_runtime_config.num_inference_steps + 1,)


def test_linear_scheduler_sigmas_property_with_shift(test_runtime_config):
    """
    Test the sigmas property of the LinearScheduler with sigma shift.
    """
    test_runtime_config.model_config = ModelConfig.schnell()  # requires_sigma_shift=True
    scheduler = LinearScheduler(runtime_config=test_runtime_config)
    expected_sigmas_from_mflux_0_9_0 = mx.array(
        np.load(
            io.BytesIO(
                bytes.fromhex(
                    # see: https://gist.github.com/anthonywu/2832147ff5f5f50c81df4d13152d2bed
                    "934e554d505901003e007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a20547275652c20277368617065273a202831352c20297d20202020200a0000803fdbb66d3fb76d5b3f9224493f6edb363f4992243f2549123f0000003fb76ddb3e6edbb63e2549923eb76d5b3e2549123e2549923d00000000"
                )
            )
        )
    )
    assert mx.allclose(scheduler.sigmas, expected_sigmas_from_mflux_0_9_0)
    assert scheduler.sigmas.shape == (test_runtime_config.num_inference_steps + 1,)
