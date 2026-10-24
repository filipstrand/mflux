import pytest

from mflux.models.common.cfg_end.cfg_end import CfgEnd

pytestmark = pytest.mark.fast


@pytest.mark.parametrize(
    ("num_steps", "cfg_end", "cutoff"),
    [
        (30, None, 30),
        (30, 1.0, 30),
        (30, 0.6, 18),
        # Python rounds halves to even, as Ideogram 4 always did: 22.5 runs 22 steps with CFG, 37.5 runs 38
        (30, 0.75, 22),
        (50, 0.75, 38),
        (50, 0.6, 30),
        # At least the first step runs CFG
        (4, 0.1, 1),
    ],
)
def test_the_cutoff_is_the_rounded_share_of_the_steps(num_steps, cfg_end, cutoff):
    assert CfgEnd.cutoff(num_steps, cfg_end) == cutoff


@pytest.mark.parametrize("cfg_end", [0.0, -0.5, 1.5, float("nan")])
def test_a_cfg_end_outside_0_to_1_is_refused(cfg_end):
    with pytest.raises(ValueError, match="cfg_end must be > 0 and <= 1"):
        CfgEnd.cutoff(30, cfg_end)


def test_only_runs_that_set_cfg_end_record_it():
    assert CfgEnd.generation_parameters(None) == {}
    assert CfgEnd.generation_parameters(0.6) == {"cfg_end": 0.6}
