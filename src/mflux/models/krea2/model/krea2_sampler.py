"""Sampling schedule for Krea-2.

Rectified-flow (``ModelType.FLUX``) sigma schedule with a static shift. The
turbo default is 8 steps, shift 1.15, CFG 1.0.

TODO(krea2): the reference turbo sampler is ``er_sde`` + ``simple`` scheduler;
this is a plain flow/Euler schedule for first light (see NOTES.md).
"""

import mlx.core as mx


def flow_sigmas(num_steps: int, shift: float = 1.15) -> mx.array:
    """Return ``num_steps + 1`` sigmas from 1 -> 0 with a static flow shift."""
    sigmas = mx.linspace(1.0, 0.0, num_steps + 1)
    return shift * sigmas / (1.0 + (shift - 1.0) * sigmas)
