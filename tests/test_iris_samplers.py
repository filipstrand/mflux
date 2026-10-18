import math

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.iris.model.iris_scheduler.iris_flow_dpm_solver import IrisFlowDPMSolver
from mflux.models.iris.model.iris_scheduler.iris_turbo_scheduler import IrisTurboScheduler


def _toy():
    mx.random.seed(0)
    return mx.random.normal((1, 3, 4, 4)), mx.random.normal((1, 3, 4, 4))  # noise, clean image


@pytest.mark.fast
class TestIrisTurboScheduler:
    def test_model_time_is_one_thousand_times_sigma(self):
        scheduler = IrisTurboScheduler()

        assert [scheduler.model_time(i).item() for i in range(4)] == pytest.approx([1000.0, 938.0, 833.0, 625.0])

    def test_an_exact_velocity_lands_on_the_flow_between_noise_and_image_at_every_step(self):
        # With v = noise - clean the state at sigma is (1 - sigma) clean + sigma noise, and every clean estimate is exact.
        noise, clean = _toy()
        scheduler = IrisTurboScheduler()
        latents = noise

        for step in range(4):
            latents, estimate = scheduler.step(latents, noise - clean, step)
            assert mx.allclose(estimate, clean, atol=1e-5).item()
            if step < 3:
                sigma = scheduler.sigmas[step + 1]
                assert mx.allclose(latents, (1 - sigma) * clean + sigma * noise, atol=1e-5).item()
        assert mx.allclose(latents, clean, atol=1e-5).item()

    def test_the_last_step_returns_the_clean_estimate_as_the_state(self):
        noise, _ = _toy()
        scheduler = IrisTurboScheduler()

        latents, estimate = scheduler.step(noise, noise * 0.5, 3)

        assert mx.array_equal(latents, estimate).item()
        assert mx.allclose(estimate, noise - 0.625 * noise * 0.5).item()


@pytest.mark.fast
class TestIrisFlowDPMSolver:
    def test_time_grid_is_the_shifted_linspace_descending_to_exactly_zero(self):
        solver = IrisFlowDPMSolver(num_inference_steps=8, shift=4.0)

        sigma = 1.0 - np.linspace(1.0, 0.001, 9)
        expected = (4 * sigma / (1 + 3 * sigma))[::-1]
        assert solver.grid == pytest.approx(list(expected))
        assert solver.grid[-1] == 0.0
        assert all(a > b for a, b in zip(solver.grid, solver.grid[1:]))
        assert solver.grid[0] == pytest.approx(4 * 0.999 / (1 + 3 * 0.999))
        assert solver.model_time(0).item() == pytest.approx(solver.grid[0] * 1000, rel=1e-6)

    @pytest.mark.parametrize("steps", [1, 2, 3, 8])
    def test_an_exact_data_prediction_follows_the_flow_and_ends_on_the_clean_image(self, steps):
        noise, clean = _toy()
        solver = IrisFlowDPMSolver(num_inference_steps=steps)
        latents = solver.grid[0] * noise + (1 - solver.grid[0]) * clean

        for step in range(steps):
            s = solver.grid[step]
            output = (latents - clean) / s  # x0 = x - s * output = clean
            latents = solver.step(latents, output, step)
            t = solver.grid[step + 1]
            assert mx.allclose(latents, t * noise + (1 - t) * clean, atol=2e-4).item()

        assert mx.allclose(latents, clean, atol=2e-4).item()

    def test_the_second_order_step_tracks_a_linearly_drifting_estimate_much_closer_than_first_order(self):
        # x_t = (t/s) x_s + t * int e^lambda x0(lambda) dlambda is the exact update for a clean estimate that is linear
        # in the log-SNR lambda; the first step has no history and is first-order, the second uses the previous estimate.
        solver = IrisFlowDPMSolver(num_inference_steps=4)
        a, b = 0.3, -0.2

        def lam(sigma):
            return math.log((1 - sigma) / sigma)

        def x0(sigma):
            return a + b * lam(sigma)

        x = mx.array([0.7])
        for step in range(2):
            s, t = solver.grid[step], solver.grid[step + 1]
            before = float(x[0])
            first_order = IrisFlowDPMSolver._first_order(x, s, t, mx.array([x0(s)])).item()
            x = solver.step(x, (x - x0(s)) / s, step)
        lambdas = np.linspace(lam(s), lam(t), 20001)
        exact = (t / s) * before + t * np.trapezoid(np.exp(lambdas) * (a + b * lambdas), lambdas)

        assert abs(float(x[0]) - exact) < abs(first_order - exact) / 5
        assert float(x[0]) != pytest.approx(first_order, abs=1e-4)

    def test_the_last_step_is_first_order_so_it_projects_exactly_onto_the_estimate(self):
        solver = IrisFlowDPMSolver(num_inference_steps=3)
        x = mx.array([0.5])
        for step in range(3):
            output = mx.array([0.25])
            expected_x0 = x - solver.grid[step] * output
            x = solver.step(x, output, step)

        assert x.item() == pytest.approx(expected_x0.item(), abs=1e-6)
