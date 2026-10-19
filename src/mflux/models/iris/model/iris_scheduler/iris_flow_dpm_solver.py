import math

import mlx.core as mx
import numpy as np


class IrisFlowDPMSolver:
    # Keeps the previous clean predictions for the second-order step, so it serves one run only: make a new one for
    # every image.
    NUM_TRAIN_TIMESTEPS = 1000

    def __init__(self, num_inference_steps: int, shift: float = 4.0, order: int = 2):
        self.num_inference_steps = num_inference_steps
        self.order = order
        sigma = 1.0 - np.linspace(1.0, 0.001, num_inference_steps + 1, dtype=np.float64)
        shifted = shift * sigma / (1 + (shift - 1) * sigma)
        self.grid = [float(s) for s in shifted[::-1]]
        self.history: list[tuple[float, mx.array]] = []

    def model_time(self, step: int) -> mx.array:
        return mx.array([self.grid[step] * IrisFlowDPMSolver.NUM_TRAIN_TIMESTEPS], dtype=mx.float32)

    def step(self, latents: mx.array, output: mx.array, step: int) -> mx.array:
        s, t = self.grid[step], self.grid[step + 1]
        x0 = latents - s * output
        self.history.append((s, x0))
        i = step + 1
        order = min(i, self.order, self.num_inference_steps + 1 - i)
        if order == 1:
            latents = self._first_order(latents, s, t, x0)
        else:
            latents = self._second_order(latents, self.history[-2], self.history[-1], t)
        if len(self.history) > 2:
            self.history.pop(0)
        return latents

    @staticmethod
    def _lambda(t: float) -> float:
        return math.inf if t <= 0.0 else math.log((1.0 - t) / t)

    @staticmethod
    def _first_order(x: mx.array, s: float, t: float, x0: mx.array) -> mx.array:
        h = IrisFlowDPMSolver._lambda(t) - IrisFlowDPMSolver._lambda(s)
        phi1 = math.expm1(-h)
        return (t / s) * x - (1.0 - t) * phi1 * x0

    @staticmethod
    def _second_order(x: mx.array, prev1: tuple[float, mx.array], prev0: tuple[float, mx.array], t: float) -> mx.array:
        (s1, x0_1), (s0, x0_0) = prev1, prev0
        lam_t = IrisFlowDPMSolver._lambda(t)
        lam_0 = IrisFlowDPMSolver._lambda(s0)
        lam_1 = IrisFlowDPMSolver._lambda(s1)
        h, h0 = lam_t - lam_0, lam_0 - lam_1
        r0 = h0 / h
        d = (x0_0 - x0_1) / r0
        phi1 = math.expm1(-h)
        return (t / s0) * x - (1.0 - t) * phi1 * x0_0 - 0.5 * (1.0 - t) * phi1 * d
