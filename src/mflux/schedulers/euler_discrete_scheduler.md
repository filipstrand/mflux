# Euler Discrete Scheduler: Porting and Verification

This document details the process of porting the `EulerDiscreteScheduler` from the Hugging Face `diffusers` library to the `mflux` (MLX) framework.

---

## 1. Porting from `diffusers`

The primary source for the port is the official `diffusers` implementation:
[`diffusers/src/diffusers/schedulers/scheduling_euler_discrete.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_euler_discrete.py)

The goal was to create a functional MLX version of this scheduler.

### Key Porting Steps and Iterations

1.  **Framework Translation**: All `torch` tensor operations were translated to their `mlx.core` equivalents (`torch.linspace` -> `mx.linspace`, etc.).

2.  **Karras Sigmas**: The logic for the improved Karras noise schedule (`use_karras_sigmas=True`) was ported, which generates a descending schedule.

---

## 2. Consistency with Foundational Papers

To validate the implementation, it was compared against the principles of relevant papers.

### A. DDPM Paper (Ho et al., 2020)

**Paper**: "Denoising Diffusion Probabilistic Models" ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239))

*   **Noise Schedule**: The underlying `linear` and `scaled_linear` beta schedules, which are used to generate the initial sigmas, are consistent with those used in DDPM-based models.
*   **Sigma Parameterization**: The scheduler converts the `alphas` from the DDPM framework into `sigmas` using the standard formula `sigma = sqrt((1 - alpha_bar) / alpha_bar)`.

### B. Karras et al. (2022)

**Paper**: "Elucidating the Design Space of Diffusion-Based Generative Models" ([arXiv:2206.00364](https://arxiv.org/abs/2206.00364))

*   **Euler Method**: The `step` function's use of the Euler method (`prev_sample = sample + derivative * dt`) to solve the probability flow ODE is a core concept from this paper.
*   **Karras Sigmas**: When `use_karras_sigmas=True`, the scheduler uses the improved noise schedule proposed in this paper, which can lead to higher-quality results in fewer steps.

---

## 3. Verification

The implementation was validated through two methods:

1.  **Unit Tests**: A test suite at `tests/schedulers/test_euler_discrete_scheduler.py` was created and updated to validate the final implementation against expected numerical outputs and properties (like sigma ordering).

2.  **Visualization**: The script `tools/visualize_scheduler_euler_discrete.py` was created to plot the sigma schedules. The resulting plot (`docs/euler_discrete_schedule_visualization.png`) visually confirms the descending order of the inference sigmas.
