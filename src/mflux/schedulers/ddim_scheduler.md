# DDIM Scheduler: Porting and Verification

This document details the process of porting the `DDIMScheduler` from the Hugging Face `diffusers` library to the `mflux` (MLX) framework. It also verifies the implementation's consistency with foundational papers on diffusion models.

---

## 1. Porting from `diffusers`

The primary source for the port is the official `diffusers` implementation:
[`diffusers/src/diffusers/schedulers/scheduling_ddim.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py)

The goal was to create a functional MLX version of this scheduler.

### Key Porting Steps

1.  **Framework Translation**: All `torch` tensor operations were translated to their `mlx.core` equivalents. This includes, but is not limited to:
    *   `torch.linspace` -> `mx.linspace`
    *   `torch.cumprod` -> `mx.cumprod`
    *   `torch.randn` -> `mx.random.normal`
    *   `tensor.sqrt()` -> `mx.sqrt(tensor)`
    *   `torch.cat` -> `mx.concatenate`

2.  **Parameter Parity**: The scheduler's `__init__` method was updated to include the full set of configuration parameters from the `diffusers` version to allow for similar configurability, supporting options like:
    *   `clip_sample`
    *   `set_alpha_to_one`
    *   `timestep_spacing` (including `"leading"`, `"trailing"`, and `"linspace"`)
    *   `rescale_betas_zero_snr`

3.  **Algorithmic Alignment**: The initial `mflux` implementation had a formula in the `step` function that differed from the reference. This was updated to align with the DDIM algorithm as implemented in `diffusers`.

4.  **Helper Functions**: Utility functions required by the scheduler, such as `betas_for_alpha_bar` and `rescale_zero_terminal_snr`, were also ported from `diffusers` and converted to use MLX operations.

---

## 2. Consistency with Foundational Papers

To validate the implementation, it was compared against the two foundational papers for this type of scheduler.

### A. DDPM Paper (Ho et al., 2020)

**Paper**: "Denoising Diffusion Probabilistic Models" ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239))

The scheduler's design is consistent with the core principles of the original DDPM:

*   **Noise Schedule**: The default `linear` schedule, with `beta_start=0.0001` and `beta_end=0.02`, is identical to the one proposed and used in the DDPM paper.
*   **Forward Process**: The `add_noise()` method is a direct implementation of the forward noising process (Equation 4 in the paper).

### B. DDIM Paper (Song et al., 2020)

**Paper**: "Denoising Diffusion Implicit Models" ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502))

The scheduler implements the inference logic from the DDIM paper.

*   **Reverse Process**: The `step()` function implements the DDIM reverse process, which is a generalization of the DDPM process that allows for deterministic sampling.
*   **The `eta` Parameter**: The relationship between the two models is handled by the `eta` parameter in the `step` function:
    *   When `eta = 0`, the process is deterministic, which is the main innovation of DDIM.
    *   When `eta = 1`, the variance term calculated by `_get_variance` makes the sampling process stochastic, similar to the DDPM process.