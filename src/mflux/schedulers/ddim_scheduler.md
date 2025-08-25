# DDIM Scheduler: Porting Process

This document details the process of porting the `DDIMScheduler` from the Hugging Face `diffusers` library to the `mflux` (MLX) framework.

---

## 1. Porting from `diffusers`

The primary source for the port is the official `diffusers` implementation:
[`diffusers/src/diffusers/schedulers/scheduling_ddim.py`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py)

The goal was to create a faithful and functional MLX version of this scheduler.

### Key Porting Steps

1.  **Framework Translation**: All `torch` tensor operations were translated to their `mlx.core` equivalents. This includes, but is not limited to:
    *   `torch.linspace` -> `mx.linspace`
    *   `torch.cumprod` -> `mx.cumprod`
    *   `torch.randn` -> `mx.random.normal`
    *   `tensor.sqrt()` -> `mx.sqrt(tensor)`
    *   `torch.cat` -> `mx.concatenate`

2.  **Parameter Parity**: The scheduler's `__init__` method includes almost the nearly set of configuration parameters from the `diffusers` version. This ensures that the scheduler is just as configurable, supporting options like:
    *   `clip_sample`
    *   `set_alpha_to_one`
    *   `timestep_spacing` (including `"leading"`, `"trailing"`, and `"linspace"`)
    *   `rescale_betas_zero_snr`

3.  **Helper Functions**: Utility functions required by the scheduler, such as `betas_for_alpha_bar` and `rescale_zero_terminal_snr`, were also ported from `diffusers` and converted to use MLX operations.

---

## 2. Consistency with Foundational Papers

To ensure correctness, the implementation was verified against the two foundational papers for this type of scheduler.

### A. DDPM Paper (Ho et al., 2020)

**Paper**: "Denoising Diffusion Probabilistic Models" ([arXiv:2006.11239](https://arxiv.org/abs/2006.11239))

Our scheduler is consistent with the core principles of the original DDPM:

*   **Noise Schedule**: The default `linear` schedule in our scheduler, with `beta_start=0.0001` and `beta_end=0.02`, is identical to the one proposed and used in the DDPM paper.
*   **Forward Process**: The `add_noise()` method is a direct implementation of the forward noising process (Equation 4 in the paper), which defines how a noisy image `x_t` is generated from an original image `x_0`.

### B. DDIM Paper (Song et al., 2020)

**Paper**: "Denoising Diffusion Implicit Models" ([arXiv:2010.02502](https://arxiv.org/abs/2010.02502))

Our scheduler correctly implements the more advanced inference logic from the DDIM paper.

*   **Reverse Process**: The `step()` function implements the DDIM reverse process, which is a generalization of the DDPM process. It allows for deterministic sampling and much faster inference by taking larger steps.
*   **The `eta` Parameter**: The consistency between the two models is captured by the `eta` parameter in our `step` function:
    *   When `eta = 0`, the process is deterministic, which is the main innovation of DDIM.
    *   When `eta = 1`, the variance term calculated by `_get_variance` makes the sampling process stochastic and equivalent to the DDPM process described in the first paper.
