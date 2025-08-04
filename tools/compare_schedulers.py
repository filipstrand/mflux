# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch",
#   "diffusers",
#   "numpy",
#   "mlx",
#   "tabulate",
# ]
# ///

"""
Compare mflux scheduler implementations with diffusers equivalents.
Usage: uv run tools/compare_schedulers.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    DDIMScheduler as DiffusersDDIM,
    EulerDiscreteScheduler as DiffusersEuler,
    FlowMatchEulerDiscreteScheduler as DiffusersFlowMatch,
)
from tabulate import tabulate

# Add parent directory to path to import mflux
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import mlx.core as mx

from mflux.schedulers import (
    DDIMScheduler as MfluxDDIM,
    EulerDiscreteScheduler as MfluxEuler,
    LinearScheduler as MfluxLinear,
)


def compare_arrays(name: str, diffusers_array, mflux_array, tolerance: float = 1e-5) -> dict:
    """Compare arrays from diffusers (torch) and mflux (mlx)."""
    # Convert to numpy for comparison
    if isinstance(diffusers_array, torch.Tensor):
        diff_np = diffusers_array.cpu().numpy()
    else:
        diff_np = np.array(diffusers_array)

    if hasattr(mflux_array, "__array__"):
        mflux_np = np.array(mflux_array)
    else:
        mflux_np = mflux_array

    # Ensure same shape
    if diff_np.shape != mflux_np.shape:
        return {
            "name": name,
            "match": "❌",
            "max_diff": "Shape mismatch",
            "notes": f"Diffusers: {diff_np.shape}, mflux: {mflux_np.shape}",
        }

    # Calculate differences
    max_diff = np.max(np.abs(diff_np - mflux_np))
    match = "✅" if max_diff < tolerance else "❌"

    return {
        "name": name,
        "match": match,
        "max_diff": f"{max_diff:.2e}",
        "notes": f"First values: {diff_np.flat[:3]}...",
    }


def compare_linear_schedulers():
    """Compare LinearScheduler implementations."""
    print("\n=== LinearScheduler Comparison ===")
    print("Note: mflux implements a simple linear schedule, while diffusers FlowMatch has more features")

    # Create schedulers
    # Note: diffusers FlowMatch uses shift=1 by default for base model
    diff_scheduler = DiffusersFlowMatch(shift=1, use_dynamic_shifting=False)
    mflux_scheduler = MfluxLinear()

    # Set timesteps
    num_steps = 50
    diff_scheduler.set_timesteps(num_steps)
    mflux_scheduler.set_timesteps(num_steps)

    # Also show what mflux actually generates (linear from 1 to 1/n)
    print(f"\nmflux linear sigmas (first 5): {np.array(mflux_scheduler.sigmas[:5])}")
    print(f"diffusers sigmas (first 5): {diff_scheduler.sigmas[:5].cpu().numpy()}")

    results = []

    # Compare sigmas
    results.append(compare_arrays("Sigmas", diff_scheduler.sigmas, mflux_scheduler.sigmas))

    # Compare timesteps
    results.append(compare_arrays("Timesteps", diff_scheduler.timesteps, mflux_scheduler.timesteps))

    # Test step function
    sample_shape = (1, 4, 64, 64)
    torch_sample = torch.randn(sample_shape)
    mlx_sample = mx.random.normal(sample_shape)

    torch_noise = torch.randn(sample_shape) * 0.1
    mlx_noise = mx.random.normal(sample_shape) * 0.1

    # Perform step at timestep index 10
    # For diffusers, we need to use the actual timestep value, not the index
    timestep_idx = 10
    if timestep_idx < len(diff_scheduler.timesteps):
        diff_timestep = diff_scheduler.timesteps[timestep_idx]
        diff_output = diff_scheduler.step(torch_noise, diff_timestep, torch_sample, return_dict=False)[0]
        mflux_output = mflux_scheduler.step(mlx_noise, timestep_idx, mlx_sample)
    else:
        # If we don't have enough timesteps, use the first one
        diff_timestep = diff_scheduler.timesteps[0]
        diff_output = diff_scheduler.step(torch_noise, diff_timestep, torch_sample, return_dict=False)[0]
        mflux_output = mflux_scheduler.step(mlx_noise, 0, mlx_sample)

    # Note: Can't directly compare outputs due to different random inputs
    results.append(
        {
            "name": "Step output shape",
            "match": "✅" if diff_output.shape == tuple(mflux_output.shape) else "❌",
            "max_diff": "N/A",
            "notes": f"Shapes: {diff_output.shape} vs {mflux_output.shape}",
        }
    )

    print(tabulate(results, headers="keys", tablefmt="grid"))


def compare_ddim_schedulers():
    """Compare DDIMScheduler implementations."""
    print("\n=== DDIMScheduler Comparison ===")

    # Create schedulers with same parameters
    diff_scheduler = DiffusersDDIM(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )
    mflux_scheduler = MfluxDDIM(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )

    # Set timesteps
    num_steps = 50
    diff_scheduler.set_timesteps(num_steps)
    mflux_scheduler.set_timesteps(num_steps)

    results = []

    # Compare betas
    results.append(compare_arrays("Betas", diff_scheduler.betas, mflux_scheduler.betas))

    # Compare alphas_cumprod
    results.append(compare_arrays("Alphas cumprod", diff_scheduler.alphas_cumprod, mflux_scheduler.alphas_cumprod))

    # Compare timesteps (note: DDIM reverses them)
    results.append(compare_arrays("Timesteps", diff_scheduler.timesteps, mflux_scheduler.timesteps))

    print(tabulate(results, headers="keys", tablefmt="grid"))


def compare_euler_discrete_schedulers():
    """Compare EulerDiscreteScheduler implementations."""
    print("\n=== EulerDiscreteScheduler Comparison ===")

    # Create schedulers
    diff_scheduler = DiffusersEuler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )
    mflux_scheduler = MfluxEuler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )

    # Set timesteps
    num_steps = 50
    diff_scheduler.set_timesteps(num_steps)
    mflux_scheduler.set_timesteps(num_steps)

    results = []

    # Compare sigmas (Euler uses sigma parameterization)
    results.append(compare_arrays("Sigmas", diff_scheduler.sigmas, mflux_scheduler.sigmas))

    # Compare timesteps
    results.append(compare_arrays("Timesteps", diff_scheduler.timesteps, mflux_scheduler.timesteps))

    # Test with Karras sigmas
    print("\n--- With Karras Sigmas ---")
    diff_karras = DiffusersEuler(use_karras_sigmas=True)
    mflux_karras = MfluxEuler(use_karras_sigmas=True)

    diff_karras.set_timesteps(10)
    mflux_karras.set_timesteps(10)

    karras_results = []
    karras_results.append(compare_arrays("Karras sigmas", diff_karras.sigmas, mflux_karras.sigmas))

    results.extend(karras_results)

    print(tabulate(results, headers="keys", tablefmt="grid"))


def main():
    """Run all scheduler comparisons."""
    print("Comparing mflux schedulers with diffusers implementations...")
    print("=" * 60)
    print("\nIMPORTANT: mflux's 'LinearScheduler' implements a simple linear schedule")
    print("(sigma = linspace(1, 1/n, n)), which is different from diffusers' FlowMatch.")

    compare_linear_schedulers()
    compare_ddim_schedulers()
    compare_euler_discrete_schedulers()

    print("\n" + "=" * 60)
    print("Notes:")
    print("- ✅ indicates values match within tolerance (1e-5)")
    print("- ❌ indicates significant differences")
    print("- Shape mismatches may indicate different conventions")
    print("- Step function outputs can't be directly compared due to different random inputs")


if __name__ == "__main__":
    main()
