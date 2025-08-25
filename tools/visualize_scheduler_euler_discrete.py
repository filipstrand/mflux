import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mflux.schedulers.euler_discrete_scheduler import EulerDiscreteScheduler


def main():
    """Generates and saves a plot comparing the EulerDiscreteScheduler's sigma schedules."""
    num_inference_steps = 50

    plt.figure(figsize=(12, 8))

    # --- Plot 1: Standard Sigma Schedule ---
    scheduler_std = EulerDiscreteScheduler(use_karras_sigmas=False)
    scheduler_std.set_timesteps(num_inference_steps)
    sigmas_std = np.array(scheduler_std.sigmas)
    timesteps_std = np.arange(len(sigmas_std))
    plt.plot(timesteps_std, sigmas_std, marker="o", linestyle="-", label="Standard Sigmas")

    # --- Plot 2: Karras Sigma Schedule ---
    scheduler_karras = EulerDiscreteScheduler(use_karras_sigmas=True)
    scheduler_karras.set_timesteps(num_inference_steps)
    sigmas_karras = np.array(scheduler_karras.sigmas)
    timesteps_karras = np.arange(len(sigmas_karras))
    plt.plot(timesteps_karras, sigmas_karras, marker="x", linestyle="--", label="Karras Sigmas")

    # --- Configure and save the plot ---
    plt.title(f"Euler Discrete Scheduler Sigma Comparison (for {num_inference_steps} steps)")
    plt.xlabel("Inference Timestep")
    plt.ylabel("Sigma (Noise Level) - Log Scale")
    plt.yscale("log")  # Use a log scale for better visibility of sigma range
    plt.legend()
    plt.grid(True, which="both", ls="--")

    # Get the directory of the current script and navigate to the project's docs/ folder
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / ".." / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = "euler_discrete_schedule_visualization.png"
    output_path = output_dir / output_filename
    plt.savefig(output_path)

    print(f"Scheduler visualization saved to '{output_path}'")

    # Automatically open the file using the system's 'open' command
    if sys.platform == "darwin":  # 'open' is specific to macOS
        try:
            subprocess.run(["open", output_path], check=True)
            print(f"Opening '{output_path}'...")
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            print(f"Could not automatically open the file: {e}")
    else:
        print(f"Please open the visualization manually at: {output_path}")


if __name__ == "__main__":
    # Note: You may need to install matplotlib: pip install matplotlib
    main()
