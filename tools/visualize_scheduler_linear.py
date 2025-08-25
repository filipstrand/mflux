import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mflux.schedulers.linear_scheduler import LinearScheduler


def main():
    """Generates and saves a plot showing the LinearScheduler's sigma schedule."""
    num_inference_steps = 30  # The number of steps for inference

    # Initialize the scheduler
    scheduler = LinearScheduler()
    scheduler.set_timesteps(num_inference_steps)

    # Extract sigmas and convert from mlx.array to numpy.array for plotting
    sigmas = np.array(scheduler.sigmas)
    timesteps = np.arange(len(sigmas))

    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.plot(timesteps, sigmas, marker="o", linestyle="-", label="Sigma values")

    # Configure and save the plot
    plt.title(f"Linear Scheduler Sigma Schedule (for {num_inference_steps} steps)")
    plt.xlabel("Inference Timestep")
    plt.ylabel("Sigma (Noise Level)")
    plt.legend()
    plt.grid(True)
    plt.xticks(np.arange(0, len(sigmas), step=max(1, len(sigmas) // 10)))

    # Get the directory of the current script and navigate to the project's docs/ folder
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / ".." / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = "linear_schedule_visualization.png"
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
