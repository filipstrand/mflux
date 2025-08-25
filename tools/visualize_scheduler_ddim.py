import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mflux.schedulers.ddim_scheduler import DDIMScheduler


def main():
    """Generates and saves a plot comparing DDIM scheduler noise schedules."""
    num_train_timesteps = 1000
    schedule_types = ["linear", "scaled_linear", "squaredcos_cap_v2"]

    plt.figure(figsize=(12, 8))

    for schedule in schedule_types:
        scheduler = DDIMScheduler(num_train_timesteps=num_train_timesteps, beta_schedule=schedule)
        alphas_cumprod = np.array(scheduler.alphas_cumprod)
        timesteps = np.arange(num_train_timesteps)
        plt.plot(timesteps, alphas_cumprod, label=schedule)

    plt.title("Comparison of DDIM Noise Schedules")
    plt.xlabel("Timesteps")
    plt.ylabel("Alphas Cumulative Product (Signal Rate)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 1)
    plt.xlim(0, num_train_timesteps)

    # Get the directory of the current script and navigate to the project's docs/ folder
    script_dir = Path(__file__).parent.resolve()
    output_dir = script_dir / ".." / "docs"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = "ddim_schedules_visualization.png"
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
