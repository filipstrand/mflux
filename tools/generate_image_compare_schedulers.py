import os
from pathlib import Path

import mlx.core as mx

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig  # New import
from mflux.flux.flux import Flux1
from mflux.schedulers.ddim_scheduler import DDIMScheduler
from mflux.schedulers.euler_discrete_scheduler import EulerDiscreteScheduler
from mflux.schedulers.linear_scheduler import LinearScheduler

OUTPUT_DIR = Path("docs/scheduler_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    """Generates three images from a single prompt and seed, using three different schedulers for comparison."""
    # --- Setup ---
    print(f"Saving images to: {OUTPUT_DIR}")
    seed = 1234
    mx.random.seed(seed)

    # --- Load Model ---
    # Initialize the model directly with an explicit base_model to fix the config error
    model_config = ModelConfig.from_name("schnell")
    flux = Flux1(model_config=model_config)

    # --- Define Schedulers to Test ---
    schedulers_to_test = {
        "linear": LinearScheduler(),
        "ddim": DDIMScheduler(),
        "euler_discrete": EulerDiscreteScheduler(),
    }

    # --- Generation Loop ---
    for name, scheduler in schedulers_to_test.items():
        print(f"\n--- Generating with {name.upper()} Scheduler ---")
        image = flux.generate_image(
            seed=seed,
            prompt="A photograph of an astronaut riding a horse on the moon.",
            config=Config(num_inference_steps=14, height=1024, width=1024, guidance=2.5),
            scheduler=scheduler,
        )

        # Save the image
        output_path = OUTPUT_DIR / f"{name}_scheduler_seed_{seed}.png"
        image.save(path=str(output_path))
        print(f"Image saved to: {output_path}")

    print("\nComparison generation complete.")


if __name__ == "__main__":
    main()
