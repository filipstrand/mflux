from pathlib import Path

from mflux import Config, Flux1, ModelConfig, StopImageGenerationException
from mflux.flux.v_cache import VCache

image_path = "/Users/filipstrand/Desktop/cat.png"
source_prompt = "A cat"
target_prompt = "A sleeping cat"
height = 256
width = 256
steps = 20
seed = 2
source_guidance = 1.5
target_guidance = 5.5
VCache.t_max = 10


def main():
    # Load the model
    flux = Flux1(
        model_config=ModelConfig.FLUX1_DEV,
        quantize=4,
    )

    try:
        # Invert an existing image
        VCache.is_inverting = True
        inverted_latents, encoded_image = flux.invert(
            seed=seed,
            prompt=source_prompt,
            init_image_path=Path(image_path),
            stepwise_output_dir=Path("/Users/filipstrand/Desktop/backward"),
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=source_guidance,
            ),
        )

        # Generate a new image based on the inverted one
        VCache.is_inverting = False
        image = flux.generate_image(
            seed=seed,
            prompt=target_prompt,
            latents=inverted_latents,
            stepwise_output_dir=Path("/Users/filipstrand/Desktop/forward"),
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=target_guidance,
            ),
        )

        # Save the image
        image.save(path="edited.png")
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
