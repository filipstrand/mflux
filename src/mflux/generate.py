from pathlib import Path

from mflux import Config, Flux1, ModelConfig, StopImageGenerationException

# image_path = "/Users/filipstrand/Desktop/lighthouse.png"
# source_prompt = "The image features a tall white lighthouse standing prominently on a hill, with a beautiful blue sky in the background. The lighthouse is illuminated by a bright light, making it a prominent landmark in the scene."
# target_prompt = "The image features Big ben clock tower standing prominently on a hill, with a beautiful blue sky in the background. The Big ben clock tower is illuminated by a bright light, making it a prominent landmark in the scene."

image_path = "/Users/filipstrand/Desktop/gas_station.png"
source_prompt = "A gas station with a white and red sign that reads 'CAFE' There are several cars parked in front of the gas station, including a white car and a van."
target_prompt = "A gas station with a white and red sign that reads 'CVPR' There are several cars parked in front of the gas station, including a white car and a van."

height = 512
width = 512
steps = 28
seed = 2
source_guidance = 1.5
target_guidance = 5.5


def main():
    # Load the model
    flux = Flux1(
        model_config=ModelConfig.FLUX1_DEV,
        quantize=4,
    )

    try:
        image = flux.generate_image(
            seed=seed,
            src_prompt=source_prompt,
            tar_prompt=target_prompt,
            src_guidance=source_guidance,
            tar_guidance=target_guidance,
            image_path=image_path,
            stepwise_output_dir=Path("/Users/filipstrand/Desktop/edit"),
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=0.0,
            ),
        )

        # Save the image
        image.save(path="edited.png")
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
