from mflux import Config, Flux1, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler


def main():
    # 1. Load the model
    flux = Flux1(
        model_config=ModelConfig.dev_fill(),
        quantize=8,
    )

    handler = StepwiseHandler(flux=flux, output_dir="/Users/filipstrand/Desktop/stepwise")
    CallbackRegistry.register_before_loop(handler)
    CallbackRegistry.register_in_loop(handler)
    CallbackRegistry.register_interrupt(handler)

    try:
        for seed in [1]:
            # 3. Generate an image for each seed value
            image = flux.generate_image(
                seed=seed,
                prompt="a white paper cup",
                config=Config(
                    num_inference_steps=25,
                    height=int(1632 / 1),
                    width=int(1232 / 1),
                    guidance=30,
                    init_image_path="/Users/filipstrand/Desktop/cup.png",
                    masked_image_path="/Users/filipstrand/Desktop/cup_mask.png",
                ),
            )
            # 4. Save the image
            image.save(path="output.png")
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
