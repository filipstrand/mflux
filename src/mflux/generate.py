from mflux import Config, Flux1, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
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
        model_config=ModelConfig.dev(),
        quantize=4,
    )

    # 2a. Register the optional callbacks - Backwards direction
    handler_backward = StepwiseHandler(flux=flux, output_dir="/Users/filipstrand/Desktop/backward", forward=False)
    CallbackRegistry.register_before_loop(handler_backward)
    CallbackRegistry.register_in_loop(handler_backward)
    # 2b. Register the optional callbacks - Forwards direction
    handler_forward = StepwiseHandler(flux=flux, output_dir="/Users/filipstrand/Desktop/forward", forward=True)
    CallbackRegistry.register_before_loop(handler_forward)
    CallbackRegistry.register_in_loop(handler_forward)

    try:
        # 1. Invert an existing image
        VCache.is_inverting = True
        inverted_latents = flux.invert(
            seed=seed,
            prompt=source_prompt,
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=source_guidance,
                init_image_path=image_path,
            ),
        )

        # 2. Generate a new image based on the inverted one
        VCache.is_inverting = False
        image = flux.generate_image(
            seed=seed,
            prompt=target_prompt,
            latents=inverted_latents,
            config=Config(
                num_inference_steps=steps,
                height=height,
                width=width,
                guidance=target_guidance,
            ),
        )

        # 3. Save the image
        image.save(path="edited.png")
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


if __name__ == "__main__":
    main()
