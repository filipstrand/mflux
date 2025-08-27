
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.error.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


def main():
    # Hardcoded configuration values
    model_name = "qwen-image"
    quantize = 6
    local_path = None

    # Generation settings
    prompt = "Luxury food photograph" + "Ultra HD, 4K, cinematic composition."
    negative_prompt = " "
    height = 512
    width = 512
    steps = 20
    guidance = 4.0
    seeds = [42]

    # 1. Load the model
    qwen = QwenImage(
        model_config=ModelConfig.from_name(model_name=model_name),
        quantize=quantize,
        local_path=local_path,
    )

    try:
        for seed in seeds:
            # 2. Generate an image for each seed value
            image = qwen.generate_image(
                seed=seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                config=Config(
                    num_inference_steps=steps,
                    height=height,
                    width=width,
                    guidance=guidance,
                ),
            )
            # 3. Save the image
            image.save(path="qwen_output_{seed}.png".format(seed=seed), export_json_metadata=False)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
