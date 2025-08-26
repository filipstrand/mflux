import time

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.error.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.qwen.qwen_image import QwenImage


def main():
    # Hardcoded configuration values
    model_name = "qwen-image"
    quantize = 6
    local_path = None

    # Generation settings
    prompt = "A coffee shop entrance features a chalkboard sign reading 'Qwen Coffee 😊 $2 per cup,' with a neon light beside it displaying '通义千问'. Next to it hangs a poster showing a beautiful Chinese woman, and beneath the poster is written 'π≈3.1415926-53589793-23846264-33832795-02384197'. Ultra HD, 4K, cinematic composition"
    negative_prompt = ""  # Empty negative prompt
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
            print(f"🎨 Generating {height}x{width} image with seed {seed}")
            start_time = time.time()
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
            generation_time = time.time() - start_time
            output_path = "qwen_output_{seed}.png".format(seed=seed)
            image.save(path=output_path, export_json_metadata=False)
            print(f"✅ Image saved to {output_path}")
            print(f"⏱️  Total generation time: {generation_time:.2f} seconds")
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)


if __name__ == "__main__":
    main()
