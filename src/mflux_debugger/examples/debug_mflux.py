from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage


def main():
    model_name = "qwen"
    prompt = "A cat holding a sign that says hello world"
    seed = 42
    height = 512
    width = 512
    num_steps = 20
    guidance = 3.5

    model = QwenImage(
        model_config=ModelConfig.from_name(model_name=model_name),
        quantize=None,
        local_path=None,
        lora_paths=None,
        lora_scales=None,
    )

    image = model.generate_image(
        seed=seed,
        prompt=prompt,
        negative_prompt=None,
        config=Config(
            num_inference_steps=num_steps,
            height=height,
            width=width,
            guidance=guidance,
            image_path=None,
            image_strength=None,
        ),
    )

    output_path = "debug_mflux_output.png"
    image.save(path=output_path, export_json_metadata=False)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
