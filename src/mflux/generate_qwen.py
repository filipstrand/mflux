import time
import mlx.core as mx

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.error.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.qwen.qwen_image import QwenImage
from mflux.qwen.utils import load_pt_tensor, torch_to_mx


def main():
    # Hardcoded configuration values
    model_name = "qwen-image"
    quantize = 8
    local_path = None

    # Generation settings
    prompt = "Wont be used since we import text embeddings"
    height = 512
    width = 512
    steps = 20
    guidance = 4.0
    seeds = [42]

    print("📖 Loading text embeddings...")
    prompt_embeds = torch_to_mx(load_pt_tensor("debug_tensors/prompt_embeds.pt"), dtype=mx.bfloat16)
    prompt_mask = torch_to_mx(load_pt_tensor("debug_tensors/prompt_embeds_mask.pt"), dtype=mx.float32)
    negative_prompt_embeds = torch_to_mx(load_pt_tensor("debug_tensors/negative_prompt_embeds.pt"), dtype=mx.bfloat16)
    negative_prompt_mask = torch_to_mx(load_pt_tensor("debug_tensors/negative_prompt_embeds_mask.pt"), dtype=mx.float32)
    initial_latents = None

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
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_mask=prompt_mask,
                negative_prompt_mask=negative_prompt_mask,
                initial_latents=initial_latents,
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
