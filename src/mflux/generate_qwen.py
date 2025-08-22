from pathlib import Path
import time
from huggingface_hub import snapshot_download
import mlx.core as mx
from transformers import Qwen2Tokenizer

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.error.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.qwen.qwen_image import QwenImage
from mflux.qwen.utils import load_pt_tensor, torch_to_mx
from mflux.qwen.qwen2_5_vl.utils import load_model as load_qwen2_5_vl_model, process_text_embeddings_mlx

def main():
    # Hardcoded configuration values
    model_name = "qwen-image"
    quantize = 8
    local_path = None

    # Generation settings
    prompt = "Luxury food photograph"
    height = 512
    width = 512
    steps = 20
    guidance = 4.0
    seeds = [42]

    print("📖 Loading text embeddings...")
    root_path = Path(
            snapshot_download(
                repo_id="Qwen/Qwen-Image",
            )
        )
    
    qwen_text_encoder = load_qwen2_5_vl_model(root_path / "text_encoder")

    tokenizer =  tokenizer = Qwen2Tokenizer.from_pretrained(root_path / "tokenizer")
    print("✅ Model loaded successfully with diffusers extensions")
    
    
    # Format with diffusers template
    template = (
                "<|im_start|>system\n"
                "Describe the image by detailing the color, shape, size, texture, "
                "quantity, text, spatial relationships of the objects and background:"
                "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            )
    template_drop_idx = 34
    tokenizer_max_length = 1024
    formatted_prompt = template.format(prompt)
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        max_length=tokenizer_max_length + template_drop_idx,
        padding=True,
        truncation=True,
        return_tensors="mlx"
    )
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    print(f"\nInput shape: {input_ids.shape}")
    
    hidden_status = qwen_text_encoder(input_ids)
    # Get diffusers-compatible embeddings
    prompt_embeds, prompt_mask = process_text_embeddings_mlx(hidden_states=hidden_status, attention_mask= attention_mask, drop_idx= template_drop_idx, dtype= mx.bfloat16)

    # prompt_embeds = torch_to_mx(load_pt_tensor("debug_tensors/prompt_embeds.pt"), dtype=mx.bfloat16)
    # prompt_mask = torch_to_mx(load_pt_tensor("debug_tensors/prompt_embeds_mask.pt"), dtype=mx.float32)
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
                negative_prompt_embeds=None,
                prompt_mask=prompt_mask,
                negative_prompt_mask=None,
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
