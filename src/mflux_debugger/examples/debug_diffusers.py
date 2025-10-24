import torch
from diffusers import DiffusionPipeline


def main():
    model_name = "Qwen/Qwen-Image"
    prompt = "A cat holding a sign that says hello world"
    seed = 42
    height = 256
    width = 256
    num_steps = 20
    guidance = 3.5

    pipe = DiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    try:
        pipe = pipe.to("mps")
        device = "mps"
    except RuntimeError:
        pipe = pipe.to("cpu")
        device = "cpu"

    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_steps,
        true_cfg_scale=guidance,
        generator=generator,
    ).images[0]

    output_path = "debug_diffusers_output.png"
    image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
