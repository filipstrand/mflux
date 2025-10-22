import torch
from diffusers import FluxPipeline


def main():
    model_name = "black-forest-labs/FLUX.1-schnell"
    prompt = "A cat holding a sign that says hello world"
    seed = 42
    height = 256
    width = 256
    num_steps = 20
    guidance = 0.0

    pipe = FluxPipeline.from_pretrained(model_name, torch_dtype=torch.bfloat16)

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
        guidance_scale=guidance,
        generator=generator,
    ).images[0]

    output_path = "debug_diffusers_output.png"
    image.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
