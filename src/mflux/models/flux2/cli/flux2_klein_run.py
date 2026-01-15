from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux2.variants.flux2_klein import Flux2Klein


def main() -> None:
    # Hardcoded values to mirror the diffusers flux_kelin.py example.
    prompt = "Luxury food photograph"
    height = 1024
    width = 1024
    guidance_scale = 1.0
    num_inference_steps = 4
    seed = 0

    model = Flux2Klein(
        model_config=ModelConfig.from_name("flux2-klein-4b"),
        quantize=None,
    )

    # Inline VAE decode debug (set latents_path to a saved diffusers latents file)
    latents_path = None  # e.g. "/path/to/flux2_klein_latents.npz"
    if latents_path:
        model.debug_decode_packed_latents(latents_path, output_path="flux2_klein_decode.png")
        return

    image_path = None  # e.g. "/path/to/input.png"
    if image_path:
        model.debug_roundtrip_image(image_path, output_path="flux2_klein_roundtrip.png")
        return

    # TODO: enable once Flux2Klein.generate_image is implemented.
    _ = (prompt, height, width, guidance_scale, num_inference_steps, seed)
    raise NotImplementedError("Flux2Klein inference will be wired after core ports.")


if __name__ == "__main__":
    main()
