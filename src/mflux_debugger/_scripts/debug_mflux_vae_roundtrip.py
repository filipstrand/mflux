from datetime import datetime
from pathlib import Path

from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.post_processing.image_util import ImageUtil
from mflux_debugger._scripts.debug_txt2img_config import TXT2IMG_DEBUG_CONFIG
from mflux_debugger.image_archive import archive_images
from mflux_debugger.image_tensor_paths import get_images_latest_framework_dir

OWL_PATH = Path("/Users/filipstrand/Desktop/owl.png")


def main():
    config = TXT2IMG_DEBUG_CONFIG

    model = FIBO(
        quantize=None,
        local_path=None,
    )

    fibo_model_config = ModelConfig(
        aliases=["fibo"],
        model_name="FIBO",
        base_model=None,
        controlnet_model=None,
        custom_transformer_model=None,
        num_train_steps=1000,
        max_sequence_length=512,
        supports_guidance=True,
        requires_sigma_shift=False,
        priority=1,
    )

    runtime_config = RuntimeConfig(
        config=Config(
            num_inference_steps=config.num_inference_steps,
            height=config.height,
            width=config.width,
            guidance=config.guidance,
            image_path=None,
            image_strength=None,
            scheduler=None,
        ),
        model_config=fibo_model_config,
    )

    if not OWL_PATH.exists():
        raise FileNotFoundError(f"Input image not found at {OWL_PATH}")

    owl_image = ImageUtil.load_image(str(OWL_PATH))
    owl_array = ImageUtil.to_array(owl_image, is_mask=False)

    latents = model.vae.encode(owl_array)
    decoded = model.vae.decode(latents)

    image = ImageUtil.to_image(
        decoded_latents=decoded,
        config=runtime_config,
        seed=config.seed,
        prompt="VAE roundtrip (MLX)",
        quantization=model.bits or 0,
        generation_time=0.0,
        lora_paths=[],
        lora_scales=[],
        image_path=None,
        image_paths=None,
        image_strength=None,
        controlnet_image_path=None,
        redux_image_paths=None,
        redux_image_strengths=None,
        masked_image_path=None,
        depth_image_path=None,
        concept_heatmap=None,
        negative_prompt=None,
    )

    archive_images("mlx")

    images_dir = get_images_latest_framework_dir("mlx")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = images_dir / f"debug_mflux_vae_roundtrip_{timestamp}.png"
    image.save(path=str(output_path), export_json_metadata=False)
    print(f"Saved MLX VAE roundtrip image: {output_path}")


if __name__ == "__main__":
    main()
