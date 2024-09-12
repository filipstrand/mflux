import PIL
from PIL import Image
import mlx.core as mx
import numpy as np

from mflux.config.runtime_config import RuntimeConfig
from mflux.post_processing.image import GeneratedImage


class ImageUtil:

    @staticmethod
    def to_image(
            decoded_latents: mx.array,
            seed: int,
            prompt: str,
            quantization: int,
            generation_time: float,
            lora_paths: list[str],
            lora_scales: list[float],
            config: RuntimeConfig,
    ) -> GeneratedImage:
        normalized = ImageUtil._denormalize(decoded_latents)
        normalized_numpy = ImageUtil._to_numpy(normalized)
        image = ImageUtil._numpy_to_pil(normalized_numpy)
        return GeneratedImage(
            image=image,
            model_config=config.model_config,
            seed=seed,
            steps=config.num_inference_steps,
            prompt=prompt,
            guidance=config.guidance,
            precision=config.precision,
            quantization=quantization,
            generation_time=generation_time,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    @staticmethod
    def _denormalize(images: mx.array) -> mx.array:
        return mx.clip((images / 2 + 0.5), 0, 1)

    @staticmethod
    def _normalize(images: mx.array) -> mx.array:
        return 2.0 * images - 1.0

    @staticmethod
    def _to_numpy(images: mx.array) -> np.ndarray:
        images = mx.transpose(images, (0, 2, 3, 1))
        images = mx.array.astype(images, mx.float32)
        images = np.array(images)
        return images

    @staticmethod
    def _numpy_to_pil(images: np.ndarray) -> PIL.Image.Image:
        images = (images * 255).round().astype("uint8")
        pil_images = [PIL.Image.fromarray(image) for image in images]
        return pil_images[0]

    @staticmethod
    def _pil_to_numpy(image: PIL.Image.Image) -> np.ndarray:
        image = np.array(image).astype(np.float32) / 255.0
        images = np.stack([image], axis=0)
        return images

    @staticmethod
    def to_array(image: PIL.Image.Image) -> mx.array:
        image = ImageUtil._pil_to_numpy(image)
        array = mx.array(image)
        array = mx.transpose(array, (0, 3, 1, 2))
        array = ImageUtil._normalize(array)
        return array
