import logging
from pathlib import Path

import PIL
import mlx.core as mx
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


class ImageUtil:

    @staticmethod
    def to_image(decoded_latents: mx.array) -> PIL.Image.Image:
        normalized = ImageUtil._denormalize(decoded_latents)
        normalized_numpy = ImageUtil._to_numpy(normalized)
        image = ImageUtil._numpy_to_pil(normalized_numpy)
        return image

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
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images[0]

    @staticmethod
    def _pil_to_numpy(image: PIL.Image.Image) -> np.ndarray:
        image = np.array(image).astype(np.float32) / 255.0
        images = np.stack([image], axis=0)
        return images

    @staticmethod
    def to_array(image: PIL.Image.Image) -> mx.array:
        image = ImageUtil.resize(image)
        image = ImageUtil._pil_to_numpy(image)
        array = mx.array(image)
        array = mx.transpose(array, (0, 3, 1, 2))
        array = ImageUtil._normalize(array)
        return array

    @staticmethod
    def resize(image):
        image = image.resize((1024, 1024), resample=PIL.Image.LANCZOS)
        return image

    @staticmethod
    def save_image(image: Image.Image, path: str) -> None:
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_name = file_path.stem
        file_extension = file_path.suffix

        # If a file already exists, create a new name with a counter
        counter = 1
        while file_path.exists():
            new_name = f"{file_name}({counter}){file_extension}"
            file_path = file_path.with_name(new_name)
            counter += 1

        try:
            image.save(file_path)
            log.info(f"Image saved successfully at: {file_path}")
        except Exception as e:
            log.info(f"Error saving image: {e}")
