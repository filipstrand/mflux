import json
import logging
import pathlib
import typing as t

import mlx.core as mx
import numpy as np
import piexif
import PIL.Image

from mflux.config.config import ConfigControlnet
from mflux.post_processing.generated_image import GeneratedImage

log = logging.getLogger(__name__)

RuntimeConfig: t.TypeAlias = "mflux.config.runtime_config.RuntimeConfig"  # noqa: F821


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
        controlnet_image_path: str | None = None,
        init_image_path: str | None = None,
        init_image_strength: float | None = None,
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
            init_image_path=init_image_path,
            init_image_strength=init_image_strength,
            controlnet_image_path=controlnet_image_path,
            controlnet_strength=config.controlnet_strength if isinstance(config.config, ConfigControlnet) else None,
        )

    @staticmethod
    def to_composite_image(generated_images: t.List[GeneratedImage]) -> PIL.Image.Image:
        # stitch horizontally
        total_width = sum(gen_img.image.width for gen_img in generated_images)
        max_height = max(gen_img.image.height for gen_img in generated_images)
        composite_img = PIL.Image.new("RGB", (total_width, max_height))
        current_x = 0
        for index, gen_img in enumerate(generated_images):
            composite_img.paste(gen_img.image, (current_x, 0))
            current_x += gen_img.image.width
        return composite_img

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

    @staticmethod
    def load_image(path: str | pathlib.Path) -> PIL.Image.Image:
        return PIL.Image.open(path)

    @staticmethod
    def scale_to_dimensions(
        image: PIL.Image.Image,
        target_width: int,
        target_height: int,
    ) -> PIL.Image.Image:
        if (image.width, image.height) != (target_width, target_height):
            return image.resize((target_width, target_height), PIL.Image.LANCZOS)
        else:
            return image

    @staticmethod
    def save_image(
            image: PIL.Image.Image,
            path: t.Union[str, pathlib.Path],
            metadata: dict | None = None,
            export_json_metadata: bool = False
    ) -> None:  # fmt: off
        file_path = pathlib.Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_name = file_path.stem
        file_extension = file_path.suffix

        # If a file already exists, create a new name with a counter
        counter = 1
        while file_path.exists():
            new_name = f"{file_name}_{counter}{file_extension}"
            file_path = file_path.with_name(new_name)
            counter += 1

        try:
            # Save image without metadata first
            image.save(file_path)
            log.info(f"Image saved successfully at: {file_path}")

            # Optionally save json metadata file
            if export_json_metadata:
                with open(f"{file_path.with_suffix('.json')}", "w") as json_file:
                    json.dump(metadata, json_file, indent=4)

            # Embed metadata
            if metadata is not None:
                ImageUtil._embed_metadata(metadata, file_path)
                log.info(f"Metadata embedded successfully at: {file_path}")
        except Exception as e:  # noqa: BLE001
            log.error(f"Error saving image: {e}")

    @staticmethod
    def _embed_metadata(metadata: dict, path: str) -> None:
        try:
            # Convert metadata dictionary to a string
            metadata_str = str(metadata)

            # Convert the string to bytes (using UTF-8 encoding)
            user_comment_bytes = metadata_str.encode("utf-8")

            # Define the UserComment tag ID
            USER_COMMENT_TAG_ID = 0x9286

            # Create a piexif-compatible dictionary structure
            exif_piexif_dict = {"Exif": {USER_COMMENT_TAG_ID: user_comment_bytes}}

            # Load the image and embed the EXIF data
            image = PIL.Image.open(path)
            exif_bytes = piexif.dump(exif_piexif_dict)
            image.info["exif"] = exif_bytes

            # Save the image with metadata
            image.save(path, exif=exif_bytes)

        except Exception as e:  # noqa: BLE001
            log.error(f"Error embedding metadata: {e}")
