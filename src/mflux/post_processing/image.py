import json
import logging
from pathlib import Path

import PIL.Image
import mlx.core as mx
import piexif

from mflux.config.model_config import ModelConfig

log = logging.getLogger(__name__)


class Image:

    def __init__(
            self,
            image: PIL.Image.Image,
            model_config: ModelConfig,
            seed: int,
            prompt: str,
            steps: int,
            guidance: float | None,
            precision: mx.Dtype,
            quantization: int,
            generation_time: float,
            lora_paths: list[str],
            lora_scales: list[float],
    ):
        self.image = image
        self.model_config = model_config
        self.seed = seed
        self.prompt = prompt
        self.steps = steps
        self.guidance = guidance
        self.precision = precision
        self.quantization = quantization
        self.generation_time = generation_time
        self.lora_paths = lora_paths
        self.lora_scales = lora_scales

    def save(self, path: str, export_json_metadata: bool = False) -> None:
        file_path = Path(path)
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
            self.image.save(file_path)
            log.info(f"Image saved successfully at: {file_path}")

            # Optionally save json metadata file
            if export_json_metadata:
                with open(f"{file_path.with_suffix('.json')}", 'w') as json_file:
                    json.dump(self._get_metadata(), json_file, indent=4)

            # Embed metadata
            self._embed_metadata(file_path)
            log.info(f"Metadata embedded successfully at: {file_path}")
        except Exception as e:
            log.error(f"Error saving image: {e}")

    def _embed_metadata(self, path: str) -> None:
        try:
            # Prepare metadata
            metadata = self._get_metadata()

            # Convert metadata dictionary to a string
            metadata_str = str(metadata)

            # Convert the string to bytes (using UTF-8 encoding)
            user_comment_bytes = metadata_str.encode('utf-8')

            # Define the UserComment tag ID
            USER_COMMENT_TAG_ID = 0x9286

            # Create an EXIF dictionary
            exif_dict = {
                '0th': {},
                'Exif': {
                    USER_COMMENT_TAG_ID: user_comment_bytes
                },
                'GPS': {},
                '1st': {},
                'thumbnail': None
            }

            # Create a piexif-compatible dictionary structure
            exif_piexif_dict = {
                'Exif': {
                    USER_COMMENT_TAG_ID: user_comment_bytes
                }
            }

            # Load the image and embed the EXIF data
            image = PIL.Image.open(path)
            exif_bytes = piexif.dump(exif_piexif_dict)
            image.info['exif'] = exif_bytes

            # Save the image with metadata
            image.save(path, exif=exif_bytes)

        except Exception as e:
            log.error(f"Error embedding metadata: {e}")

    def _get_metadata(self) -> dict:
        return {
            'model': str(self.model_config.alias),
            'seed': str(self.seed),
            'steps': str(self.steps),
            'guidance': "None" if self.model_config == ModelConfig.FLUX1_SCHNELL else str(self.guidance),
            'precision': f"{self.precision}",
            'quantization': "None" if self.quantization is None else f"{self.quantization} bit",
            'generation_time': f"{self.generation_time:.2f} seconds",
            'lora_paths': ', '.join(self.lora_paths) if self.lora_paths else '',
            'lora_scales': ', '.join([f"{scale:.2f}" for scale in self.lora_scales]) if self.lora_scales else '',
            'prompt': self.prompt,
        }
