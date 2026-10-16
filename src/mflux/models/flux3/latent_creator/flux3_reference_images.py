import math

import numpy as np
import PIL.Image
from PIL import ImageOps

# Mirrors flux3.pipeline (reference_canvas, prepare_reference, prepare_multi_reference, default_size).
DOWNSAMPLING_RATIO = 16
MATCH_MAX_PIXELS = 4 * 1024 * 1024  # first reference of a single-reference edit, padded canvas included
MIN_SIDE, MAX_SIDE = 64, 4096
MAX_REFERENCES = 10
MULTI_REFERENCE_MAX_PIXELS = 1024 * 1024  # each reference of a multi-reference edit

Box = tuple[int, int, int, int]


class Flux3ReferenceImages:
    @staticmethod
    def prepare(
        images: list[PIL.Image.Image],
        crop_to_fit: bool = True,
    ) -> tuple[list[PIL.Image.Image], Box | None]:
        """Prepare 1-10 references. Returns the images and, for a single padded reference with `crop_to_fit`,
        the box of the fitted image inside the canvas (the output is cropped back to it)."""
        if not images:
            raise ValueError("At least one reference image is required")
        if len(images) > MAX_REFERENCES:
            raise ValueError(f"At most {MAX_REFERENCES} reference images are supported, got {len(images)}")
        if len(images) > 1:
            return [Flux3ReferenceImages.prepare_multi_reference(image) for image in images], None
        image, box = Flux3ReferenceImages.prepare_reference(images[0])
        return [image], (box if crop_to_fit else None)

    @staticmethod
    def default_size(references: list[PIL.Image.Image]) -> tuple[int, int]:
        return references[0].size

    @staticmethod
    def resolve_canvas(
        references: list[PIL.Image.Image], box: Box | None, width: int | None, height: int | None
    ) -> tuple[int, int, Box | None]:
        # Generate on the reference canvas and crop back to the fitted box when no size was given, or when the
        # size equals the box (what replaying the cropped size from a metadata sidecar provides).
        canvas_width, canvas_height = Flux3ReferenceImages.default_size(references)
        if box is not None and (not (width or height) or (width, height) == (box[2] - box[0], box[3] - box[1])):
            return canvas_width, canvas_height, box
        return width or canvas_width, height or canvas_height, None

    @staticmethod
    def to_array(image: PIL.Image.Image) -> np.ndarray:
        # (1, 3, H, W) float32 in [-1, 1]
        return (np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 127.5 - 1.0)[None]

    @staticmethod
    def reference_canvas(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int]]:
        scale = min(1.0, math.sqrt(MATCH_MAX_PIXELS / (width * height)), MAX_SIDE / max(width, height))
        while True:
            fit = (max(1, round(width * scale)), max(1, round(height * scale)))
            canvas = tuple(min(MAX_SIDE, max(MIN_SIDE, Flux3ReferenceImages._ceil16(v))) for v in fit)
            if canvas[0] * canvas[1] <= MATCH_MAX_PIXELS:
                return fit, canvas
            scale *= 0.995

    @staticmethod
    def prepare_reference(image: PIL.Image.Image) -> tuple[PIL.Image.Image, Box | None]:
        image = ImageOps.exif_transpose(image).convert("RGB")
        fit, (w, h) = Flux3ReferenceImages.reference_canvas(image.width, image.height)
        if fit != image.size:
            image = image.resize(fit, PIL.Image.Resampling.LANCZOS)
        if (w, h) == image.size:
            return image, None
        left, top = (w - image.width) // 2, (h - image.height) // 2
        padding = ((top, h - image.height - top), (left, w - image.width - left), (0, 0))
        # reflect needs a pad shorter than the side
        mode = "reflect" if min(image.size) > DOWNSAMPLING_RATIO else "symmetric"
        padded = PIL.Image.fromarray(np.pad(np.asarray(image), padding, mode=mode))
        return padded, (left, top, left + image.width, top + image.height)

    @staticmethod
    def prepare_multi_reference(image: PIL.Image.Image) -> PIL.Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        width, height = image.size
        if width * height > MULTI_REFERENCE_MAX_PIXELS:
            scale = math.sqrt(MULTI_REFERENCE_MAX_PIXELS / (width * height))
            width, height = int(width * scale), int(height * scale)
            image = image.resize((width, height), PIL.Image.Resampling.LANCZOS)
        w = width // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
        h = height // DOWNSAMPLING_RATIO * DOWNSAMPLING_RATIO
        if w == 0 or h == 0:
            raise ValueError(f"Reference image too small to condition on (needs >= {DOWNSAMPLING_RATIO}px per side)")
        left, top = (width - w) // 2, (height - h) // 2
        return image.crop((left, top, left + w, top + h))

    @staticmethod
    def _ceil16(value: int) -> int:
        return -(-value // DOWNSAMPLING_RATIO) * DOWNSAMPLING_RATIO
