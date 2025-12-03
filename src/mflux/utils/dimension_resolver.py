from pathlib import Path

import PIL.Image

from mflux.cli.defaults import defaults as ui_defaults
from mflux.utils.scale_factor import ScaleFactor


class DimensionResolver:
    @staticmethod
    def resolve(
        height: int | ScaleFactor,
        width: int | ScaleFactor,
        reference_image_path: Path | str | None = None,
    ) -> tuple[int, int]:
        height_is_scale = isinstance(height, ScaleFactor)
        width_is_scale = isinstance(width, ScaleFactor)

        # If neither dimension uses ScaleFactor, just return as-is
        if not height_is_scale and not width_is_scale:
            return int(width), int(height)

        # ScaleFactor requires a reference image - fall back to defaults if not provided
        if reference_image_path is None:
            resolved_width = ui_defaults.WIDTH if width_is_scale else int(width)
            resolved_height = ui_defaults.HEIGHT if height_is_scale else int(height)
            return resolved_width, resolved_height

        # Open image lazily - PIL.Image.open only reads metadata, not pixel data
        with PIL.Image.open(reference_image_path) as orig_image:
            orig_width, orig_height = orig_image.size

        # Resolve height
        if height_is_scale:
            resolved_height = height.get_scaled_value(orig_height)
        else:
            resolved_height = int(height)

        # Resolve width
        if width_is_scale:
            resolved_width = width.get_scaled_value(orig_width)
        else:
            resolved_width = int(width)

        return resolved_width, resolved_height
