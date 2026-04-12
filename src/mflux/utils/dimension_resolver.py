from pathlib import Path

import PIL.Image

from mflux.cli.defaults import defaults as ui_defaults
from mflux.cli.defaults.defaults import DIMENSION_STEP_PIXELS
from mflux.utils.scale_factor import ScaleFactor


def _snap(value: float, step: int = DIMENSION_STEP_PIXELS) -> int:
    """Round down to the nearest multiple of *step* (default 16)."""
    return int(value - value % step)


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

        # --- Aspect-ratio-preserving scale propagation ---
        # When only one dimension carries a non-unity ScaleFactor and the other
        # is still at the "auto" default (ScaleFactor(1)), propagate the
        # explicit scale to both dimensions so the reference aspect ratio is
        # preserved.  E.g. --height 1.2x  →  both dimensions scale by 1.2×.
        if height_is_scale and width_is_scale:
            if height.value != 1.0 and width.value == 1.0:
                width = ScaleFactor(value=height.value)
            elif width.value != 1.0 and height.value == 1.0:
                height = ScaleFactor(value=width.value)

        # --- Absolute-pixel + auto: infer the other from aspect ratio ---
        # E.g. --height 800 (int) with width still at auto (ScaleFactor(1)):
        #   width = 800 × (orig_width / orig_height), snapped to 16 px.
        if not height_is_scale and width_is_scale and width.value == 1.0:
            aspect = orig_width / orig_height
            return _snap(int(height) * aspect), int(height)

        if height_is_scale and height.value == 1.0 and not width_is_scale:
            aspect = orig_height / orig_width
            return int(width), _snap(int(width) * aspect)

        # Standard per-dimension resolution
        resolved_height = height.get_scaled_value(orig_height) if isinstance(height, ScaleFactor) else int(height)
        resolved_width = width.get_scaled_value(orig_width) if isinstance(width, ScaleFactor) else int(width)

        return resolved_width, resolved_height
