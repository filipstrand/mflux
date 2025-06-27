import re
from dataclasses import dataclass
from typing import Union

from mflux.ui.defaults import DIMENSION_STEP_PIXELS


@dataclass
class ScaleFactor:
    value: Union[int, float]

    def __post_init__(self):
        """Validate that the scale factor is positive"""
        if self.value <= 0:
            raise ValueError("Scale factor must be positive")

    def __str__(self):
        """String representation as multiplier"""
        if isinstance(self.value, int) or self.value.is_integer():
            return f"{int(self.value)}x"
        return f"{self.value}x"

    def get_scaled_value(self, orig_value, pixel_steps=DIMENSION_STEP_PIXELS) -> int:
        return int(self.value * orig_value - (self.value * orig_value) % pixel_steps)


# Regex pattern for scale factors
SCALE_FACTOR_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)?)x$",  # Matches integer or decimal followed by 'x'
    re.IGNORECASE,
)


def parse_scale_factor(text: str) -> ScaleFactor:
    """Parse a scale factor string into a ScaleFactor dataclass"""
    match = SCALE_FACTOR_PATTERN.match(text.strip())
    if not match:
        raise ValueError(f"Invalid scale factor format: '{text}'. Expected format: '2x', '1.5x', etc.")

    value_str = match.group(1)

    # Convert to int if it's a whole number, otherwise float
    if "." in value_str:
        value = float(value_str)
    else:
        value = int(value_str)

    return ScaleFactor(value)
