import re
from dataclasses import dataclass
from typing import Union

from mflux.cli.defaults.defaults import DIMENSION_STEP_PIXELS

# Regex pattern for scale factors
SCALE_FACTOR_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)?)x$",  # Matches integer or decimal followed by 'x'
    re.IGNORECASE,
)


@dataclass
class ScaleFactor:
    value: Union[int, float]

    def __post_init__(self):
        if self.value <= 0:
            raise ValueError("Scale factor must be positive")

    def __str__(self):
        if isinstance(self.value, int) or self.value.is_integer():
            return f"{int(self.value)}x"
        return f"{self.value}x"

    def get_scaled_value(self, orig_value, pixel_steps=DIMENSION_STEP_PIXELS) -> int:
        return int(self.value * orig_value - (self.value * orig_value) % pixel_steps)

    @staticmethod
    def parse(text: str) -> "ScaleFactor":
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
