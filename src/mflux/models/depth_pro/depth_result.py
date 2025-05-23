from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from PIL import Image


@dataclass
class DepthResult:
    depth_image: Image.Image
    depth_array: mx.array
    min_depth: float
    max_depth: float

    def apply_transformation(self, transform_type=None, strength=1.0) -> "DepthResult":
        """Apply a non-linear transformation to a depth map.

        Args:
            transform_type: Type of transformation to apply ('foreground', 'background', 'sigmoid', 'log', None)
            strength: Strength of the transformation (higher values create more contrast)

        Returns:
            A new DepthResult object with the transformed depth image
        """
        if transform_type is None:
            return self

        # Get the depth array (normalized between 0-1)
        # Convert PIL image back to numpy array and normalize
        depth_np = np.array(self.depth_image) / 255.0

        if transform_type == "foreground":
            # Prioritize foreground (accentuate near objects)
            # Use power function: values closer to 0 (foreground) are stretched out
            transformed = depth_np ** (1 + strength)

        elif transform_type == "background":
            # Prioritize background (accentuate far objects)
            # Inverse power function: values closer to 1 (background) are stretched out
            transformed = 1 - (1 - depth_np) ** (1 + strength)

        elif transform_type == "sigmoid":
            # Sigmoid transformation - increases contrast in the middle range
            # Shift the values to be centered around 0.5
            centered = depth_np - 0.5
            transformed = 1 / (1 + np.exp(-centered * strength * 10))

        elif transform_type == "log":
            # Logarithmic transformation - increases detail in darker regions
            # Add small epsilon to avoid log(0)
            epsilon = 0.001
            transformed = np.log(depth_np + epsilon) / np.log(1 + epsilon)
            # Normalize back to 0-1 range
            transformed = (transformed - transformed.min()) / (transformed.max() - transformed.min())

        else:
            transformed = depth_np

        # Convert back to uint8 image
        transformed_image = Image.fromarray((transformed * 255).astype(np.uint8))

        # Return a new DepthResult object with the transformed image
        # Keep the original depth_array and depth values since they represent the raw data
        return DepthResult(
            depth_image=transformed_image,
            depth_array=self.depth_array,
            min_depth=self.min_depth,
            max_depth=self.max_depth,
        )
