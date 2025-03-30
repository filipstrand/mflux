import mlx.core as mx
import mlx.nn as nn


class FOVHead(nn.Module):
    """Head for generating depth maps from features."""

    def __init__(self):
        super().__init__()
        # Default decoder dimension
        dim_decoder = 256

        # Create a sequential model for depth map generation
        self.layers = [
            # Reduce feature channels and apply spatial convolution
            nn.Conv2d(in_channels=dim_decoder, out_channels=dim_decoder // 2, kernel_size=3, stride=1, padding=1),
            # Upsample spatially with transposed convolution
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            # Further refine features
            nn.Conv2d(in_channels=dim_decoder // 2, out_channels=32, kernel_size=3, stride=1, padding=1),
            # Apply ReLU activation
            nn.ReLU(),
            # Final output convolution to generate depth map (1 channel)
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0),
            # Final ReLU to ensure positive depth values
            nn.ReLU(),
        ]

        # Initialize biases
        # Set the bias of the final conv layer to 0, which is a common practice
        # This will be initialized from weights later

    def __call__(self, features: mx.array) -> mx.array:
        """
        Generate depth map from input features

        Args:
            features: Input features from the decoder [B, C, H, W]

        Returns:
            Depth map [B, 1, H*2, W*2] (upsampled due to transpose conv)
        """
        # Process features through layers sequentially
        x = features
        for layer in self.layers:
            x = layer(x)

        return x
