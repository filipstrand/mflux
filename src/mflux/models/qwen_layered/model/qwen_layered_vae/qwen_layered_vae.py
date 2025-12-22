import mlx.core as mx
from mlx import nn

from mflux.models.qwen.model.qwen_vae.qwen_image_causal_conv_3d import QwenImageCausalConv3D
from mflux.models.qwen_layered.model.qwen_layered_vae.qwen_layered_decoder_3d import QwenLayeredDecoder3D
from mflux.models.qwen_layered.model.qwen_layered_vae.qwen_layered_encoder_3d import QwenLayeredEncoder3D


class QwenLayeredVAE(nn.Module):
    """
    RGBA-VAE for Qwen-Image-Layered.
    
    Handles 4-channel RGBA images and supports multi-layer encoding/decoding
    where the temporal dimension is used as the layer dimension.
    
    - Encoder: RGBA [B, 4, N, H, W] -> Latent [B, 16, N, H/8, W/8]
    - Decoder: Latent [B, 16, N, H/8, W/8] -> RGBA [B, 4, N, H, W]
    """

    # Same latent normalization as base Qwen-Image
    LATENTS_MEAN = mx.array([-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]).reshape(1, 16, 1, 1, 1)  # fmt: off
    LATENTS_STD = mx.array([2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]).reshape(1, 16, 1, 1, 1)  # fmt: off

    def __init__(self, input_channels: int = 4, output_channels: int = 4):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.decoder = QwenLayeredDecoder3D(output_channels=output_channels)
        self.encoder = QwenLayeredEncoder3D(input_channels=input_channels)
        self.post_quant_conv = QwenImageCausalConv3D(16, 16, 1, 1, 0)
        self.quant_conv = QwenImageCausalConv3D(32, 32, 1, 1, 0)

    def decode(self, latents: mx.array, num_layers: int = 1) -> mx.array:
        """
        Decode latents to RGBA images.
        
        Args:
            latents: [B, 16, H, W] for single layer or [B, N, 16, H, W] for multi-layer
            num_layers: Number of output layers
            
        Returns:
            RGBA images [B, 4, H, W] for single layer or [B, N, 4, H, W] for multi-layer
        """
        # Handle both single-layer and multi-layer cases
        if latents.ndim == 4:
            # Single layer: [B, 16, H, W] -> [B, 16, 1, H, W]
            latents = latents.reshape(latents.shape[0], latents.shape[1], 1, latents.shape[2], latents.shape[3])
        elif latents.ndim == 5 and latents.shape[2] != num_layers:
            # Multi-layer packed: [B, N, 16, H, W] -> [B, 16, N, H, W]
            latents = mx.transpose(latents, (0, 2, 1, 3, 4))

        # Denormalize
        latents = latents * QwenLayeredVAE.LATENTS_STD + QwenLayeredVAE.LATENTS_MEAN
        latents = self.post_quant_conv(latents)
        decoded = self.decoder(latents)
        
        if num_layers == 1:
            # Single layer output: [B, 4, 1, H, W] -> [B, 4, H, W]
            return decoded[:, :, 0, :, :]
        else:
            # Multi-layer output: [B, 4, N, H, W] -> [B, N, 4, H, W]
            return mx.transpose(decoded, (0, 2, 1, 3, 4))

    def encode(self, images: mx.array) -> mx.array:
        """
        Encode RGBA images to latents.
        
        Args:
            images: RGBA images [B, 4, H, W] or [B, N, 4, H, W] for multi-layer
            
        Returns:
            Latents [B, 16, H/8, W/8] or [B, N, 16, H/8, W/8] for multi-layer
        """
        is_multi_layer = images.ndim == 5
        
        if images.ndim == 4:
            # Single image: [B, 4, H, W] -> [B, 4, 1, H, W]
            images = images.reshape(images.shape[0], images.shape[1], 1, images.shape[2], images.shape[3])
        else:
            # Multi-layer: [B, N, 4, H, W] -> [B, 4, N, H, W]
            images = mx.transpose(images, (0, 2, 1, 3, 4))

        latents = self.encoder(images)
        latents = self.quant_conv(latents)
        latents = latents[:, :16, :, :, :]
        
        # Normalize
        latents = (latents - QwenLayeredVAE.LATENTS_MEAN) / QwenLayeredVAE.LATENTS_STD
        
        if not is_multi_layer:
            # Single layer: [B, 16, 1, H, W] -> [B, 16, H, W]
            return latents[:, :, 0, :, :]
        else:
            # Multi-layer: [B, 16, N, H, W] -> [B, N, 16, H, W]
            return mx.transpose(latents, (0, 2, 1, 3, 4))

    def encode_condition_image(self, image: mx.array) -> mx.array:
        """
        Encode the input RGB condition image (converted to RGBA with alpha=1).
        
        Args:
            image: RGB image [B, 3, H, W]
            
        Returns:
            Latent [B, 16, H/8, W/8]
        """
        # Convert RGB to RGBA by adding alpha=1 channel
        batch_size = image.shape[0]
        height, width = image.shape[2], image.shape[3]
        alpha_channel = mx.ones((batch_size, 1, height, width), dtype=image.dtype)
        rgba_image = mx.concatenate([image, alpha_channel], axis=1)
        
        return self.encode(rgba_image)
