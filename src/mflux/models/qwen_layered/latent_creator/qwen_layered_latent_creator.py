import mlx.core as mx


class QwenLayeredLatentCreator:
    """
    Latent creator for Qwen-Image-Layered.
    
    Handles multi-layer latent creation, packing, and unpacking.
    The layer dimension is treated as a temporal dimension in the 3D VAE.
    """

    @staticmethod
    def create_noise(seed: int, num_layers: int, height: int, width: int) -> mx.array:
        """
        Create initial noise for N output layers.
        
        Args:
            seed: Random seed
            num_layers: Number of output layers
            height: Image height
            width: Image width
            
        Returns:
            Noise tensor [1, N, 16, H/8, W/8]
        """
        mx.random.seed(seed)
        latent_height = height // 8
        latent_width = width // 8
        num_channels = 16
        
        # Create noise for all layers
        noise = mx.random.normal(
            shape=(1, num_layers, num_channels, latent_height, latent_width)
        )
        return noise

    @staticmethod
    def pack_latents(
        latents: mx.array,
        num_layers: int,
        height: int,
        width: int,
        patch_size: int = 2,
    ) -> mx.array:
        """
        Pack N-layer latents for transformer input.
        
        Args:
            latents: [B, N, 16, H, W] - multi-layer latents
            num_layers: Number of layers
            height: Image height  
            width: Image width
            patch_size: Patch size for patchification (default 2)
            
        Returns:
            Packed latents [B, N*H*W/(patch_size^2), C] for transformer
        """
        batch_size = latents.shape[0]
        num_channels = latents.shape[2]
        latent_height = height // 8
        latent_width = width // 8
        
        # [B, N, 16, H, W] -> [B, N, H, W, 16]
        latents = mx.transpose(latents, (0, 1, 3, 4, 2))
        
        # Patchify: [B, N, H, W, 16] -> [B, N, H/p, W/p, p*p*16]
        patched_height = latent_height // patch_size
        patched_width = latent_width // patch_size
        
        # Reshape for patchification
        latents = latents.reshape(
            batch_size,
            num_layers,
            patched_height,
            patch_size,
            patched_width,
            patch_size,
            num_channels,
        )
        # [B, N, H/p, p, W/p, p, C] -> [B, N, H/p, W/p, p, p, C]
        latents = mx.transpose(latents, (0, 1, 2, 4, 3, 5, 6))
        # [B, N, H/p, W/p, p*p*C]
        latents = latents.reshape(
            batch_size,
            num_layers,
            patched_height,
            patched_width,
            patch_size * patch_size * num_channels,
        )
        
        # Flatten spatial and layer dimensions: [B, N*H/p*W/p, C]
        latents = latents.reshape(
            batch_size,
            num_layers * patched_height * patched_width,
            patch_size * patch_size * num_channels,
        )
        
        return latents

    @staticmethod
    def unpack_latents(
        latents: mx.array,
        num_layers: int,
        height: int,
        width: int,
        patch_size: int = 2,
        out_channels: int = 16,
    ) -> mx.array:
        """
        Unpack transformer output to N-layer latents.
        
        Args:
            latents: [B, N*H*W/(patch_size^2), p*p*out_channels] - packed latents
            num_layers: Number of layers
            height: Image height
            width: Image width
            patch_size: Patch size used in packing
            out_channels: Number of output channels (default 16)
            
        Returns:
            Unpacked latents [B, N, out_channels, H/8, W/8]
        """
        batch_size = latents.shape[0]
        latent_height = height // 8
        latent_width = width // 8
        patched_height = latent_height // patch_size
        patched_width = latent_width // patch_size
        
        # [B, N*H/p*W/p, p*p*C] -> [B, N, H/p, W/p, p*p*C]
        latents = latents.reshape(
            batch_size,
            num_layers,
            patched_height,
            patched_width,
            patch_size * patch_size * out_channels,
        )
        
        # Unpatchify: [B, N, H/p, W/p, p, p, C]
        latents = latents.reshape(
            batch_size,
            num_layers,
            patched_height,
            patched_width,
            patch_size,
            patch_size,
            out_channels,
        )
        # [B, N, H/p, W/p, p, p, C] -> [B, N, H/p, p, W/p, p, C]
        latents = mx.transpose(latents, (0, 1, 2, 4, 3, 5, 6))
        # [B, N, H, W, C]
        latents = latents.reshape(
            batch_size,
            num_layers,
            latent_height,
            latent_width,
            out_channels,
        )
        # [B, N, C, H, W]
        latents = mx.transpose(latents, (0, 1, 4, 2, 3))
        
        return latents

    @staticmethod
    def pack_condition_image(
        latents: mx.array,
        height: int,
        width: int,
        patch_size: int = 2,
    ) -> mx.array:
        """
        Pack condition image latents (single layer) for transformer.
        
        Args:
            latents: [B, 16, H, W] - single layer latents
            height: Image height
            width: Image width
            patch_size: Patch size
            
        Returns:
            Packed latents [B, H*W/(patch_size^2), p*p*16]
        """
        batch_size = latents.shape[0]
        num_channels = latents.shape[1]
        latent_height = height // 8
        latent_width = width // 8
        patched_height = latent_height // patch_size
        patched_width = latent_width // patch_size
        
        # [B, 16, H, W] -> [B, H, W, 16]
        latents = mx.transpose(latents, (0, 2, 3, 1))
        
        # Patchify
        latents = latents.reshape(
            batch_size,
            patched_height,
            patch_size,
            patched_width,
            patch_size,
            num_channels,
        )
        latents = mx.transpose(latents, (0, 1, 3, 2, 4, 5))
        latents = latents.reshape(
            batch_size,
            patched_height * patched_width,
            patch_size * patch_size * num_channels,
        )
        
        return latents
