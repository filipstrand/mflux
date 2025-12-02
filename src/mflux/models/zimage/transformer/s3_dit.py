import mlx.core as mx
import mlx.nn as nn

from mflux.models.zimage.embeddings import CaptionEmbed, PatchEmbed, RoPE3D, TimestepEmbed
from mflux.models.zimage.transformer.context_refiner import ContextRefinerBlock
from mflux.models.zimage.transformer.final_layer import FinalLayer
from mflux.models.zimage.transformer.transformer_block import S3DiTBlock


class S3DiT(nn.Module):
    """Scalable Single-Stream Diffusion Transformer.

    Full Z-Image transformer architecture:
    1. Embed inputs (patches, captions, timestep)
    2. Refine image patches with noise_refiner (WITH adaLN)
    3. Refine captions with context_refiner (WITHOUT adaLN)
    4. Concatenate refined captions + images into single stream
    5. 30 transformer blocks with AdaLN
    6. Extract and unpatchify image tokens
    """

    # Hardcoded architecture
    DIM = 3840
    N_LAYERS = 30
    N_REFINER_LAYERS = 2
    VAE_SCALE_FACTOR = 8  # VAE spatial downsampling factor (latent = pixel / 8)
    SEQ_MULTI_OF = 32  # Sequence padding multiple for caption alignment

    def __init__(self):
        super().__init__()

        # Input embeddings (from Phase 1)
        self.x_embedder = PatchEmbed()
        self.cap_embedder = CaptionEmbed()
        self.t_embedder = TimestepEmbed()

        # RoPE
        self.rope = RoPE3D()

        # Noise refiner (2 layers) - processes IMAGE patches WITH adaLN
        self.noise_refiner = [S3DiTBlock() for _ in range(self.N_REFINER_LAYERS)]

        # Context refiner (2 layers) - processes CAPTIONS WITHOUT adaLN
        self.context_refiner = [ContextRefinerBlock() for _ in range(self.N_REFINER_LAYERS)]

        # Main transformer blocks (30 layers)
        self.transformer_blocks = [S3DiTBlock() for _ in range(self.N_LAYERS)]

        # Output projection
        self.final_layer = FinalLayer()

        # Compiled forward implementation (lazy initialization)
        self._compiled_forward = None

        # Lazy cache for quantization check
        self._quantized_cached = None

    # Hardcoded patch size constant
    PATCH_SIZE = 2  # From PatchEmbed

    def _pad_to_multiple(self, length: int) -> int:
        """Pad length to next multiple of SEQ_MULTI_OF."""
        return length + ((-length) % self.SEQ_MULTI_OF)

    def precompute_rope(self, height: int, width: int, caption_len: int) -> tuple[mx.array, mx.array]:
        """Precompute RoPE frequencies for given dimensions.

        Call once before denoising loop, then pass result to __call__.

        Args:
            height: Latent height (pixels / VAE_SCALE_FACTOR)
            width: Latent width (pixels / VAE_SCALE_FACTOR)
            caption_len: Number of caption tokens

        Returns:
            Tuple of (freqs_cos, freqs_sin) for the combined image + caption sequence
        """
        h_patches = height // self.PATCH_SIZE
        w_patches = width // self.PATCH_SIZE

        # Compute padded caption length (matches _forward_impl logic)
        padded_cap_len = self._pad_to_multiple(caption_len)

        # Get image frequencies with time offset
        img_freqs_cos, img_freqs_sin = self.rope.get_image_freqs(h_patches, w_patches, time_offset=padded_cap_len + 1)

        # Get caption frequencies
        cap_freqs_cos, cap_freqs_sin = self.rope.get_caption_freqs(caption_len)

        # Concatenate: image first, then caption (matches S3DiT order)
        freqs_cos = mx.concatenate([img_freqs_cos, cap_freqs_cos], axis=0)
        freqs_sin = mx.concatenate([img_freqs_sin, cap_freqs_sin], axis=0)

        return freqs_cos, freqs_sin

    def precompute_timestep_embeddings(self, scheduler, num_steps: int) -> mx.array:
        """Precompute all timestep embeddings for the denoising loop.

        Args:
            scheduler: ZImageScheduler with timesteps
            num_steps: Number of inference steps

        Returns:
            Tensor of shape [num_steps, embed_dim] with all timestep embeddings
        """
        # Normalize timesteps as done in zimage.py
        all_t = mx.stack([(1000.0 - scheduler.timesteps[i]) / 1000.0 for i in range(num_steps)])
        return self.t_embedder(all_t)

    def __call__(
        self,
        latents: mx.array,  # [B, C, H, W] VAE latents
        text_embeddings: mx.array,  # [B, S, cap_feat_dim] from Qwen
        timestep: mx.array | None = None,  # [B] timestep values (optional if temb provided)
        rope: tuple[mx.array, mx.array] | None = None,  # Optional precomputed RoPE
        temb: mx.array | None = None,  # Optional precomputed timestep embedding
    ) -> mx.array:
        """Forward pass through S3-DiT with compilation.

        Note: Compilation is skipped for quantized models to avoid memory spike.
        mx.compile() traces through the function and captures quantized weights
        as dequantized fp16 constants, causing ~25GB memory increase.

        Args:
            latents: VAE latent representations [B, 16, H, W]
            text_embeddings: Caption embeddings from Qwen3 [B, S, 2560]
            timestep: Timestep values [B] (optional if temb provided)
            rope: Optional precomputed RoPE frequencies (freqs_cos, freqs_sin)
            temb: Optional precomputed timestep embedding [B, 256]

        Returns:
            Patch predictions [B, N_patches, 64]
        """
        # Skip compilation for quantized models to avoid memory spike
        # (compilation captures dequantized weights as constants)
        if self._is_quantized():
            return self._forward_impl(latents, text_embeddings, timestep, rope, temb)

        # Lazy compilation: compile on first call (fp16 models only)
        if self._compiled_forward is None:
            self._compiled_forward = mx.compile(self._forward_impl)

        return self._compiled_forward(latents, text_embeddings, timestep, rope, temb)

    def _is_quantized(self) -> bool:
        """Check if model contains quantized layers (cached after first check).

        Returns:
            True if any layer is quantized, False otherwise
        """
        if self._quantized_cached is None:
            self._quantized_cached = any(isinstance(m, nn.QuantizedLinear) for m in self.modules())
        return self._quantized_cached

    def _forward_impl(
        self,
        latents: mx.array,  # [B, C, H, W] VAE latents
        text_embeddings: mx.array,  # [B, S, cap_feat_dim] from Qwen
        timestep: mx.array | None = None,  # [B] timestep values (optional if temb provided)
        rope: tuple[mx.array, mx.array] | None = None,  # Optional precomputed RoPE
        temb: mx.array | None = None,  # Optional precomputed timestep embedding
    ) -> mx.array:
        """Internal forward implementation (compiled).

        Args:
            latents: VAE latent representations [B, 16, H, W]
            text_embeddings: Caption embeddings from Qwen3 [B, S, 2560]
            timestep: Timestep values [B] (optional if temb provided)
            rope: Optional precomputed RoPE frequencies (freqs_cos, freqs_sin)
            temb: Optional precomputed timestep embedding [B, 256]

        Returns:
            Patch predictions [B, N_patches, 64]
        """
        height, width = latents.shape[2], latents.shape[3]

        # 1. Embed inputs
        x = self.x_embedder(latents)  # [B, N_patches, dim]
        cap = self.cap_embedder(text_embeddings)  # [B, S, dim]

        # Use precomputed timestep embedding if provided, otherwise compute from timestep
        if temb is None:
            temb = self.t_embedder(timestep)  # [B, temb_dim=256]

        n_cap_tokens = cap.shape[1]
        n_img_tokens = x.shape[1]

        # Compute patch dimensions
        h_patches = height // self.x_embedder.PATCH_SIZE
        w_patches = width // self.x_embedder.PATCH_SIZE

        # 2. Get RoPE frequencies - use precomputed if available, otherwise compute
        if rope is not None:
            # Use precomputed RoPE frequencies (optimization for denoising loop)
            full_freqs_cos, full_freqs_sin = rope

            # Split into image and caption parts
            n_img_patches = h_patches * w_patches
            img_freqs_cos = full_freqs_cos[:n_img_patches]
            img_freqs_sin = full_freqs_sin[:n_img_patches]
            cap_freqs_cos = full_freqs_cos[n_img_patches:]
            cap_freqs_sin = full_freqs_sin[n_img_patches:]
        else:
            # Compute RoPE frequencies (backwards compatibility)
            # Diffusers pads caption length to multiple of 32 for position computation
            padded_cap_len = self._pad_to_multiple(n_cap_tokens)
            img_freqs_cos, img_freqs_sin = self.rope.get_image_freqs(
                h_patches, w_patches, time_offset=padded_cap_len + 1
            )
            cap_freqs_cos, cap_freqs_sin = self.rope.get_caption_freqs(n_cap_tokens)

            # Concatenate for full sequence
            full_freqs_cos = mx.concatenate([img_freqs_cos, cap_freqs_cos], axis=0)
            full_freqs_sin = mx.concatenate([img_freqs_sin, cap_freqs_sin], axis=0)

        img_rope = (img_freqs_cos, img_freqs_sin)
        cap_rope = (cap_freqs_cos, cap_freqs_sin)
        full_rope = (full_freqs_cos, full_freqs_sin)

        # 3. Refine image patches with noise_refiner (WITH adaLN)
        for refiner in self.noise_refiner:
            x = refiner(x, temb, rope=img_rope)

        # 4. Refine captions with context_refiner (WITHOUT adaLN, but WITH RoPE)
        for refiner in self.context_refiner:
            cap = refiner(cap, rope=cap_rope)

        # 5. SINGLE-STREAM: Concatenate images + captions (IMAGE FIRST per diffusers)
        hidden_states = mx.concatenate([x, cap], axis=1)

        # 6. Apply main transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb, rope=full_rope)

        # 7. Extract image tokens (at the beginning) and project to output
        img_tokens = hidden_states[:, :n_img_tokens, :]
        out = self.final_layer(img_tokens, temb)

        return out

    def unpatchify(self, x: mx.array, height: int, width: int) -> mx.array:
        """Convert patches back to spatial layout.

        Args:
            x: Patch tensor [B, N_patches, patch_size² × C]
            height: Latent height
            width: Latent width

        Returns:
            Spatial tensor [B, C, H, W]
        """
        patch_size = self.x_embedder.PATCH_SIZE
        in_channels = self.x_embedder.IN_CHANNELS

        h_patches = height // patch_size
        w_patches = width // patch_size

        x = x.reshape(x.shape[0], h_patches, w_patches, patch_size, patch_size, in_channels)
        x = x.transpose(0, 5, 1, 3, 2, 4)
        x = x.reshape(x.shape[0], in_channels, height, width)
        return x
