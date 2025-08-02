from typing import Optional

import mlx.core as mx
from mlx import nn

from mflux.models.ip_adapter.ip_adapter import IPAdapter, MLPProjection


class FluxIPAdapter(nn.Module):
    """
    IP-Adapter specifically designed for FLUX models.
    Integrates image conditioning into the FLUX transformer architecture.
    """

    def __init__(
        self,
        image_encoder_hidden_size: int = 1024,
        cross_attention_dim: int = 4096,  # FLUX uses 4096 for T5 embeddings
        num_tokens: int = 4,
        scale: float = 1.0,
        use_mlp_projection: bool = True,
    ):
        super().__init__()

        self.scale = scale
        self.num_tokens = num_tokens
        self.cross_attention_dim = cross_attention_dim

        if use_mlp_projection:
            self.image_proj = MLPProjection(
                input_dim=image_encoder_hidden_size,
                hidden_dim=cross_attention_dim,
                output_dim=cross_attention_dim,
                num_tokens=num_tokens,
            )
        else:
            self.image_proj = IPAdapter(
                image_encoder_hidden_size=image_encoder_hidden_size,
                cross_attention_dim=cross_attention_dim,
                num_tokens=num_tokens,
                scale=scale,
            )

        # Additional layers for FLUX-specific integration
        self.norm = nn.LayerNorm(cross_attention_dim)

    def __call__(
        self,
        image_embeds: mx.array,
        encoder_hidden_states: Optional[mx.array] = None,
    ) -> mx.array:
        """
        Process image embeddings and optionally combine with text embeddings.

        Args:
            image_embeds: Image embeddings from CLIP [batch_size, hidden_size]
            encoder_hidden_states: Optional text embeddings to combine with

        Returns:
            Combined embeddings for conditioning
        """
        # Project image embeddings
        image_prompt_embeds = self.image_proj(image_embeds)
        image_prompt_embeds = self.norm(image_prompt_embeds)

        # Scale the contribution
        image_prompt_embeds = image_prompt_embeds * self.scale

        # Combine with text embeddings if provided
        if encoder_hidden_states is not None:
            # Concatenate image and text tokens
            combined_embeds = mx.concatenate([encoder_hidden_states, image_prompt_embeds], axis=1)
            return combined_embeds
        else:
            return image_prompt_embeds

    def set_scale(self, scale: float):
        """Dynamically adjust the IP-Adapter influence."""
        self.scale = scale


class FluxIPAdapterProcessor:
    """
    Processor for integrating IP-Adapter with FLUX attention layers.
    """

    def __init__(self, ip_adapter: FluxIPAdapter):
        self.ip_adapter = ip_adapter

    def __call__(
        self,
        attn_layer: nn.Module,
        hidden_states: mx.array,
        encoder_hidden_states: mx.array,
        image_embeds: Optional[mx.array] = None,
        **kwargs,
    ) -> mx.array:
        """
        Apply IP-Adapter conditioning to FLUX attention.
        """
        if image_embeds is not None:
            # Process image embeddings through IP-Adapter
            image_prompt_embeds = self.ip_adapter(image_embeds)

            # Combine with encoder hidden states
            enhanced_encoder_states = mx.concatenate([encoder_hidden_states, image_prompt_embeds], axis=1)
        else:
            enhanced_encoder_states = encoder_hidden_states

        # Apply the attention layer with enhanced conditioning
        return attn_layer(
            hidden_states=hidden_states,
            encoder_hidden_states=enhanced_encoder_states,
            **kwargs,
        )


def prepare_ip_adapter_image_embeds(
    ip_adapter: FluxIPAdapter,
    image_encoder,
    pil_image,
    device: str = "cpu",
    num_images_per_prompt: int = 1,
) -> mx.array:
    """
    Prepare image embeddings for IP-Adapter conditioning.

    Args:
        ip_adapter: The IP-Adapter model
        image_encoder: CLIP or similar image encoder
        pil_image: PIL Image or list of images
        device: Device to run on
        num_images_per_prompt: Number of images per prompt

    Returns:
        Processed image embeddings
    """
    if not isinstance(pil_image, list):
        pil_image = [pil_image]

    # Encode images
    image_embeds = []
    for img in pil_image:
        # This would typically use a CLIP image processor
        # For now, assume image is already preprocessed
        embed = image_encoder(img)
        image_embeds.append(embed)

    image_embeds = mx.stack(image_embeds, axis=0)

    # Duplicate for multiple images per prompt
    if num_images_per_prompt > 1:
        image_embeds = mx.repeat(image_embeds, num_images_per_prompt, axis=0)

    return image_embeds
