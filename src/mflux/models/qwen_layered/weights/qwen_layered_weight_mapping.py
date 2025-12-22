from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms
from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping


class QwenLayeredWeightMapping(WeightMapping):
    """
    Weight mapping for Qwen-Image-Layered model.
    
    Extends the base Qwen mapping with:
    - RGBA-VAE encoder/decoder (4-channel I/O)
    - Additional timestep embedding for layered model
    """

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        """
        Get transformer weight mapping.
        
        The layered model has an additional timestep embedding weight.
        Otherwise identical to base Qwen transformer.
        """
        # Start with base Qwen mapping
        mappings = QwenWeightMapping.get_transformer_mapping()
        
        # Add layered-specific mappings
        mappings.extend([
            # Additional timestep embedding for layers
            WeightTarget(
                to_pattern="time_text_embed.addition_t_embedding.weight",
                from_pattern=["time_text_embed.addition_t_embedding.weight"],
                required=False,
            ),
        ])
        
        return mappings

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        """
        Get VAE weight mapping for RGBA-VAE.
        
        Same structure as base Qwen VAE, but encoder/decoder handle 4 channels.
        The weight shapes for conv_in/conv_out will have different channel counts.
        """
        return QwenWeightMapping.get_vae_mapping()

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        """
        Get text encoder mapping.
        
        Identical to base Qwen - same Qwen2.5-VL encoder.
        """
        return QwenWeightMapping.get_text_encoder_mapping()
