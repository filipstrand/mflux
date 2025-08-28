import mlx.nn as nn

from mflux.config.config import Config
from mflux.models.qwen.model.qwen_transformer.qwen_image_transformer_applier import QwenImageTransformerApplier
from mflux.models.qwen.weights.qwen_text_encoder_loader import QwenTextEncoderLoader
from mflux.models.qwen.weights.qwen_weight_handler import QwenImageWeightHandler
from mflux.utils.quantization_util import QuantizationUtil


class QwenWeightUtil:
    @staticmethod
    def flatten(params):
        return [(k, v) for p in params for (k, v) in p]

    @staticmethod
    def reshape_weights(key, value):
        if len(value.shape) == 4:
            value = value.transpose(0, 2, 3, 1)
        value = value.reshape(-1).reshape(value.shape).astype(Config.precision)
        return [(key, value)]

    @staticmethod
    def set_weights_and_quantize(
        quantize_arg: int | None,
        weights: "QwenImageWeightHandler",
        vae: nn.Module,
        transformer: nn.Module,
        text_encoder: nn.Module | None = None,
    ) -> int | None:
        if weights.meta_data.quantization_level is None and quantize_arg is None:
            QwenWeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            return None

        if weights.meta_data.quantization_level is None and quantize_arg is not None:
            bits = quantize_arg
            QwenWeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            QuantizationUtil.quantize_qwen_models(text_encoder, vae, transformer, bits, weights)
            return bits

        if weights.meta_data.quantization_level is not None:
            bits = weights.meta_data.quantization_level
            QuantizationUtil.quantize_qwen_models(text_encoder, vae, transformer, bits, weights)
            QwenWeightUtil._set_model_weights(weights, vae, transformer, text_encoder)
            return bits

        raise Exception("Error setting weights")

    @staticmethod
    def _set_model_weights(
        weights: QwenImageWeightHandler,
        vae: nn.Module,
        transformer: nn.Module,
        text_encoder: nn.Module | None = None,
    ):
        vae.update(weights.vae, strict=False)
        
        # TARGETED FIX: Manually apply quant_conv bias that tree_unflatten missed
        # This ensures critical VAE weights are loaded correctly
        QwenWeightUtil._apply_critical_vae_weights(weights, vae)
        QwenImageTransformerApplier.apply_from_handler(module=transformer, weights=weights.transformer)
        nested_weights = QwenTextEncoderLoader.convert_to_nested_dict(weights.qwen_text_encoder)
        text_encoder.update(nested_weights, strict=False)
    
    @staticmethod
    def _apply_critical_vae_weights(weights: "QwenImageWeightHandler", vae: nn.Module):
        """
        Manually apply critical VAE weights that tree_unflatten might miss.
        
        This is a targeted fix for the quant_conv bias loading issue where
        manual assignment works but tree_unflatten doesn't apply the weights.
        
        Args:
            weights: QwenImageWeightHandler with raw weight data
            vae: The VAE module to apply weights to
        """
        try:
            # Access the raw diffusers weights directly
            if hasattr(weights, 'meta_data') and hasattr(weights.meta_data, 'vae_path'):
                from pathlib import Path
                from safetensors import safe_open
                import mlx.core as mx
                
                vae_file = Path(weights.meta_data.vae_path) / "diffusion_pytorch_model.safetensors"
                
                if vae_file.exists():
                    with safe_open(vae_file, framework="pt", device="cpu") as f:
                        # Apply quant_conv bias if it exists and is missing
                        if "quant_conv.bias" in f.keys():
                            correct_bias = f.get_tensor("quant_conv.bias")
                            correct_bias_mlx = mx.array(correct_bias.float().numpy())
                            
                            # Check if bias is missing (all zeros)
                            current_bias = vae.quant_conv.conv3d.bias
                            if mx.max(mx.abs(current_bias)) < 0.001:  # Bias is missing
                                vae.quant_conv.conv3d.bias = correct_bias_mlx
                                print("   🔧 Applied missing quant_conv bias")
                        
                        # Apply post_quant_conv bias if needed
                        if "post_quant_conv.bias" in f.keys():
                            correct_bias = f.get_tensor("post_quant_conv.bias")
                            correct_bias_mlx = mx.array(correct_bias.float().numpy())
                            
                            current_bias = vae.post_quant_conv.conv3d.bias
                            if mx.max(mx.abs(current_bias)) < 0.001:  # Bias is missing
                                vae.post_quant_conv.conv3d.bias = correct_bias_mlx
                                print("   🔧 Applied missing post_quant_conv bias")
                                
        except Exception as e:
            print(f"   ⚠️  Critical weight application failed: {e}")
            # Don't raise - this is a best-effort fix
