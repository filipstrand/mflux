import mlx.core as mx
import torch
from transformers import Qwen3VLForConditionalGeneration

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.fibo.weights import FIBOWeightHandler
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_mapping import FIBOVLMWeightMapping
from mflux.models.flux.weights.weight_handler import MetaData


class FIBOVLMWeightHandler:
    @staticmethod
    def load_vlm_regular_weights(
        repo_id: str = "briaai/FIBO-vlm",
        local_path: str | None = None,
    ) -> "FIBOWeightHandler":
        # Load model - try offline first (use cache), fall back to online if needed
        pretrained_path = local_path or repo_id
        if local_path:
            # If explicit local path, use local_files_only
            model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_model_name_or_path=pretrained_path,
                dtype=torch.bfloat16,
                local_files_only=True,
            )
        else:
            # Try offline first (use cache), fall back to online if not cached
            try:
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=pretrained_path,
                    dtype=torch.bfloat16,
                    local_files_only=True,
                )
            except (FileNotFoundError, OSError):
                # Model not in cache, allow download if online
                model = Qwen3VLForConditionalGeneration.from_pretrained(
                    pretrained_model_name_or_path=pretrained_path,
                    dtype=torch.bfloat16,
                    local_files_only=False,
                )
        num_layers = model.config.text_config.num_hidden_layers
        depth = model.config.vision_config.depth
        state_dict = model.state_dict()

        raw_decoder_weights = {
            k: FIBOVLMWeightHandler._to_mlx(v)
            for k, v in state_dict.items()
            if k.startswith(("model.language_model", "lm_head"))
        }

        raw_visual_weights = {
            k: FIBOVLMWeightHandler._to_mlx(v) for k, v in state_dict.items() if k.startswith("model.visual")
        }

        decoder_weights = WeightMapper.apply_mapping(
            hf_weights=raw_decoder_weights,
            mapping=FIBOVLMWeightMapping.get_vlm_decoder_mapping(num_layers=num_layers),
            num_blocks=num_layers,
        )
        visual_weights = WeightMapper.apply_mapping(
            hf_weights=raw_visual_weights,
            mapping=FIBOVLMWeightMapping.get_vlm_visual_mapping(depth=depth),
            num_blocks=depth,
        )

        return FIBOWeightHandler(
            decoder=decoder_weights,
            visual=visual_weights,
            meta_data=MetaData(quantization_level=None, scale=None, is_lora=False, mflux_version=None),
        )

    @staticmethod
    def _to_mlx(tensor: torch.Tensor) -> mx.array:
        return mx.array((tensor.to(torch.float16) if tensor.dtype == torch.bfloat16 else tensor).detach().cpu().numpy())
