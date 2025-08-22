import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "qwen2_5_vl"
    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    out_hidden_size: int = 1536
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    tokens_per_second: int = 2
    temporal_patch_size: int = 2
    window_size: int = 112
    patch_size: int = 14
    fullatt_block_indexes: list[int] = field(default_factory=lambda: [7, 15, 23, 31])


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = 128000
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.rope_scaling:
            required_keys = {"mrope_section", "type"}
            if not all(key in self.rope_scaling for key in required_keys):
                raise ValueError(f"rope_scaling must contain keys {required_keys}")

            if not self.rope_scaling["type"] in ["mrope", "default"]:
                raise ValueError(f"rope_scaling type must be 'mrope' or 'default'")


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig
    vision_config: VisionConfig
    model_type: str
    ignore_index: int = -100
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None

    @classmethod
    def from_dict(cls, params):
        # Copy text config parameters from root level
        excluded_keys = {"vision_config"}
        params["text_config"] = dict(
            filter(lambda x: x[0] not in excluded_keys, params.items())
        )
        
        # Convert vision_config dict to VisionConfig object
        if "vision_config" in params and isinstance(params["vision_config"], dict):
            params["vision_config"] = VisionConfig.from_dict(params["vision_config"])
        
        # Convert text_config dict to TextConfig object
        if "text_config" in params and isinstance(params["text_config"], dict):
            params["text_config"] = TextConfig.from_dict(params["text_config"])

        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
