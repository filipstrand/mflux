from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.common_models.qwen3_5 import QWEN3_5_9B_TEXT_CONFIG, Qwen35TextModel
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class MOSSImageWeightMapping:
    # Each layer is listed by its index, so a checkpoint with fewer layers than the model is
    # rejected at load time instead of leaving the rest at their random initial values.
    NUM_TRANSFORMER_LAYERS = 42
    NUM_TEXT_BLOCKS = 2

    _TEXT_BLOCK_SUFFIXES = [
        "norm1.weight",
        "norm2.weight",
        "q_norm.weight",
        "k_norm.weight",
        *(
            f"{name}.{param}"
            for name in ("qkv", "proj", "ffn_gate", "ffn_up", "ffn_down")
            for param in ("weight", "bias")
        ),
    ]
    _LAYER_SUFFIXES = [
        "norm1.weight",
        "norm2.weight",
        "q_norm.weight",
        "k_norm.weight",
        *(f"proj.{name}.weight" for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down")),
    ]
    _TRANSFORMER_TOP_LEVEL = [
        "video_embedder.weight",
        "video_embedder.bias",
        "text_embedder.proj.weight",
        "text_embedder.proj.bias",
        "time_embedding.0.weight",
        "time_embedding.0.bias",
        "time_embedding.2.weight",
        "time_embedding.2.bias",
        "shared_block_modulation.weight",
        "shared_block_modulation.bias",
        "video_head_modulation.weight",
        "video_head_modulation.bias",
        "final_norm_video.weight",
        "final_linear_video.weight",
    ]
    _MLP_SUFFIXES = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    _NORM_SUFFIXES = ["input_layernorm.weight", "post_attention_layernorm.weight"]
    _LINEAR_ATTENTION_SUFFIXES = [
        "linear_attn.A_log",
        "linear_attn.dt_bias",
        "linear_attn.conv1d.weight",
        "linear_attn.norm.weight",
        "linear_attn.in_proj_qkv.weight",
        "linear_attn.in_proj_z.weight",
        "linear_attn.in_proj_a.weight",
        "linear_attn.in_proj_b.weight",
        "linear_attn.out_proj.weight",
    ]
    _FULL_ATTENTION_SUFFIXES = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    ]

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        names = list(MOSSImageWeightMapping._TRANSFORMER_TOP_LEVEL)
        for block in range(MOSSImageWeightMapping.NUM_TEXT_BLOCKS):
            names += [f"text_embedder.blocks.{block}.{s}" for s in MOSSImageWeightMapping._TEXT_BLOCK_SUFFIXES]
        for layer in range(MOSSImageWeightMapping.NUM_TRANSFORMER_LAYERS):
            names += [f"layers.{layer}.{s}" for s in MOSSImageWeightMapping._LAYER_SUFFIXES]
        return [WeightTarget(to_pattern=name, from_pattern=[name]) for name in names]

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        names = ["embed_tokens.weight", "norm.weight"]
        layer_types = Qwen35TextModel.layer_types_for(
            QWEN3_5_9B_TEXT_CONFIG["num_hidden_layers"],
            QWEN3_5_9B_TEXT_CONFIG["full_attention_interval"],
        )
        for layer, layer_type in enumerate(layer_types):
            mixer = (
                MOSSImageWeightMapping._LINEAR_ATTENTION_SUFFIXES
                if layer_type == "linear_attention"
                else MOSSImageWeightMapping._FULL_ATTENTION_SUFFIXES
            )
            suffixes = [*mixer, *MOSSImageWeightMapping._MLP_SUFFIXES, *MOSSImageWeightMapping._NORM_SUFFIXES]
            names += [f"layers.{layer}.{s}" for s in suffixes]
        return [WeightTarget(to_pattern=name, from_pattern=[f"language_model.{name}"]) for name in names]

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        return Flux2WeightMapping.get_vae_mapping()
