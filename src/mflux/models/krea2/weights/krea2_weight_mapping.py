from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


class Krea2WeightMapping(WeightMapping):
    @staticmethod
    def get_text_encoder_mapping(prefix: str = "language_model") -> List[WeightTarget]:
        return [
            WeightTarget("embed_tokens.weight", [f"{prefix}.embed_tokens.weight"]),
            WeightTarget(
                "layers.{block}.self_attn.q_proj.weight",
                [f"{prefix}.layers.{{block}}.self_attn.q_proj.weight"],
            ),
            WeightTarget(
                "layers.{block}.self_attn.k_proj.weight",
                [f"{prefix}.layers.{{block}}.self_attn.k_proj.weight"],
            ),
            WeightTarget(
                "layers.{block}.self_attn.v_proj.weight",
                [f"{prefix}.layers.{{block}}.self_attn.v_proj.weight"],
            ),
            WeightTarget(
                "layers.{block}.self_attn.o_proj.weight",
                [f"{prefix}.layers.{{block}}.self_attn.o_proj.weight"],
            ),
            WeightTarget(
                "layers.{block}.self_attn.q_norm.weight",
                [f"{prefix}.layers.{{block}}.self_attn.q_norm.weight"],
            ),
            WeightTarget(
                "layers.{block}.self_attn.k_norm.weight",
                [f"{prefix}.layers.{{block}}.self_attn.k_norm.weight"],
            ),
            WeightTarget("layers.{block}.mlp.gate_proj.weight", [f"{prefix}.layers.{{block}}.mlp.gate_proj.weight"]),
            WeightTarget("layers.{block}.mlp.up_proj.weight", [f"{prefix}.layers.{{block}}.mlp.up_proj.weight"]),
            WeightTarget("layers.{block}.mlp.down_proj.weight", [f"{prefix}.layers.{{block}}.mlp.down_proj.weight"]),
            WeightTarget(
                "layers.{block}.input_layernorm.weight",
                [f"{prefix}.layers.{{block}}.input_layernorm.weight"],
            ),
            WeightTarget(
                "layers.{block}.post_attention_layernorm.weight",
                [f"{prefix}.layers.{{block}}.post_attention_layernorm.weight"],
            ),
            WeightTarget("norm.weight", [f"{prefix}.norm.weight"]),
        ]
