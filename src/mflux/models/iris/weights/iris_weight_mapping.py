from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


class IrisWeightMapping(WeightMapping):
    DUAL_DEPTH = 8
    DEPTH = 24
    PIXEL_DEPTH = 4

    _DUAL_BLOCK = [
        "adaln_img.bias",
        "adaln_txt.bias",
        "attn.k_norm_x.weight",
        "attn.k_norm_y.weight",
        "attn.k_proj_x.weight",
        "attn.k_proj_y.weight",
        "attn.proj_x.bias",
        "attn.proj_x.weight",
        "attn.proj_y.bias",
        "attn.proj_y.weight",
        "attn.q_norm_x.weight",
        "attn.q_norm_y.weight",
        "attn.q_proj_x.weight",
        "attn.q_proj_y.weight",
        "attn.v_proj_x.weight",
        "attn.v_proj_y.weight",
        "attn_gate_x.weight",
        "attn_gate_y.weight",
        "attn_post_norm_x.weight",
        "attn_post_norm_y.weight",
        "mlp_post_norm_x.weight",
        "mlp_post_norm_y.weight",
        "mlp_x.w1.weight",
        "mlp_x.w2.weight",
        "mlp_x.w3.weight",
        "mlp_y.w1.weight",
        "mlp_y.w2.weight",
        "mlp_y.w3.weight",
        "norm_x1.weight",
        "norm_x2.weight",
        "norm_y1.weight",
        "norm_y2.weight",
    ]  # fmt: skip

    _SINGLE_BLOCK = [
        "adaln.bias",
        "attn_gate.weight",
        "attn_post_norm.weight",
        "attn_proj.bias",
        "attn_proj.weight",
        "k_norm.weight",
        "k_proj.weight",
        "mlp.w1.weight",
        "mlp.w2.weight",
        "mlp.w3.weight",
        "mlp_post_norm.weight",
        "norm1.weight",
        "norm2.weight",
        "q_norm.weight",
        "q_proj.weight",
        "v_proj.weight",
    ]  # fmt: skip

    _PIXEL_BLOCK = [
        "adaln.bias",
        "adaln.weight",
        "attn.k_norm.weight",
        "attn.proj.bias",
        "attn.proj.weight",
        "attn.q_norm.weight",
        "attn.qkv.weight",
        "compress.bias",
        "compress.weight",
        "expand.bias",
        "expand.weight",
        "mlp.fc1.bias",
        "mlp.fc1.weight",
        "mlp.fc2.bias",
        "mlp.fc2.weight",
        "norm1.weight",
        "norm2.weight",
    ]  # fmt: skip

    _GLOBAL = [
        "final_layer.linear.bias",
        "final_layer.linear.weight",
        "final_layer.norm.weight",
        "modulation_cores.adaln_img.bias",
        "modulation_cores.adaln_img.weight",
        "modulation_cores.adaln_txt.bias",
        "modulation_cores.adaln_txt.weight",
        "pixel_embedder.proj.bias",
        "pixel_embedder.proj.weight",
        "s_embedder.proj.bias",
        "s_embedder.proj.weight",
        "y_embedder.layer_pool.bias",
        "y_embedder.layer_pool.weight",
        "y_embedder.refiner.norm.weight",
        "y_embedder.refiner.proj.bias",
        "y_embedder.refiner.proj.weight",
        "y_pos_embedding",
    ]  # fmt: skip

    _LAYER_BLOCK = [
        "attn.proj.bias",
        "attn.proj.weight",
        "attn.qkv.weight",
        "norm1.weight",
        "norm2.weight",
    ]  # fmt: skip

    _REFINER_BLOCK = [
        "attn.k_norm.weight",
        "attn.proj.bias",
        "attn.proj.weight",
        "attn.q_norm.weight",
        "attn.qkv.weight",
        "mlp.w1.weight",
        "mlp.w2.weight",
        "mlp.w3.weight",
        "norm1.weight",
        "norm2.weight",
    ]  # fmt: skip

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        pairs = [(name, name) for name in IrisWeightMapping._GLOBAL]
        for part in ("weight", "bias"):
            pairs.append((f"t_embedder.linear_1.{part}", f"t_embedder.mlp.0.{part}"))
            pairs.append((f"t_embedder.linear_2.{part}", f"t_embedder.mlp.2.{part}"))

        for i in range(IrisWeightMapping.DEPTH):
            parts = (
                IrisWeightMapping._DUAL_BLOCK if i < IrisWeightMapping.DUAL_DEPTH else IrisWeightMapping._SINGLE_BLOCK
            )
            pairs += [(f"blocks.{i}.{part}", f"blocks.{i}.{part}") for part in parts]

        for i in range(IrisWeightMapping.PIXEL_DEPTH):
            pairs += [
                (f"pixel_blocks.{i}.{part}", f"pixel_blocks.{i}.{part}") for part in IrisWeightMapping._PIXEL_BLOCK
            ]

        for i in range(2):
            prefix = f"y_embedder.layer_blocks.{i}"
            pairs += [(f"{prefix}.{part}", f"{prefix}.{part}") for part in IrisWeightMapping._LAYER_BLOCK]
            for part in ("weight", "bias"):
                pairs.append((f"{prefix}.mlp.fc1.{part}", f"{prefix}.mlp.0.{part}"))
                pairs.append((f"{prefix}.mlp.fc2.{part}", f"{prefix}.mlp.2.{part}"))
            prefix = f"y_embedder.refiner.blocks.{i}"
            pairs += [(f"{prefix}.{part}", f"{prefix}.{part}") for part in IrisWeightMapping._REFINER_BLOCK]

        return [WeightTarget(to_pattern=to_name, from_pattern=[from_name]) for to_name, from_name in pairs]
