from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget


def _reshape_mod6(w):
    # raw blocks.{b}.mod.lin is (6*hidden,) -> (6, hidden)
    return w.reshape(6, -1)


class Krea2WeightMapping(WeightMapping):
    """HF -> MLX module naming, supporting BOTH Krea 2 checkpoint conventions.

    The Krea 2 *Turbo* (distilled) checkpoint ships in the original Krea module naming
    (blocks.*, txtfusion.*, first/last/tmlp/tproj/txtmlp). The Krea 2 *Raw* (undistilled) checkpoint
    ships in diffusers naming (transformer_blocks.*, text_fusion.*, img_in/final_layer/time_embed/
    time_mod_proj/txt_in), which already matches the MLX module names 1:1.

    Each WeightTarget lists both source keys as `from_pattern` candidates; only the key that exists in
    the loaded checkpoint is matched (WeightMapper iterates over the actual weights), so the same
    mapping loads either checkpoint. The per-block modulation table needs a (6*hidden,)->(6,hidden)
    reshape ONLY for Turbo's `mod.lin`; Raw already stores it as `scale_shift_table` (6, hidden).
    """

    # ----- transformer -----
    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        t: List[WeightTarget] = []

        # img_in (turbo: first | raw: img_in)
        t.append(WeightTarget(to_pattern="img_in.weight", from_pattern=["first.weight", "img_in.weight"]))
        t.append(WeightTarget(to_pattern="img_in.bias", from_pattern=["first.bias", "img_in.bias"]))

        # time_embed (turbo: tmlp.0/tmlp.2 | raw: time_embed.linear_1/2)
        t.append(WeightTarget(to_pattern="time_embed.linear_1.weight", from_pattern=["tmlp.0.weight", "time_embed.linear_1.weight"]))  # noqa: E501
        t.append(WeightTarget(to_pattern="time_embed.linear_1.bias", from_pattern=["tmlp.0.bias", "time_embed.linear_1.bias"]))  # noqa: E501
        t.append(WeightTarget(to_pattern="time_embed.linear_2.weight", from_pattern=["tmlp.2.weight", "time_embed.linear_2.weight"]))  # noqa: E501
        t.append(WeightTarget(to_pattern="time_embed.linear_2.bias", from_pattern=["tmlp.2.bias", "time_embed.linear_2.bias"]))  # noqa: E501

        # time_mod_proj (turbo: tproj.1 | raw: time_mod_proj)
        t.append(WeightTarget(to_pattern="time_mod_proj.weight", from_pattern=["tproj.1.weight", "time_mod_proj.weight"]))  # noqa: E501
        t.append(WeightTarget(to_pattern="time_mod_proj.bias", from_pattern=["tproj.1.bias", "time_mod_proj.bias"]))

        # txt_in (turbo: txtmlp.0 norm / txtmlp.1 linear_1 / txtmlp.3 linear_2 | raw: txt_in.*)
        t.append(WeightTarget(to_pattern="txt_in.norm.weight", from_pattern=["txtmlp.0.scale", "txt_in.norm.weight"]))
        t.append(WeightTarget(to_pattern="txt_in.linear_1.weight", from_pattern=["txtmlp.1.weight", "txt_in.linear_1.weight"]))  # noqa: E501
        t.append(WeightTarget(to_pattern="txt_in.linear_1.bias", from_pattern=["txtmlp.1.bias", "txt_in.linear_1.bias"]))
        t.append(WeightTarget(to_pattern="txt_in.linear_2.weight", from_pattern=["txtmlp.3.weight", "txt_in.linear_2.weight"]))  # noqa: E501
        t.append(WeightTarget(to_pattern="txt_in.linear_2.bias", from_pattern=["txtmlp.3.bias", "txt_in.linear_2.bias"]))

        # text fusion projector (turbo: txtfusion.projector | raw: text_fusion.projector)
        t.append(WeightTarget(to_pattern="text_fusion.projector.weight", from_pattern=["txtfusion.projector.weight", "text_fusion.projector.weight"]))  # noqa: E501

        # text fusion blocks (layerwise + refiner), 2 each
        for kind in ("layerwise_blocks", "refiner_blocks"):
            t.extend(Krea2WeightMapping._fusion_block_targets(kind))

        # transformer blocks (28)
        t.extend(Krea2WeightMapping._transformer_block_targets())

        # final layer (turbo: last | raw: final_layer)
        t.append(WeightTarget(to_pattern="final_layer.norm.weight", from_pattern=["last.norm.scale", "final_layer.norm.weight"]))  # noqa: E501
        t.append(WeightTarget(to_pattern="final_layer.linear.weight", from_pattern=["last.linear.weight", "final_layer.linear.weight"]))  # noqa: E501
        t.append(WeightTarget(to_pattern="final_layer.linear.bias", from_pattern=["last.linear.bias", "final_layer.linear.bias"]))  # noqa: E501
        # Turbo stores the final modulation flat (needs no reshape: it is already (2, hidden) once
        # split); raw stores it directly as scale_shift_table.
        t.append(
            WeightTarget(
                to_pattern="final_layer.scale_shift_table",
                from_pattern=["last.modulation.lin", "final_layer.scale_shift_table"],
            )
        )
        # last.up / last.down are unused legacy weights in the distilled checkpoint -> not mapped.
        return t

    @staticmethod
    def _attn_targets(raw_prefix: str, mlx_prefix: str) -> List[WeightTarget]:
        # mlx_prefix doubles as the raw (diffusers) source prefix, since the MLX module names
        # mirror diffusers exactly.
        return [
            WeightTarget(to_pattern=f"{mlx_prefix}.to_q.weight", from_pattern=[f"{raw_prefix}.wq.weight", f"{mlx_prefix}.to_q.weight"]),  # noqa: E501
            WeightTarget(to_pattern=f"{mlx_prefix}.to_k.weight", from_pattern=[f"{raw_prefix}.wk.weight", f"{mlx_prefix}.to_k.weight"]),  # noqa: E501
            WeightTarget(to_pattern=f"{mlx_prefix}.to_v.weight", from_pattern=[f"{raw_prefix}.wv.weight", f"{mlx_prefix}.to_v.weight"]),  # noqa: E501
            WeightTarget(to_pattern=f"{mlx_prefix}.to_gate.weight", from_pattern=[f"{raw_prefix}.gate.weight", f"{mlx_prefix}.to_gate.weight"]),  # noqa: E501
            WeightTarget(to_pattern=f"{mlx_prefix}.to_out.0.weight", from_pattern=[f"{raw_prefix}.wo.weight", f"{mlx_prefix}.to_out.0.weight"]),  # noqa: E501
            WeightTarget(to_pattern=f"{mlx_prefix}.norm_q.weight", from_pattern=[f"{raw_prefix}.qknorm.qnorm.scale", f"{mlx_prefix}.norm_q.weight"]),  # noqa: E501
            WeightTarget(to_pattern=f"{mlx_prefix}.norm_k.weight", from_pattern=[f"{raw_prefix}.qknorm.knorm.scale", f"{mlx_prefix}.norm_k.weight"]),  # noqa: E501
        ]

    @staticmethod
    def _ff_targets(raw_prefix: str, mlx_prefix: str) -> List[WeightTarget]:
        return [
            WeightTarget(to_pattern=f"{mlx_prefix}.gate.weight", from_pattern=[f"{raw_prefix}.gate.weight", f"{mlx_prefix}.gate.weight"]),  # noqa: E501
            WeightTarget(to_pattern=f"{mlx_prefix}.up.weight", from_pattern=[f"{raw_prefix}.up.weight", f"{mlx_prefix}.up.weight"]),  # noqa: E501
            WeightTarget(to_pattern=f"{mlx_prefix}.down.weight", from_pattern=[f"{raw_prefix}.down.weight", f"{mlx_prefix}.down.weight"]),  # noqa: E501
        ]

    @staticmethod
    def _fusion_block_targets(kind: str, count: int = 2) -> List[WeightTarget]:
        targets: List[WeightTarget] = []
        for i in range(count):
            raw = f"txtfusion.{kind}.{i}"
            mlx = f"text_fusion.{kind}.{i}"
            targets.append(WeightTarget(to_pattern=f"{mlx}.norm1.weight", from_pattern=[f"{raw}.prenorm.scale", f"{mlx}.norm1.weight"]))  # noqa: E501
            targets.append(WeightTarget(to_pattern=f"{mlx}.norm2.weight", from_pattern=[f"{raw}.postnorm.scale", f"{mlx}.norm2.weight"]))  # noqa: E501
            targets.extend(Krea2WeightMapping._attn_targets(f"{raw}.attn", f"{mlx}.attn"))
            targets.extend(Krea2WeightMapping._ff_targets(f"{raw}.mlp", f"{mlx}.ff"))
        return targets

    @staticmethod
    def _transformer_block_targets(count: int = 28) -> List[WeightTarget]:
        targets: List[WeightTarget] = []
        for b in range(count):
            raw = f"blocks.{b}"
            mlx = f"transformer_blocks.{b}"
            targets.append(WeightTarget(to_pattern=f"{mlx}.norm1.weight", from_pattern=[f"{raw}.prenorm.scale", f"{mlx}.norm1.weight"]))  # noqa: E501
            targets.append(WeightTarget(to_pattern=f"{mlx}.norm2.weight", from_pattern=[f"{raw}.postnorm.scale", f"{mlx}.norm2.weight"]))  # noqa: E501
            # Turbo: blocks.{b}.mod.lin is flat (6*hidden,) -> reshape to (6, hidden). Raw already
            # stores scale_shift_table as (6, hidden) -> mapped 1:1 without a reshape transform.
            targets.append(
                WeightTarget(
                    to_pattern=f"{mlx}.scale_shift_table",
                    from_pattern=[f"{raw}.mod.lin"],
                    transform=_reshape_mod6,
                    required=False,
                )
            )
            targets.append(
                WeightTarget(
                    to_pattern=f"{mlx}.scale_shift_table",
                    from_pattern=[f"{mlx}.scale_shift_table"],
                    required=False,
                )
            )
            targets.extend(Krea2WeightMapping._attn_targets(f"{raw}.attn", f"{mlx}.attn"))
            targets.extend(Krea2WeightMapping._ff_targets(f"{raw}.mlp", f"{mlx}.ff"))
        return targets

    # ----- text encoder (Qwen3-VL language model only) -----
    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        return [
            WeightTarget(to_pattern="embed_tokens.weight", from_pattern=["language_model.embed_tokens.weight"]),
            WeightTarget(to_pattern="norm.weight", from_pattern=["language_model.norm.weight"]),
            WeightTarget(
                to_pattern="layers.{layer}.input_layernorm.weight",
                from_pattern=["language_model.layers.{layer}.input_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.post_attention_layernorm.weight",
                from_pattern=["language_model.layers.{layer}.post_attention_layernorm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.q_proj.weight",
                from_pattern=["language_model.layers.{layer}.self_attn.q_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.k_proj.weight",
                from_pattern=["language_model.layers.{layer}.self_attn.k_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.v_proj.weight",
                from_pattern=["language_model.layers.{layer}.self_attn.v_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.o_proj.weight",
                from_pattern=["language_model.layers.{layer}.self_attn.o_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.q_norm.weight",
                from_pattern=["language_model.layers.{layer}.self_attn.q_norm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.self_attn.k_norm.weight",
                from_pattern=["language_model.layers.{layer}.self_attn.k_norm.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.gate_proj.weight",
                from_pattern=["language_model.layers.{layer}.mlp.gate_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.up_proj.weight",
                from_pattern=["language_model.layers.{layer}.mlp.up_proj.weight"],
            ),
            WeightTarget(
                to_pattern="layers.{layer}.mlp.down_proj.weight",
                from_pattern=["language_model.layers.{layer}.mlp.down_proj.weight"],
            ),
        ]

    # ----- VAE: reuse qwen mapping -----
    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        from mflux.models.qwen.weights.qwen_weight_mapping import QwenWeightMapping

        return QwenWeightMapping.get_vae_mapping()
