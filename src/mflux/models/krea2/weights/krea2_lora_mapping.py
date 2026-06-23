from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class Krea2LoRAMapping(LoRAMapping):
    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []
        targets.extend(Krea2LoRAMapping._get_global_targets())
        targets.extend(Krea2LoRAMapping._get_text_fusion_targets("layerwise_blocks"))
        targets.extend(Krea2LoRAMapping._get_text_fusion_targets("refiner_blocks"))
        targets.extend(Krea2LoRAMapping._get_transformer_block_targets())
        return targets

    @staticmethod
    def _get_global_targets() -> list[LoRATarget]:
        return [
            Krea2LoRAMapping._target("img_in", aliases=["first"]),
            Krea2LoRAMapping._target("time_embed.linear_1", aliases=["tmlp.0"]),
            Krea2LoRAMapping._target("time_embed.linear_2", aliases=["tmlp.2"]),
            Krea2LoRAMapping._target("time_mod_proj", aliases=["tproj.1"]),
            Krea2LoRAMapping._target("text_fusion.projector", aliases=["txtfusion.projector"]),
            Krea2LoRAMapping._target("txt_in.linear_1", aliases=["txtmlp.1"]),
            Krea2LoRAMapping._target("txt_in.linear_2", aliases=["txtmlp.3"]),
            Krea2LoRAMapping._target("final_layer.linear", aliases=["last.linear"]),
        ]

    @staticmethod
    def _get_text_fusion_targets(group: str) -> list[LoRATarget]:
        block = f"text_fusion.{group}.{{block}}"
        official = f"txtfusion.{group}.{{block}}"
        return [
            Krea2LoRAMapping._target(f"{block}.attn.to_q", aliases=[f"{official}.attn.wq"]),
            Krea2LoRAMapping._target(f"{block}.attn.to_k", aliases=[f"{official}.attn.wk"]),
            Krea2LoRAMapping._target(f"{block}.attn.to_v", aliases=[f"{official}.attn.wv"]),
            Krea2LoRAMapping._target(f"{block}.attn.to_gate", aliases=[f"{official}.attn.gate"]),
            Krea2LoRAMapping._target(
                f"{block}.attn.to_out.0",
                aliases=[f"{block}.attn.to_out", f"{official}.attn.wo"],
            ),
            Krea2LoRAMapping._target(f"{block}.ff.gate", aliases=[f"{official}.mlp.gate"]),
            Krea2LoRAMapping._target(f"{block}.ff.up", aliases=[f"{official}.mlp.up"]),
            Krea2LoRAMapping._target(f"{block}.ff.down", aliases=[f"{official}.mlp.down"]),
        ]

    @staticmethod
    def _get_transformer_block_targets() -> list[LoRATarget]:
        return [
            Krea2LoRAMapping._target(
                "transformer_blocks.{block}.attn.to_q",
                aliases=["blocks.{block}.attn.wq"],
            ),
            Krea2LoRAMapping._target(
                "transformer_blocks.{block}.attn.to_k",
                aliases=["blocks.{block}.attn.wk"],
            ),
            Krea2LoRAMapping._target(
                "transformer_blocks.{block}.attn.to_v",
                aliases=["blocks.{block}.attn.wv"],
            ),
            Krea2LoRAMapping._target(
                "transformer_blocks.{block}.attn.to_gate",
                aliases=["blocks.{block}.attn.gate"],
            ),
            Krea2LoRAMapping._target(
                "transformer_blocks.{block}.attn.to_out.0",
                aliases=["transformer_blocks.{block}.attn.to_out", "blocks.{block}.attn.wo"],
            ),
            Krea2LoRAMapping._target(
                "transformer_blocks.{block}.ff.gate",
                aliases=["blocks.{block}.mlp.gate"],
            ),
            Krea2LoRAMapping._target(
                "transformer_blocks.{block}.ff.up",
                aliases=["blocks.{block}.mlp.up"],
            ),
            Krea2LoRAMapping._target(
                "transformer_blocks.{block}.ff.down",
                aliases=["blocks.{block}.mlp.down"],
            ),
        ]

    @staticmethod
    def _target(model_path: str, aliases: list[str] | None = None) -> LoRATarget:
        module_paths = [model_path, *(aliases or [])]
        flat_paths = [Krea2LoRAMapping._flatten(path) for path in module_paths]
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=Krea2LoRAMapping._matrix_patterns(module_paths, flat_paths, "up"),
            possible_down_patterns=Krea2LoRAMapping._matrix_patterns(module_paths, flat_paths, "down"),
            possible_alpha_patterns=Krea2LoRAMapping._alpha_patterns(module_paths, flat_paths),
        )

    @staticmethod
    def _matrix_patterns(module_paths: list[str], flat_paths: list[str], direction: str) -> list[str]:
        if direction == "up":
            suffixes = [
                "lora_B.weight",
                "lora_B.default.weight",
                "lora_up.weight",
                "lora_up.default.weight",
                "lora.up.weight",
                "lora.up.default.weight",
            ]
        else:
            suffixes = [
                "lora_A.weight",
                "lora_A.default.weight",
                "lora_down.weight",
                "lora_down.default.weight",
                "lora.down.weight",
                "lora.down.default.weight",
            ]

        patterns = []
        for path in module_paths:
            for prefix in ("", "transformer.", "diffusion_model.", "base_model.model."):
                patterns.extend(f"{prefix}{path}.{suffix}" for suffix in suffixes)

        flat_suffix = "lora_up" if direction == "up" else "lora_down"
        for path in flat_paths:
            patterns.extend(
                [
                    f"lora_unet_{path}.{flat_suffix}.weight",
                    f"lora_unet_{path}.{flat_suffix}.default.weight",
                ]
            )

        return patterns

    @staticmethod
    def _alpha_patterns(module_paths: list[str], flat_paths: list[str]) -> list[str]:
        patterns = []
        for path in module_paths:
            patterns.extend(
                f"{prefix}{path}.alpha" for prefix in ("", "transformer.", "diffusion_model.", "base_model.model.")
            )
        patterns.extend(f"lora_unet_{path}.alpha" for path in flat_paths)
        return patterns

    @staticmethod
    def _flatten(path: str) -> str:
        return path.replace(".", "_")
