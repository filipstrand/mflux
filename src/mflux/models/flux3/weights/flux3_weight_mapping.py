from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightMapping, WeightTarget
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class Flux3WeightMapping(WeightMapping):
    # Module names follow the diffusers checkpoint, so every target is an identity except attn.to_out.0.
    _MODE_BLOCK_PARTS = [
        ("attn.to_q.weight", "attn.to_q.weight"),
        ("attn.to_k.weight", "attn.to_k.weight"),
        ("attn.to_v.weight", "attn.to_v.weight"),
        ("attn.to_out.weight", "attn.to_out.0.weight"),
        ("attn.norm_q.weight", "attn.norm_q.weight"),
        ("attn.norm_k.weight", "attn.norm_k.weight"),
        ("ff.linear_in.weight", "ff.linear_in.weight"),
        ("ff.linear_out.weight", "ff.linear_out.weight"),
    ]

    @staticmethod
    def get_transformer_mapping(flow_map: bool = True) -> List[WeightTarget]:
        names = [
            "context_embedder.weight",
            "timestep_embedder.linear_1.weight",
            "timestep_embedder.linear_2.weight",
            "norm_out.linear.weight",
            "proj_out.weight",
        ]
        target_time = ["target_timestep_embedder.linear_1.weight", "target_timestep_embedder.linear_2.weight"]
        if flow_map:
            names += target_time
        names += [f"x_embedders.{mode}.weight" for mode in ("image", "image_cond")]
        for mode in ("image", "image_cond", "txt"):
            names += [f"early_stream_modulations.{mode}.linear.weight", f"single_stream_modulations.{mode}.linear.weight"]  # fmt: skip
        targets = [WeightTarget(to_pattern=name, from_pattern=[name]) for name in names]
        if not flow_map:
            # The undistilled Base checkpoint has no target time embedding. Map it when present anyway, so that
            # Flux3Initializer can tell a Distilled checkpoint loaded as Base and stop with an error.
            targets += [WeightTarget(to_pattern=name, from_pattern=[name], required=False) for name in target_time]

        block_prefixes = ["mode_blocks.image", "mode_blocks.image_cond", "txt_mode_blocks"]
        for prefix in block_prefixes:
            targets += Flux3WeightMapping._block_targets(prefix, max_blocks=5)
        targets += Flux3WeightMapping._block_targets("single_transformer_blocks", max_blocks=28)
        return targets

    @staticmethod
    def get_base_transformer_mapping() -> List[WeightTarget]:
        return Flux3WeightMapping.get_transformer_mapping(flow_map=False)

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        # The FLUX 3 VAE is the FLUX.2 VAE (identical weights).
        return Flux2WeightMapping.get_vae_mapping()

    @staticmethod
    def _block_targets(prefix: str, max_blocks: int) -> List[WeightTarget]:
        return [
            WeightTarget(
                to_pattern=f"{prefix}.{{block}}.{to_part}",
                from_pattern=[f"{prefix}.{{block}}.{from_part}"],
                max_blocks=max_blocks,
            )
            for to_part, from_part in Flux3WeightMapping._MODE_BLOCK_PARTS
        ]
