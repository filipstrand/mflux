from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget

# Number of single-stream transformer blocks in the Krea 2 transformer.
NUM_TRANSFORMER_BLOCKS = 28


def _target(model_path: str) -> LoRATarget:
    """A LoRATarget for a single Krea 2 transformer Linear.

    `model_path` is the MLX module path (relative to the transformer), e.g.
    "transformer_blocks.{block}.attn.to_q" or "img_in". It may contain a "{block}" placeholder.

    We accept the file-side key under several conventions so a LoRA produced by either this fork
    (mflux training, prefix "transformer.") or external tooling loads cleanly:
      - transformer.<path>.lora_A/B.weight + .alpha   (mflux / diffusers-PEFT convention)
      - diffusion_model.<path>.lora_A/B.weight + .alpha (ai-toolkit / z-image convention)
      - <path>.lora_A/B.weight + .alpha                 (bare, no prefix)
      - kohya-style lora_up/lora_down with the same prefixes
    """
    paths = (
        f"transformer.{model_path}",
        f"diffusion_model.{model_path}",
        model_path,
    )
    up = []
    down = []
    alpha = []
    for p in paths:
        up.append(f"{p}.lora_B.weight")
        up.append(f"{p}.lora_up.weight")
        down.append(f"{p}.lora_A.weight")
        down.append(f"{p}.lora_down.weight")
        alpha.append(f"{p}.alpha")
    return LoRATarget(
        model_path=model_path,
        possible_up_patterns=up,
        possible_down_patterns=down,
        possible_alpha_patterns=alpha,
    )


class Krea2LoRAMapping(LoRAMapping):
    """LoRA target mapping for the Krea 2 transformer (shared by inference and training).

    Targets, per single-stream transformer block:
      - attention projections: attn.to_q, attn.to_k, attn.to_v, attn.to_gate, attn.to_out.0
      - SwiGLU feed-forward: ff.gate, ff.up, ff.down
    Plus the optional global Linears: img_in, txt_in.linear_1, txt_in.linear_2, time_mod_proj,
    final_layer.linear. Whatever a given LoRA file actually contains is applied; absent targets are
    simply not matched (no error), so a LoRA trained on the attention+ff subset still loads here.
    """

    _ATTN_PROJECTIONS = ("to_q", "to_k", "to_v", "to_gate", "to_out.0")
    _FF_PROJECTIONS = ("gate", "up", "down")
    _GLOBAL_PROJECTIONS = (
        "img_in",
        "txt_in.linear_1",
        "txt_in.linear_2",
        "time_mod_proj",
        "final_layer.linear",
    )

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets: list[LoRATarget] = []

        # Per-block attention + feed-forward.
        for proj in Krea2LoRAMapping._ATTN_PROJECTIONS:
            targets.append(_target(f"transformer_blocks.{{block}}.attn.{proj}"))
        for proj in Krea2LoRAMapping._FF_PROJECTIONS:
            targets.append(_target(f"transformer_blocks.{{block}}.ff.{proj}"))

        # Optional global Linears.
        for proj in Krea2LoRAMapping._GLOBAL_PROJECTIONS:
            targets.append(_target(proj))

        return targets
