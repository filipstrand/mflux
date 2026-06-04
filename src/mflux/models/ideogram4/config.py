from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Ideogram4Variant:
    name: str
    aliases: tuple[str, ...]
    repo_id: str
    local_dir_name: str


VARIANTS: dict[str, Ideogram4Variant] = {
    "ideogram-4-fp8": Ideogram4Variant(
        name="ideogram-4-fp8",
        aliases=(
            "ideogram-4-fp8",
            "ideogram4-fp8",
            "ideogram4",
            "ideogram-4",
            "ideogram",
            "ideogram-ai/ideogram-4-fp8",
        ),
        repo_id="ideogram-ai/ideogram-4-fp8",
        local_dir_name="ideogram-4-fp8",
    )
}

_ALIASES = {alias.lower(): variant for variant in VARIANTS.values() for alias in variant.aliases}

REQUIRED_FILES = (
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors",
    "unconditional_transformer/config.json",
    "unconditional_transformer/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
)

FP8_TEXT_ENCODER_CONFIG_KEY = "ideogram_fp8_weight_only"


def get_variant(name: str | Ideogram4Variant = "ideogram-4-fp8") -> Ideogram4Variant:
    if isinstance(name, Ideogram4Variant):
        return name
    key = name.strip().lower().rstrip("/")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        supported = ", ".join(sorted(_ALIASES))
        raise ValueError(f"Unknown Ideogram 4 variant {name!r}. Supported: {supported}") from exc


def is_ideogram4_alias(name: str | None) -> bool:
    if name is None:
        return False
    try:
        get_variant(name)
    except ValueError:
        return False
    return True


def variant_from_local_path(model_path: str | Path) -> Ideogram4Variant:
    root = Path(model_path).expanduser()
    model_index = _load_json(root / "model_index.json")
    if str(model_index.get("_class_name") or "").lower() == "ideogram4pipeline":
        _validate_fp8_layout(root)
        return VARIANTS["ideogram-4-fp8"]
    if "ideogram" in root.name.lower():
        _validate_fp8_layout(root)
        return VARIANTS["ideogram-4-fp8"]
    raise ValueError(f"Could not infer Ideogram 4 variant from local path: {root}")


def validate_model_layout(model_path: str | Path) -> Path:
    root = Path(model_path).expanduser()
    missing = [relative for relative in REQUIRED_FILES if not (root / relative).exists()]
    if missing:
        preview = ", ".join(missing[:4])
        if len(missing) > 4:
            preview += ", ..."
        raise FileNotFoundError(f"Missing Ideogram 4 model files under {root}: {preview}")
    model_index = _load_json(root / "model_index.json")
    class_name = str(model_index.get("_class_name") or "")
    if class_name and class_name != "Ideogram4Pipeline":
        raise ValueError(f"Expected Ideogram4Pipeline model_index, got {class_name!r}")
    _validate_fp8_layout(root)
    return root


def validate_dimensions(*, width: int, height: int) -> None:
    for label, value in (("width", width), ("height", height)):
        if value < 256 or value > 2048:
            raise ValueError(f"{label} must be in [256, 2048], got {value}")
        if value % 16:
            raise ValueError(f"{label} must be a multiple of 16, got {value}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
    except OSError as exc:
        raise ValueError(f"Could not read JSON file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc
    return value if isinstance(value, dict) else {}


def _validate_fp8_layout(root: Path) -> None:
    text_encoder_config = _load_json(root / "text_encoder" / "config.json")
    if text_encoder_config.get(FP8_TEXT_ENCODER_CONFIG_KEY) is True:
        return
    raise ValueError(
        "Ideogram 4 support currently requires the FP8 checkpoint layout "
        f"(expected text_encoder/config.json to contain "
        f"{FP8_TEXT_ENCODER_CONFIG_KEY!r}: true): {root}"
    )
