from enum import Enum
from typing import NamedTuple


class QuantizationAction(Enum):
    NONE = "No quantization - use full precision"
    STORED = "Use quantization level stored in weights"
    REQUESTED = "Apply requested quantization on-the-fly"


class PathAction(Enum):
    LOCAL = "Use local filesystem path"
    HUGGINGFACE_CACHED = "Use cached HuggingFace files (no network)"
    HUGGINGFACE = "Download from HuggingFace"
    ERROR = "Path not found"


class LoraAction(Enum):
    LOCAL = "Use local filesystem path"
    REGISTRY = "Lookup in LORA_LIBRARY_PATH registry"
    HUGGINGFACE_COLLECTION_CACHED = "Use cached file from HuggingFace collection (no network)"
    HUGGINGFACE_COLLECTION = "Download specific file from HuggingFace collection"
    HUGGINGFACE_REPO_CACHED = "Use cached HuggingFace repository (no network)"
    HUGGINGFACE_REPO = "Download from HuggingFace repository"
    ERROR = "LoRA not found"


class ConfigAction(Enum):
    EXACT_MATCH = "Model name exactly matches a known alias"
    EXPLICIT_BASE = "Use explicitly provided --base-model"
    INFER_SUBSTRING = "Infer base model from substring match"
    ERROR = "Cannot determine model configuration"


class Rule(NamedTuple):
    priority: int
    name: str
    check: str
    action: QuantizationAction | PathAction | LoraAction | ConfigAction
