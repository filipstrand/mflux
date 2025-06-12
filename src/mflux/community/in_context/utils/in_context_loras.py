"""Configuration for In-Context LoRAs from Hugging Face."""

from mflux.weights.weight_handler_lora_huggingface import WeightHandlerLoRAHuggingFace

# Default Hugging Face repository for In-Context LoRAs
LORA_REPO_ID = "ali-vilab/In-Context-LoRA"

# Mapping from simple names to actual LoRA filenames
LORA_NAME_MAP = {
    "couple": "couple-profile.safetensors",
    "storyboard": "film-storyboard.safetensors",
    "font": "font-design.safetensors",
    "home": "home-decoration.safetensors",
    "illustration": "portrait-illustration.safetensors",
    "portrait": "portrait-photography.safetensors",
    "ppt": "ppt-templates.safetensors",
    "sandstorm": "sandstorm-visual-effect.safetensors",
    "sparklers": "sparklers-visual-effect.safetensors",
    "identity": "visual-identity-design.safetensors",
}

# IC-Edit specific configuration
IC_EDIT_LORA_REPO_ID = "RiverZ/normal-lora"
IC_EDIT_LORA_FILENAME = "pytorch_lora_weights.safetensors"
IC_EDIT_LORA_SCALE = 1.0


def get_lora_filename(simple_name: str) -> str:
    if simple_name not in LORA_NAME_MAP:
        valid_names = ", ".join(sorted(LORA_NAME_MAP.keys()))
        raise ValueError(f"Unknown LoRA name: {simple_name}. Valid names are: {valid_names}")
    return LORA_NAME_MAP[simple_name]


def prepare_ic_edit_loras(additional_lora_paths: list[str] | None = None) -> list[str]:
    print(f"🔍 Downloading IC-Edit LoRA from {IC_EDIT_LORA_REPO_ID}")

    # Download the required IC-Edit LoRA
    ic_edit_lora_path = WeightHandlerLoRAHuggingFace.download_lora(
        repo_id=IC_EDIT_LORA_REPO_ID,
        lora_name=IC_EDIT_LORA_FILENAME,
    )

    # IC-Edit LoRA is always required and goes first
    lora_paths = [ic_edit_lora_path]

    # Add any additional user-specified LoRAs
    if additional_lora_paths:
        lora_paths.extend(additional_lora_paths)

    print(f"✅ IC-Edit LoRA ready: {ic_edit_lora_path}")
    if len(lora_paths) > 1:
        print(f"📋 Additional LoRAs: {lora_paths[1:]}")

    return lora_paths
