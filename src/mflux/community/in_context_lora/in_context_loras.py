"""Configuration for In-Context LoRAs from Hugging Face."""

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


def get_lora_filename(simple_name: str) -> str:
    if simple_name not in LORA_NAME_MAP:
        valid_names = ", ".join(sorted(LORA_NAME_MAP.keys()))
        raise ValueError(f"Unknown LoRA name: {simple_name}. Valid names are: {valid_names}")
    return LORA_NAME_MAP[simple_name]
