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


def get_lora_path(simple_name: str) -> str:
    if simple_name not in LORA_NAME_MAP:
        valid_names = ", ".join(sorted(LORA_NAME_MAP.keys()))
        raise ValueError(f"Unknown LoRA name: {simple_name}. Valid names are: {valid_names}")
    return f"{LORA_REPO_ID}:{LORA_NAME_MAP[simple_name]}"


def get_ic_edit_lora_path() -> str:
    return f"{IC_EDIT_LORA_REPO_ID}:{IC_EDIT_LORA_FILENAME}"
