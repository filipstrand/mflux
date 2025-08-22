
import glob
import json
from typing import Optional, Union
from huggingface_hub import snapshot_download
import mlx.core as mx
import mlx.nn as nn
import logging

from pathlib import Path

from transformers import AutoConfig
from .config import ModelConfig as Qwen2_5_vl_Config
from .qwen2_5_vl import Model as Qwen2_5_vl_Model
from .vision import VisionModel as Qwen2_5_vl_VisionModel
from .language import LanguageModel as Qwen2_5_vl_LanguageModel

def skip_multimodal_module(path: str) -> bool:
    """
    Check if a multimodal module (vision/audio) should skip quantization.

    Args:
        path: The module path to check

    Returns:
        bool: True if the module is multimodal and should skip quantization, False otherwise
    """
    return (
        "vision_model" in path
        or "vision_tower" in path
        or "audio_model" in path
        or "audio_tower" in path
    )


def update_module_configs(model_config, model_class, config, modules):
    """Updates configuration for model modules like text and vision modules.

    Args:
        model_config: The model configuration object that will be updated
        model_class: The model class containing component config classes
        config: Dictionary containing configuration parameters
        modules: List of module names to update configs for (e.g. ["text", "vision"])

    Returns:
        The updated model_config object
    """
    for config_name in modules:
        config_attr = f"{config_name}_config"
        if hasattr(model_config, config_attr):
            config_class = getattr(model_class, f"{config_name.title()}Config")
            setattr(
                model_config, config_attr, config_class.from_dict(config[config_attr])
            )
    return model_config

def get_model_path(
    path_or_hf_repo: str, revision: Optional[str] = None, force_download: bool = False
) -> Path:
    """
    Ensures the model is available locally. If the path does not exist locally,
    it is downloaded from the Hugging Face Hub.

    Args:
        path_or_hf_repo (str): The local path or Hugging Face repository ID of the model.
        revision (str, optional): A revision id which can be a branch name, a tag, or a commit hash.

    Returns:
        Path: The path to the model.
    """
    model_path = Path(path_or_hf_repo)
    if not model_path.exists():
        model_path = Path(
            snapshot_download(
                repo_id=path_or_hf_repo,
                revision=revision,
                allow_patterns=[
                    "*.json",
                    "*.safetensors",
                    "*.py",
                    "*.model",
                    "*.tiktoken",
                    "*.txt",
                    "*.jinja",
                ],
                force_download=force_download,
            )
        )
    return model_path


def load_config(model_path: Union[str, Path], **kwargs) -> dict:
    """Load model configuration from a path or Hugging Face repo.

    Args:
        model_path: Local path or Hugging Face repo ID to load config from
        **kwargs: Additional keyword arguments to pass to the config loader

    Returns:
        dict: Model configuration

    Raises:
        FileNotFoundError: If config.json is not found at the path
    """
    if isinstance(model_path, str):
        model_path = get_model_path(model_path)

    try:
        return AutoConfig.from_pretrained(model_path, **kwargs).to_dict()
    except ValueError:
        try:
            with open(model_path / "config.json", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Config not found at {model_path}") from exc


def load_model(model_path: Path, lazy: bool = False, **kwargs) -> nn.Module:
    """
    Load and initialize the model from a given path.

    Args:
        model_path (Path): The path to load the model from.
        lazy (bool): If False eval the model parameters to make sure they are
            loaded in memory before returning, otherwise they will be loaded
            when needed. Default: ``False``
        revision (str, optional): A revision id which can be a branch name,
            a tag, or a commit hash. Default: ``None``.

    Returns:
        nn.Module: The loaded and initialized model.

    Raises:
        FileNotFoundError: If the weight files (.safetensors) are not found.
        ValueError: If the model class or args class are not found or cannot be instantiated.
    """
    config = load_config(model_path, **kwargs)
    quantization = config.get("quantization", None)

    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        logging.error(f"No safetensors found in {model_path}")
        message = f"""
No safetensors found in {model_path}
Create safetensors using the following code:
```
from transformers import AutoModelForCausalLM, AutoProcessor

model_id= "<huggingface_model_id>"
model = AutoModelForCausalLM.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

model.save_pretrained("<local_dir>")
processor.save_pretrained("<local_dir>")
```
Then use the <local_dir> as the --hf-path in the convert script.
```
python -m mlx_vlm.convert --hf-path <local_dir> --mlx-path <mlx_dir>
```
        """
        raise FileNotFoundError(message)

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # Initialize text and vision configs if not present
    config.setdefault("text_config", {})
    config.setdefault("vision_config", {})
    config.setdefault("audio_config", {})

    # Initialize model config and update it with module configs
    model_config = Qwen2_5_vl_Config.from_dict(config)

    model = Qwen2_5_vl_Model(model_config)

    # Sanitize weights
    weights = sanitize_weights(model, weights)
    weights = sanitize_weights(
        Qwen2_5_vl_VisionModel, weights, model_config.vision_config
    )
    weights = sanitize_weights(
        Qwen2_5_vl_LanguageModel, weights, model_config.text_config
    )

    if (quantization := config.get("quantization", None)) is not None:
        # Handle legacy models which may or may not have vision quantized
        # TODO: Re-upload the models with the new quantization config and remove this
        skip_vision = config.get("vision_config", {}).get("skip_vision", False)

        def get_class_predicate(p, m):
            # Always skip vision and audio models
            if skip_multimodal_module(p) and skip_vision:
                return False
            # Handle custom per layer quantizations
            if p in config["quantization"]:
                return config["quantization"][p]
            if not hasattr(m, "to_quantized"):
                return False
            # Skip layers not divisible by 64
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            # Handle legacy models which may not have everything quantized
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            class_predicate=get_class_predicate,
        )

    model.load_weights(list(weights.items()))
    if not lazy:
        mx.eval(model.parameters())

    model.eval()
    return model


def sanitize_weights(model_obj, weights, config=None):
    """Helper function to sanitize weights if the model has a sanitize method"""
    if hasattr(model_obj, "sanitize"):
        if config is not None:
            model_obj = model_obj(config)
        weights = model_obj.sanitize(weights)
    return weights


def extract_masked_hidden(hidden_states, attention_mask):
    """
    MLX version of _extract_masked_hidden
    Extracts valid tokens for each sequence based on attention mask
    """
    batch_size = hidden_states.shape[0]
    split_hidden_states = []
    
    for i in range(batch_size):
        # Get mask for this sequence
        mask = attention_mask[i]
        # Find valid token positions (where mask == 1)
        valid_length = mx.sum(mask).item()
        valid_length = int(valid_length)
        # Extract only valid tokens
        valid_hidden = hidden_states[i, :valid_length, :]
        split_hidden_states.append(valid_hidden)
    
    return split_hidden_states


def process_text_embeddings_mlx(hidden_states, attention_mask, drop_idx=1, dtype=mx.float32):
    """
    MLX implementation of the QwenImage text processing logic
    
    Args:
        hidden_states: Encoder output hidden states [batch_size, seq_len, hidden_dim]
        attention_mask: Attention mask [batch_size, seq_len]
        drop_idx: Number of initial tokens to drop (default: 1)
        dtype: Target data type for output
    
    Returns:
        prompt_embeds: Padded embeddings [batch_size, max_seq_len, hidden_dim]
        encoder_attention_mask: Padded attention mask [batch_size, max_seq_len]
    """
    
    # Step 1: Extract masked hidden states (split by actual lengths)
    split_hidden_states = extract_masked_hidden(hidden_states, attention_mask)
    
    # Step 2: Drop initial tokens (e.g., special tokens)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    
    # Step 3: Create attention masks for valid tokens
    attn_mask_list = [mx.ones(e.shape[0], dtype=mx.int32) for e in split_hidden_states]
    
    # Step 4: Find maximum sequence length in batch
    max_seq_len = max([e.shape[0] for e in split_hidden_states])
    
    # Step 5: Pad prompt embeddings to max length
    padded_embeds = []
    for u in split_hidden_states:
        current_len = u.shape[0]
        hidden_dim = u.shape[1]
        if current_len < max_seq_len:
            # Create zero padding
            padding = mx.zeros((max_seq_len - current_len, hidden_dim), dtype=u.dtype)
            # Concatenate original embeddings with padding
            padded = mx.concatenate([u, padding], axis=0)
        else:
            padded = u
        padded_embeds.append(padded)
    
    # Stack into batch tensor
    prompt_embeds = mx.stack(padded_embeds, axis=0)
    
    # Step 6: Pad attention masks to max length
    padded_masks = []
    for mask in attn_mask_list:
        current_len = mask.shape[0]
        if current_len < max_seq_len:
            # Create zero padding for mask
            padding = mx.zeros(max_seq_len - current_len, dtype=mask.dtype)
            # Concatenate original mask with padding
            padded = mx.concatenate([mask, padding], axis=0)
        else:
            padded = mask
        padded_masks.append(padded)
    
    # Stack into batch tensor
    encoder_attention_mask = mx.stack(padded_masks, axis=0)
    
    # Step 7: Convert to target dtype
    prompt_embeds = prompt_embeds.astype(dtype)
    
    return prompt_embeds, encoder_attention_mask