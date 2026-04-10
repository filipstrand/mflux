"""Weight loading utilities for Z-Image i2L models.

Handles downloading and transforming weights from three HuggingFace repositories:
  - DiffSynth-Studio/General-Image-Encoders (SigLIP2-G384 + DINOv3-7B)
  - DiffSynth-Studio/Z-Image-i2L (i2L decoder)
"""

import logging
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_unflatten

logger = logging.getLogger(__name__)

# HuggingFace repo and file paths
ENCODERS_REPO = "DiffSynth-Studio/General-Image-Encoders"
I2L_REPO = "DiffSynth-Studio/Z-Image-i2L"
SIGLIP2_FILE = "SigLIP2-G384/model.safetensors"
DINOV3_FILE = "DINOv3-7B/model.safetensors"
I2L_FILE = "model.safetensors"


def _download_weights(repo_id: str, filename: str) -> Path:
    """Download a single safetensors file from HuggingFace Hub."""
    logger.info(f"Downloading {filename} from {repo_id}...")
    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _transpose_conv2d(tensor: mx.array) -> mx.array:
    """Transpose Conv2d weight from PyTorch [out, in, kH, kW] to MLX [out, kH, kW, in]."""
    if len(tensor.shape) == 4:
        return tensor.transpose(0, 2, 3, 1)
    return tensor


def load_siglip2_weights(precision: mx.Dtype = mx.bfloat16) -> dict:
    """Load and transform SigLIP2-G384 weights for MLX.

    Handles:
    - Transpose all Linear weights from [out, in] to [in, out]
    - Transpose Conv2d weights from [out, in, kH, kW] to [out, kH, kW, in]
    - Split fused head.attention.in_proj_weight/bias into query/key/value projections
    """
    path = _download_weights(ENCODERS_REPO, SIGLIP2_FILE)
    raw = mx.load(str(path))

    transformed = {}
    for key, tensor in raw.items():
        tensor = tensor.astype(precision)

        # Handle the fused in_proj_weight/bias in the pooling head
        if key == "head.attention.in_proj_weight":
            # [4608, 1536] -> 3 x [1536, 1536]
            q, k, v = mx.split(tensor, 3, axis=0)
            transformed["head.query_proj.weight"] = q
            transformed["head.key_proj.weight"] = k
            transformed["head.value_proj.weight"] = v
            continue
        elif key == "head.attention.in_proj_bias":
            # [4608] -> 3 x [1536]
            q, k, v = mx.split(tensor, 3, axis=0)
            transformed["head.query_proj.bias"] = q
            transformed["head.key_proj.bias"] = k
            transformed["head.value_proj.bias"] = v
            continue
        elif key == "head.attention.out_proj.weight":
            transformed["head.out_proj.weight"] = tensor
            continue
        elif key == "head.attention.out_proj.bias":
            transformed["head.out_proj.bias"] = tensor
            continue

        # Transpose Conv2d weights only
        if key.endswith(".weight") and len(tensor.shape) == 4:
            tensor = _transpose_conv2d(tensor)

        transformed[key] = tensor

    return tree_unflatten(list(transformed.items()))


def load_dinov3_weights(precision: mx.Dtype = mx.bfloat16) -> dict:
    """Load and transform DINOv3-7B weights for MLX.

    Handles:
    - Transpose all Linear weights from [out, in] to [in, out]
    - Transpose Conv2d weights from [out, in, kH, kW] to [out, kH, kW, in]
    - Rename layer_scale{1,2}.lambda1 -> layer_scale{1,2}.gamma
    - Skip mask_token (not used in inference)
    """
    path = _download_weights(ENCODERS_REPO, DINOV3_FILE)
    raw = mx.load(str(path))

    transformed = {}
    for key, tensor in raw.items():
        tensor = tensor.astype(precision)

        # Skip mask_token — only used for training
        if "mask_token" in key:
            continue

        # Rename lambda1 -> gamma for LayerScale
        key = key.replace(".lambda1", ".gamma")

        # Transpose Conv2d weights only
        if key.endswith(".weight") and len(tensor.shape) == 4:
            tensor = _transpose_conv2d(tensor)

        transformed[key] = tensor

    return tree_unflatten(list(transformed.items()))


def load_i2l_decoder_weights(precision: mx.Dtype = mx.bfloat16) -> dict:
    """Load and transform Z-Image i2L decoder weights for MLX.

    All weights are Linear layers — transpose from [out, in] to [in, out].
    Weight names match the model structure 1:1.
    """
    path = _download_weights(I2L_REPO, I2L_FILE)
    raw = mx.load(str(path))

    transformed = {}
    for key, tensor in raw.items():
        transformed[key] = tensor.astype(precision)

    return tree_unflatten(list(transformed.items()))
