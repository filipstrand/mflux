from pathlib import Path

import mlx.core as mx
import torch
from diffusers import BriaFiboPipeline
from mlx.utils import tree_unflatten

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.fibo.weights.fibo_weight_mapping import FIBOWeightMapping
from mflux.models.flux.weights.weight_handler import (
    MetaData,
    WeightHandler as FluxWeightHandler,
)


class FIBOWeightHandler:
    def __init__(
        self,
        meta_data: MetaData,
        vae: dict | None = None,
        transformer: dict | None = None,
    ):
        self.vae = vae
        self.transformer = transformer
        self.meta_data = meta_data

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> "FIBOWeightHandler":
        """
        Load FIBO weights, preferring pre-exported MLX safetensors when available.

        The loading order mirrors Flux/Qwen:
        1. If `local_path` (or downloaded cache) contains saved MLX weights with
           mflux metadata, load them directly via `mx.load` and `tree_unflatten`.
        2. Otherwise, fall back to converting from the PyTorch BriaFiboPipeline.
        """
        # 1) Resolve root path for any locally saved MLX weights
        root_path: Path | None = None
        if local_path:
            root_path = Path(local_path)
        elif repo_id:
            # Reuse the generic Flux weight download helper (HuggingFace snapshot)
            root_path = FluxWeightHandler.download_or_get_cached_weights(repo_id)

        vae_weights = None
        transformer_weights = None
        quantization_level: int | None = None
        mflux_version: str | None = None

        # 2) Try loading saved MLX weights (with mflux metadata) via mx.load
        if root_path is not None:
            (
                vae_weights,
                quantization_level,
                mflux_version,
            ) = FIBOWeightHandler._try_load_saved_component(root_path, "vae")
            (
                transformer_weights,
                quantization_level_t,
                mflux_version_t,
            ) = FIBOWeightHandler._try_load_saved_component(root_path, "transformer")

            # Prefer transformer metadata if present
            if quantization_level_t is not None:
                quantization_level = quantization_level_t
            if mflux_version_t is not None:
                mflux_version = mflux_version_t

        # 3) Fallback: convert from PyTorch BriaFiboPipeline if saved weights not found
        if vae_weights is None or transformer_weights is None:
            vae_weights = FIBOWeightHandler._load_vae_weights(repo_id, local_path)
            transformer_weights = FIBOWeightHandler._load_transformer_weights(repo_id, local_path)
            quantization_level = None
            mflux_version = None

        return FIBOWeightHandler(
            vae=vae_weights,
            transformer=transformer_weights,
            meta_data=MetaData(
                quantization_level=quantization_level,
                scale=None,
                is_lora=False,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def _load_vae_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        """
        Fallback path: load VAE weights from the PyTorch BriaFiboPipeline state_dict.

        This mirrors the declarative style used by Flux/Qwen:
        - We grab the flat `vae.state_dict()` from the diffusers pipeline.
        - We convert tensors to MLX arrays (handling bfloat16 -> float16).
        - All reshaping/transpose specifics live in `FIBOWeightMapping.get_vae_mapping()`.
        """
        pipe = FIBOWeightHandler._load_pipeline(repo_id, local_path)

        # Flat PyTorch state dict (keys like "encoder.conv_in.weight", "decoder.mid_block.resnets.0.conv1.weight", etc.)
        vae_state_dict = pipe.vae.state_dict()

        # Convert PyTorch tensors to MLX arrays in a generic way
        raw_weights: dict[str, mx.array] = {}
        for key, tensor in vae_state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            raw_weights[key] = mx.array(tensor.detach().cpu().numpy())

        # Apply declarative mapping (all structure/reshaping is handled in mapping)
        mapping = FIBOWeightMapping.get_vae_mapping()
        mapped_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_blocks=4)

        return mapped_weights

    @staticmethod
    def _load_transformer_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        """
        Load the PyTorch FIBO transformer weights and convert them to MLX arrays.

        For the transformer we mirror the diffusers module structure in MLX, so we can
        keep the state_dict keys as-is (no additional mapping required).
        """
        pipe = FIBOWeightHandler._load_pipeline(repo_id, local_path)

        transformer_state_dict = pipe.transformer.state_dict()

        raw_weights: dict[str, mx.array] = {}
        for key, tensor in transformer_state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)
            raw_weights[key] = mx.array(tensor.detach().cpu().numpy())

        # Apply declarative transformer mapping.
        # FIBO has 8 joint + 38 single blocks = 46 transformer layers.
        mapping = FIBOWeightMapping.get_transformer_mapping()
        mapped_weights = WeightMapper.apply_mapping(
            raw_weights,
            mapping,
            num_blocks=38,  # cover transformer_blocks (0-7) and single_transformer_blocks (0-37)
            num_layers=46,  # caption_projection layers
        )

        return mapped_weights

    @staticmethod
    def _load_pipeline(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> BriaFiboPipeline:
        if local_path:
            return BriaFiboPipeline.from_pretrained(local_path, torch_dtype=torch.bfloat16)
        return BriaFiboPipeline.from_pretrained(repo_id or "briaai/FIBO", torch_dtype=torch.bfloat16)

    # ------------------------------------------------------------------
    # Saved-weights helpers (mx.load-based, similar to Qwen/Flux)
    # ------------------------------------------------------------------
    @staticmethod
    def _try_load_saved_component(
        root_path: Path,
        component_name: str,
    ) -> tuple[dict | None, int | None, str | None]:
        """
        Try to load a saved MLX component (vae/transformer) from safetensors.

        Expects files under:
            root_path / component_name / *.safetensors

        Returns (weights_dict or None, quantization_level, mflux_version).
        """
        component_path = root_path / component_name
        if not component_path.exists():
            return None, None, None

        # Look for safetensors shards
        shard_files = sorted(f for f in component_path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            return None, None, None

        all_weights: dict[str, mx.array] = {}
        quantization_level: int | None = None
        mflux_version: str | None = None

        for idx, shard in enumerate(shard_files):
            data = mx.load(str(shard), return_metadata=True)
            weights_dict = data[0]
            all_weights.update(dict(weights_dict.items()))

            # Use metadata from the first shard (if present)
            if idx == 0 and len(data) > 1:
                quantization_level = data[1].get("quantization_level")
                mflux_version = data[1].get("mflux_version")

        # If there is no metadata, treat as non-saved (HF) weights and bail out
        if quantization_level is None and mflux_version is None:
            return None, None, None

        # Saved models: weights are already in MLX layout; just unflatten.
        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, mflux_version
