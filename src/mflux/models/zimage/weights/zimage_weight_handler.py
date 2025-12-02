"""
Z-Image weight handler for loading S3-DiT transformer, Qwen3 text encoder, and VAE weights.

Supports:
- Loading from HuggingFace Hub or local path
- Sharded safetensors files
- Pre-quantized weights (metadata detection)
- Runtime quantization (3/4/5/6/8-bit)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx.utils import tree_unflatten
from safetensors.torch import load_file as torch_load_file

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.flux.weights.weight_handler import MetaData
from mflux.models.zimage.weights.zimage_weight_mapping import ZImageWeightMapping
from mflux.utils.download import snapshot_download

if TYPE_CHECKING:
    from mflux.models.zimage.text_encoder import Qwen3Encoder, Qwen3Tokenizer
    from mflux.models.zimage.transformer import S3DiT


@dataclass
class ZImageComponents:
    """Container for all Z-Image model components."""

    transformer: "S3DiT"
    text_encoder: "Qwen3Encoder"
    tokenizer: "Qwen3Tokenizer"
    vae: nn.Module  # AutoencoderKL from FLUX


class ZImageWeightHandler:
    """Load Z-Image weights from HuggingFace or local path."""

    # HuggingFace repo mapping
    REPO_MAP = {
        "zimage-turbo": "Tongyi-MAI/Z-Image-Turbo",
    }

    # Architecture constants
    N_TRANSFORMER_BLOCKS = 30
    N_REFINER_LAYERS = 2
    N_TEXT_ENCODER_LAYERS = 36

    # Quantization support
    VALID_QUANTIZE_BITS = [3, 4, 5, 6, 8]
    DEFAULT_GROUP_SIZE = 64  # mflux standard

    # Expected memory sizes in GB
    EXPECTED_SIZES = {
        # (transformer_gb, text_encoder_gb, total_gb)
        None: (24.6, 8.0, 32.8),  # fp16
        16: (24.6, 8.0, 32.8),  # bf16/fp16
        8: (12.3, 4.0, 16.5),  # 8-bit
        6: (9.2, 3.0, 12.4),  # 6-bit
        4: (6.2, 2.0, 8.4),  # 4-bit
        3: (4.6, 1.5, 6.3),  # 3-bit
    }

    def __init__(
        self,
        meta_data: MetaData,
        transformer: dict | None = None,
        text_encoder: dict | None = None,
        vae: dict | None = None,
    ):
        self.meta_data = meta_data
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.vae = vae

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> "ZImageWeightHandler":
        """Load Z-Image weights from HuggingFace or local path.

        Args:
            repo_id: HuggingFace repo ID or alias ("zimage-turbo")
            local_path: Optional local path override

        Returns:
            ZImageWeightHandler with loaded weights
        """
        root_path: Path | None = None
        if local_path:
            root_path = Path(local_path)
        elif repo_id:
            # Resolve alias to repo ID
            actual_repo_id = ZImageWeightHandler.REPO_MAP.get(repo_id, repo_id)
            root_path = ZImageWeightHandler._download_weights(actual_repo_id)

        if root_path is None:
            raise ValueError("Either repo_id or local_path must be provided")

        # Try to load pre-saved mflux format first
        transformer_weights, quantization_level, mflux_version = ZImageWeightHandler._try_load_saved_component(
            root_path, "transformer"
        )
        text_encoder_weights, _, _ = ZImageWeightHandler._try_load_saved_component(root_path, "text_encoder")
        vae_weights, _, _ = ZImageWeightHandler._try_load_saved_component(root_path, "vae")

        # Fall back to HuggingFace format if not found
        if transformer_weights is None:
            transformer_weights = ZImageWeightHandler._load_transformer_weights(root_path)
            quantization_level = None
            mflux_version = None

        if text_encoder_weights is None:
            text_encoder_weights = ZImageWeightHandler._load_text_encoder_weights(root_path)

        if vae_weights is None:
            vae_weights = ZImageWeightHandler._load_vae_weights(root_path)

        return ZImageWeightHandler(
            meta_data=MetaData(
                quantization_level=quantization_level,
                scale=None,
                is_lora=False,
                mflux_version=mflux_version,
            ),
            transformer=transformer_weights,
            text_encoder=text_encoder_weights,
            vae=vae_weights,
        )

    @staticmethod
    def _download_weights(repo_id: str) -> Path:
        """Download weights from HuggingFace Hub."""
        return Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "transformer/*.safetensors",
                    "transformer/*.json",
                    "text_encoder/*.safetensors",
                    "text_encoder/*.json",
                    "tokenizer/*",
                    "vae/*.safetensors",
                    "vae/*.json",
                ],
            )
        )

    @staticmethod
    def _load_safetensors_shards(path: Path) -> dict[str, mx.array]:
        """Load all safetensor shards from a directory.

        Handles:
        - Sharded files (model-00001-of-00003.safetensors)
        - Single files (model.safetensors)
        - bfloat16 to float16 conversion
        """
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            # Use torch loader for bfloat16 support
            torch_weights = torch_load_file(str(shard))
            for key, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                all_weights[key] = mx.array(tensor.numpy())

        return all_weights

    @staticmethod
    def _load_transformer_weights(root_path: Path) -> dict:
        """Load and map transformer weights."""
        transformer_path = root_path / "transformer"
        if transformer_path.exists() and list(transformer_path.glob("*.safetensors")):
            raw_weights = ZImageWeightHandler._load_safetensors_shards(transformer_path)
        else:
            raw_weights = ZImageWeightHandler._load_safetensors_shards(root_path)

        mapping = ZImageWeightMapping.get_transformer_mapping()
        return WeightMapper.apply_mapping(
            raw_weights,
            mapping,
            num_blocks=ZImageWeightHandler.N_TRANSFORMER_BLOCKS,
        )

    @staticmethod
    def _load_text_encoder_weights(root_path: Path) -> dict:
        """Load and map text encoder weights."""
        text_encoder_path = root_path / "text_encoder"
        raw_weights = ZImageWeightHandler._load_safetensors_shards(text_encoder_path)

        mapping = ZImageWeightMapping.get_text_encoder_mapping()
        return WeightMapper.apply_mapping(
            raw_weights,
            mapping,
            num_layers=ZImageWeightHandler.N_TEXT_ENCODER_LAYERS,
        )

    @staticmethod
    def _load_vae_weights(root_path: Path) -> dict:
        """Load VAE weights.

        Z-Image uses the same VAE as FLUX. Weights need:
        1. Conv weight transposition from PyTorch OIHW to MLX OHWI format
        2. Restructuring to match FLUX VAE model structure
        """
        from mflux.config.config import Config

        vae_path = root_path / "vae"
        raw_weights = ZImageWeightHandler._load_safetensors_shards(vae_path)

        # Transpose conv weights from PyTorch OIHW to MLX OHWI and cast to precision
        processed_weights = []
        for key, value in raw_weights.items():
            if len(value.shape) == 4:
                value = value.transpose(0, 2, 3, 1)  # OIHW -> OHWI
            value = value.reshape(-1).reshape(value.shape).astype(Config.precision)
            processed_weights.append((key, value))

        weights = tree_unflatten(processed_weights)

        # Restructure to match FLUX VAE model structure
        weights["decoder"]["conv_in"] = {"conv2d": weights["decoder"]["conv_in"]}
        weights["decoder"]["conv_out"] = {"conv2d": weights["decoder"]["conv_out"]}
        weights["decoder"]["conv_norm_out"] = {"norm": weights["decoder"]["conv_norm_out"]}
        weights["encoder"]["conv_in"] = {"conv2d": weights["encoder"]["conv_in"]}
        weights["encoder"]["conv_out"] = {"conv2d": weights["encoder"]["conv_out"]}
        weights["encoder"]["conv_norm_out"] = {"norm": weights["encoder"]["conv_norm_out"]}

        return weights

    @staticmethod
    def _try_load_saved_component(
        root_path: Path,
        component_name: str,
    ) -> tuple[dict | None, int | None, str | None]:
        """Try to load a pre-saved mflux component.

        Returns:
            (weights, quantization_level, mflux_version) or (None, None, None)
        """
        component_path = root_path / component_name
        if not component_path.exists():
            return None, None, None

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

            if idx == 0 and len(data) > 1:
                quantization_level = data[1].get("quantization_level")
                mflux_version = data[1].get("mflux_version")

        if quantization_level is None and mflux_version is None:
            return None, None, None

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, mflux_version

    @classmethod
    def estimate_memory(cls, quantize: int | None = None) -> dict:
        """Estimate memory requirements for given quantization level.

        Args:
            quantize: Quantization bits (3, 4, 5, 6, 8) or None for fp16

        Returns:
            Dict with memory estimates in GB
        """
        trans, text, total = cls.EXPECTED_SIZES.get(quantize, cls.EXPECTED_SIZES[None])
        return {
            "transformer_gb": trans,
            "text_encoder_gb": text,
            "total_gb": total,
            "quantize": quantize or 16,
        }

    @classmethod
    def recommend_quantization(cls, available_memory_gb: float) -> int | None:
        """Recommend quantization level based on available memory.

        Args:
            available_memory_gb: Available GPU memory in GB

        Returns:
            Recommended quantization bits, or None for fp16
        """
        # Add 2GB headroom for inference
        required = available_memory_gb - 2.0

        for bits in [None, 8, 6, 4, 3]:
            _, _, total = cls.EXPECTED_SIZES[bits]
            if total <= required:
                return bits

        # If even 3-bit doesn't fit, return 3-bit anyway
        return 3

    def num_transformer_blocks(self) -> int:
        """Get number of transformer blocks from weights."""
        return self.N_TRANSFORMER_BLOCKS

    def num_refiner_layers(self) -> int:
        """Get number of context refiner layers from weights."""
        return self.N_REFINER_LAYERS

    def load_all(
        self,
        root_path: Path,
        quantize: int | None = None,
    ) -> ZImageComponents:
        """Create and load all Z-Image model components.

        Args:
            root_path: Path to model directory (for tokenizer)
            quantize: Optional quantization bits (3, 4, 5, 6, 8)

        Returns:
            ZImageComponents with all loaded models
        """
        # Import models here to avoid circular imports
        from mflux.models.flux.model.flux_vae.vae import VAE
        from mflux.models.zimage.text_encoder import Qwen3Encoder, Qwen3Tokenizer
        from mflux.models.zimage.transformer import S3DiT

        # Create model instances
        transformer = S3DiT()
        text_encoder = Qwen3Encoder()
        vae = VAE()
        # Qwen3Tokenizer expects root path - it uses subfolder="tokenizer" internally
        tokenizer = Qwen3Tokenizer(tokenizer_path=str(root_path))

        # Apply weights
        self._apply_weights(
            transformer=transformer,
            text_encoder=text_encoder,
            vae=vae,
            quantize=quantize,
        )

        return ZImageComponents(
            transformer=transformer,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            vae=vae,
        )

    def _apply_weights(
        self,
        transformer: nn.Module,
        text_encoder: nn.Module,
        vae: nn.Module,
        quantize: int | None = None,
    ) -> int | None:
        """Apply loaded weights to model instances.

        Args:
            transformer: S3DiT model instance
            text_encoder: Qwen3Encoder model instance
            vae: VAE model instance
            quantize: Optional quantization bits

        Returns:
            Actual quantization level used (from metadata or argument)
        """
        # Determine quantization level (metadata stores as string)
        q_level = self.meta_data.quantization_level
        if q_level is not None:
            q_level = int(q_level)
            # Pre-quantized weights - quantize models first, then load
            self._quantize_model(transformer, q_level)
            self._quantize_model(text_encoder, q_level)
            # VAE is not quantized - needs precision
            self._set_weights(transformer, text_encoder, vae)
            return q_level
        elif quantize is not None:
            # Runtime quantization - load weights, then quantize
            self._set_weights(transformer, text_encoder, vae)
            self._quantize_model(transformer, quantize)
            self._quantize_model(text_encoder, quantize)
            # VAE is not quantized - needs precision
            return quantize
        else:
            # No quantization - just load fp16 weights
            self._set_weights(transformer, text_encoder, vae)
            return None

    def _set_weights(
        self,
        transformer: nn.Module,
        text_encoder: nn.Module,
        vae: nn.Module,
    ) -> None:
        """Set weights on model instances."""
        if self.transformer is not None:
            transformer.update(self.transformer, strict=False)
        if self.text_encoder is not None:
            text_encoder.update(self.text_encoder, strict=False)
        if self.vae is not None:
            vae.update(self.vae, strict=False)

    def _quantize_model(self, model: nn.Module, bits: int) -> None:
        """Apply MLX quantization to a model.

        Args:
            model: nn.Module to quantize
            bits: Quantization bits (3, 4, 5, 6, 8)

        Raises:
            ValueError: If bits is not a valid quantization level
        """
        if bits not in self.VALID_QUANTIZE_BITS:
            raise ValueError(f"Invalid quantization bits: {bits}. Valid options: {self.VALID_QUANTIZE_BITS}")

        # MLX quantization with mflux standard group size
        nn.quantize(model, bits=bits, group_size=self.DEFAULT_GROUP_SIZE)

    @classmethod
    def load_model(
        cls,
        alias: str = "zimage-turbo",
        quantize: int | None = None,
        local_path: str | None = None,
    ) -> ZImageComponents:
        """Convenience method to load complete model.

        Args:
            alias: Model alias ("zimage-turbo") or HuggingFace repo ID
            quantize: Optional quantization bits (3, 4, 5, 6, 8)
            local_path: Optional local path override

        Returns:
            ZImageComponents with all loaded models
        """
        # Determine root path
        if local_path:
            root_path = Path(local_path)
        else:
            actual_repo_id = cls.REPO_MAP.get(alias, alias)
            root_path = cls._download_weights(actual_repo_id)

        # Load weights
        handler = cls.load_regular_weights(repo_id=alias, local_path=local_path)

        # Create and load models
        return handler.load_all(root_path, quantize=quantize)
