"""
Qwen-Image DreamBooth Training Initializer.

Handles initialization of all training components:
- Model loading with LoRA/DoRA layers
- Dataset preparation with embedding caching
- Training state creation
"""

from pathlib import Path
from typing import Any

import mlx.core as mx

from mflux.models.common.config.config import Config
from mflux.models.common.lora.layer.adapter_factory import AdapterFactory, AdapterType
from mflux.models.qwen.utils.model_resolution import resolve_qwen_model_spec
from mflux.models.qwen.variants.training.dataset.qwen_dataset import (
    QwenDataset,
    QwenDatasetFromFolder,
    QwenExampleSpec,
)
from mflux.models.qwen.variants.training.optimization.embedding_cache import EmbeddingCache
from mflux.models.qwen.variants.training.state.qwen_training_spec import QwenTrainingSpec
from mflux.models.qwen.variants.training.state.qwen_training_state import QwenTrainingState


class QwenDreamBoothInitializer:
    """
    Initialize Qwen-Image training components.

    Handles:
    - Loading QwenImage model
    - Applying LoRA/DoRA layers to transformer
    - Preparing dataset with embedding caching
    - Creating training state
    """

    @staticmethod
    def initialize(
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        adapter_type: str = "lora",
    ) -> tuple[Any, Config, QwenTrainingSpec, QwenTrainingState]:
        """
        Initialize all training components.

        Args:
            config_path: Path to training config JSON
            checkpoint_path: Path to checkpoint to resume from
            adapter_type: Type of adapter ("lora" or "dora")

        Returns:
            Tuple of (qwen_model, config, training_spec, training_state)
        """
        # Load training specification
        spec = QwenTrainingSpec.resolve(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

        # Override adapter type if specified
        if adapter_type:
            spec.lora_layers.adapter_type = adapter_type

        # Load QwenImage model
        from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage

        model_config, model_path = resolve_qwen_model_spec(spec.model)
        qwen = QwenImage(
            quantize=spec.quantize,
            model_path=model_path,
            model_config=model_config,
        )

        # Apply LoRA/DoRA layers to transformer
        QwenDreamBoothInitializer._apply_adapter_layers(
            qwen=qwen,
            spec=spec,
        )

        # Create inference config
        config = Config(
            width=spec.width,
            height=spec.height,
            guidance=spec.guidance,
            num_inference_steps=spec.steps,
            model_config=qwen.model_config,
        )

        # Prepare dataset with optional caching
        cache = None
        if spec.cache.enabled:
            cache = EmbeddingCache(Path(spec.cache.cache_dir).expanduser())

        # Convert example specs to QwenExampleSpec format
        raw_data = [QwenExampleSpec(prompt=ex.prompt, image=ex.image) for ex in spec.examples]

        dataset = QwenDataset.prepare_dataset(
            qwen=qwen,
            raw_data=raw_data,
            width=spec.width,
            height=spec.height,
            augment=True,
            cache=cache,
        )

        # Create training state
        state = QwenTrainingState.create(
            dataset=dataset,
            spec=spec,
            model=qwen,
        )

        return qwen, config, spec, state

    @staticmethod
    def _apply_adapter_layers(qwen: Any, spec: QwenTrainingSpec) -> None:
        """
        Apply LoRA/DoRA layers to transformer.

        Args:
            qwen: QwenImage model
            spec: Training specification with LoRA config
        """
        if spec.lora_layers.transformer_blocks is None:
            print("No LoRA layers specified, using defaults")
            # Default: apply to attention layers in middle blocks
            block_range = list(range(20, 40))  # Middle 20 blocks
            layer_types = ["attn.to_q", "attn.to_k", "attn.to_v"]
            lora_rank = 16
        else:
            block_range = spec.lora_layers.transformer_blocks.block_range.get_blocks()
            layer_types = spec.lora_layers.transformer_blocks.layer_types
            lora_rank = spec.lora_layers.transformer_blocks.lora_rank

        adapter_type = AdapterType.from_string(spec.lora_layers.adapter_type)
        scale = spec.lora_layers.lora_scale

        # Apply adapter layers
        QwenDreamBoothInitializer._apply_to_transformer(
            transformer=qwen.transformer,
            block_range=block_range,
            layer_types=layer_types,
            adapter_type=adapter_type,
            rank=lora_rank,
            scale=scale,
        )

        # Force evaluation
        mx.eval(qwen.parameters())

        print(
            f"Applied {adapter_type.value.upper()} layers: "
            f"blocks {block_range[0]}-{block_range[-1]}, "
            f"rank={lora_rank}, scale={scale}"
        )

    @staticmethod
    def _apply_to_transformer(
        transformer: Any,
        block_range: list[int],
        layer_types: list[str],
        adapter_type: AdapterType,
        rank: int,
        scale: float,
    ) -> None:
        """
        Apply adapter layers to specific transformer blocks.

        Args:
            transformer: QwenTransformer model
            block_range: List of block indices to modify
            layer_types: List of layer type patterns to match
            adapter_type: Type of adapter to apply
            rank: LoRA rank
            scale: LoRA scale
        """
        from mlx import nn

        applied_count = 0

        # Iterate through transformer blocks
        for block_idx in block_range:
            block = transformer.transformer_blocks[block_idx]

            for layer_path in layer_types:
                # Navigate to the target layer
                parts = layer_path.split(".")
                target = block

                try:
                    for part in parts[:-1]:
                        target = getattr(target, part)

                    layer_name = parts[-1]
                    layer = getattr(target, layer_name)

                    if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
                        # Replace with adapter layer
                        adapter = AdapterFactory.from_linear(
                            linear=layer,
                            adapter_type=adapter_type,
                            r=rank,
                            scale=scale,
                        )
                        setattr(target, layer_name, adapter)
                        applied_count += 1

                except AttributeError:
                    # Layer path not found in this block
                    pass

        print(f"Applied {applied_count} adapter layers")


def initialize_from_folder(
    folder_path: str,
    default_prompt: str | None = None,
    adapter_type: str = "lora",
    output_path: str = "~/qwen_training",
    **kwargs,
) -> tuple[Any, Config, QwenTrainingSpec, QwenTrainingState]:
    """
    Quick initialization from a folder of images.

    Args:
        folder_path: Path to folder with images and caption files
        default_prompt: Default prompt if no caption file exists
        adapter_type: Type of adapter ("lora" or "dora")
        output_path: Path for training outputs
        **kwargs: Additional training spec overrides

    Returns:
        Tuple of (qwen_model, config, training_spec, training_state)
    """
    # Create example specs from folder
    example_specs = QwenDatasetFromFolder.from_folder(
        folder=folder_path,
        default_prompt=default_prompt,
    )

    # Create minimal spec
    spec_dict = {
        "model": kwargs.get("model", "qwen-image"),
        "seed": kwargs.get("seed", 42),
        "steps": kwargs.get("steps", 4),
        "guidance": kwargs.get("guidance", 4.0),
        "width": kwargs.get("width", 1024),
        "height": kwargs.get("height", 1024),
        "training_loop": {
            "num_epochs": kwargs.get("num_epochs", 100),
            "batch_size": kwargs.get("batch_size", 1),
            "gradient_accumulation_steps": kwargs.get("gradient_accumulation_steps", 1),
        },
        "optimizer": {
            "name": kwargs.get("optimizer", "AdamW"),
            "learning_rate": kwargs.get("learning_rate", 1e-4),
        },
        "lr_scheduler": {
            "name": kwargs.get("lr_scheduler", "cosine"),
            "warmup_steps": kwargs.get("warmup_steps", 100),
        },
        "lora_layers": {
            "adapter_type": adapter_type,
            "transformer_blocks": {
                "block_range": {"start": 20, "end": 40},
                "layer_types": ["attn.to_q", "attn.to_k", "attn.to_v"],
                "lora_rank": kwargs.get("lora_rank", 16),
            },
        },
        "save": {
            "checkpoint_frequency": kwargs.get("checkpoint_frequency", 500),
            "output_path": output_path,
        },
        "examples": {
            "path": str(folder_path),
            "images": [{"image": str(ex.image.name), "prompt": ex.prompt} for ex in example_specs],
        },
    }

    # Create temp config and initialize
    import json
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(spec_dict, f)
        config_path = f.name

    return QwenDreamBoothInitializer.initialize(
        config_path=config_path,
        adapter_type=adapter_type,
    )
