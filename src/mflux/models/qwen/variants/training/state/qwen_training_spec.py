"""
Qwen-Image Training Specification.

Configuration dataclasses for Qwen-Image training, supporting:
- LoRA and DoRA adapters
- Learning rate scheduling
- Gradient accumulation
- EMA model tracking
- Embedding caching
"""

import datetime
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QwenExampleSpec:
    """Single training example specification."""

    image: Path
    prompt: str

    @classmethod
    def create(
        cls,
        param: dict[str, str],
        base_path: Path | None,
        images_path: str,
    ) -> "QwenExampleSpec":
        image_path = Path(param["image"])

        # Resolve relative paths
        if not image_path.is_absolute():
            if base_path is not None:
                image_path = base_path.parent / images_path / image_path
            else:
                image_path = Path(images_path) / image_path

        return cls(image=image_path, prompt=param["prompt"])


@dataclass
class TrainingLoopSpec:
    """Training loop configuration."""

    num_epochs: int
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    iterator_state_path: str | None = None


@dataclass
class OptimizerSpec:
    """Optimizer configuration."""

    name: str = "AdamW"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    state_path: str | None = None


@dataclass
class LRSchedulerSpec:
    """Learning rate scheduler configuration."""

    name: str = "cosine"  # cosine, onecycle, linear_warmup
    warmup_steps: int = 100
    min_lr: float = 1e-6
    # OneCycle specific
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4


@dataclass
class EMASpec:
    """EMA model configuration."""

    enabled: bool = True
    decay: float = 0.9999


@dataclass
class CacheSpec:
    """Embedding cache configuration."""

    enabled: bool = True
    cache_dir: str = "~/.cache/qwen_training"


@dataclass
class SaveSpec:
    """Checkpoint saving configuration."""

    checkpoint_frequency: int = 500
    output_path: str = "~/qwen_training"


@dataclass
class InstrumentationSpec:
    """Validation and monitoring configuration."""

    plot_frequency: int = 100
    generate_image_frequency: int = 500
    validation_prompt: str = ""


@dataclass
class StatisticsSpec:
    """Training statistics configuration."""

    state_path: str | None = None


@dataclass
class BlockRange:
    """Range specification for transformer blocks."""

    start: int | None = None
    end: int | None = None
    indices: list[int] | None = None

    def get_blocks(self) -> list[int]:
        if self.indices:
            return self.indices
        if self.start is not None and self.end is not None:
            return list(range(self.start, self.end))
        raise ValueError("Either 'start'/'end' or 'indices' must be provided")


@dataclass
class TransformerBlocksSpec:
    """LoRA/DoRA target specification for transformer blocks."""

    block_range: BlockRange
    layer_types: list[str] = field(
        default_factory=lambda: ["attn.to_q", "attn.to_k", "attn.to_v", "attn.attn_to_out.0"]
    )
    lora_rank: int = 16


@dataclass
class LoraLayersSpec:
    """LoRA/DoRA configuration."""

    adapter_type: str = "lora"  # "lora" or "dora"
    transformer_blocks: TransformerBlocksSpec | None = None
    lora_scale: float = 1.0
    state_path: str | None = None


@dataclass
class QwenTrainingSpec:
    """
    Complete training specification for Qwen-Image.

    This dataclass holds all configuration needed to run training,
    including model parameters, optimizer settings, LoRA configuration,
    and data paths.
    """

    model: str = "qwen-image"
    seed: int = 42
    steps: int = 4  # Inference steps for validation
    guidance: float = 4.0
    quantize: int | None = None  # Quantization bits (None = no quant)
    width: int = 1024
    height: int = 1024

    training_loop: TrainingLoopSpec = field(default_factory=TrainingLoopSpec)
    optimizer: OptimizerSpec = field(default_factory=OptimizerSpec)
    lr_scheduler: LRSchedulerSpec = field(default_factory=LRSchedulerSpec)
    ema: EMASpec = field(default_factory=EMASpec)
    cache: CacheSpec = field(default_factory=CacheSpec)
    saver: SaveSpec = field(default_factory=SaveSpec)
    instrumentation: InstrumentationSpec | None = None
    statistics: StatisticsSpec = field(default_factory=StatisticsSpec)
    lora_layers: LoraLayersSpec = field(default_factory=LoraLayersSpec)

    examples: list[QwenExampleSpec] = field(default_factory=list)

    config_path: str | None = None
    checkpoint_path: str | None = None

    @staticmethod
    def resolve(
        config_path: str | None = None,
        checkpoint_path: str | None = None,
    ) -> "QwenTrainingSpec":
        """
        Resolve training spec from config file or checkpoint.

        Args:
            config_path: Path to JSON config file
            checkpoint_path: Path to checkpoint to resume from

        Returns:
            Resolved QwenTrainingSpec
        """
        if not config_path and not checkpoint_path:
            raise ValueError("Either config_path or checkpoint_path required")

        if checkpoint_path:
            return QwenTrainingSpec._from_checkpoint(checkpoint_path)

        return QwenTrainingSpec._from_config(config_path)

    @staticmethod
    def _from_config(path: str) -> "QwenTrainingSpec":
        """Load spec from JSON config file."""
        with open(Path(path), "r") as f:
            data = json.load(f)

        return QwenTrainingSpec.from_dict(data, config_path=path)

    @staticmethod
    def from_dict(config: dict[str, Any], config_path: str | None = None) -> "QwenTrainingSpec":
        """
        Create spec from configuration dictionary.

        Args:
            config: Configuration dictionary
            config_path: Path to config file (for resolving relative paths)

        Returns:
            QwenTrainingSpec instance
        """
        base_path = Path(config_path).absolute() if config_path else None

        # Parse examples
        examples = []
        if "examples" in config:
            images_path = config["examples"].get("path", ".")
            for ex in config["examples"].get("images", []):
                examples.append(QwenExampleSpec.create(ex, base_path, images_path))

        # Parse sub-specs
        training_loop = TrainingLoopSpec(**config.get("training_loop", {}))
        optimizer = OptimizerSpec(**config.get("optimizer", {}))
        lr_scheduler = LRSchedulerSpec(**config.get("lr_scheduler", {}))
        ema = EMASpec(**config.get("ema", {}))
        cache = CacheSpec(**config.get("cache", {}))
        saver = SaveSpec(**config.get("save", {}))

        # Resolve output path
        saver.output_path = QwenTrainingSpec._resolve_output_path(saver.output_path, new_folder=True)

        instrumentation = None
        if config.get("instrumentation"):
            instrumentation = InstrumentationSpec(**config["instrumentation"])

        statistics = StatisticsSpec(**config.get("statistics", {}))

        # Parse LoRA config
        lora_config = config.get("lora_layers", {})
        lora_layers = LoraLayersSpec(
            adapter_type=lora_config.get("adapter_type", "lora"),
            lora_scale=lora_config.get("lora_scale", 1.0),
        )
        if "transformer_blocks" in lora_config:
            tb = lora_config["transformer_blocks"]
            lora_layers.transformer_blocks = TransformerBlocksSpec(
                block_range=BlockRange(**tb.get("block_range", {})),
                layer_types=tb.get("layer_types", ["attn.to_q", "attn.to_k", "attn.to_v"]),
                lora_rank=tb.get("lora_rank", 16),
            )

        return QwenTrainingSpec(
            model=config.get("model", "qwen-image"),
            seed=config.get("seed", 42),
            steps=config.get("steps", 4),
            guidance=config.get("guidance", 4.0),
            quantize=config.get("quantize"),
            width=config.get("width", 1024),
            height=config.get("height", 1024),
            training_loop=training_loop,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            ema=ema,
            cache=cache,
            saver=saver,
            instrumentation=instrumentation,
            statistics=statistics,
            lora_layers=lora_layers,
            examples=examples,
            config_path=str(base_path) if base_path else None,
        )

    @staticmethod
    def _from_checkpoint(path: str) -> "QwenTrainingSpec":
        """
        Load spec from checkpoint ZIP file.

        Extracts the config.json from the checkpoint and creates
        a QwenTrainingSpec with checkpoint_path set for state restoration.
        """
        import tempfile
        import zipfile

        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        if not zipfile.is_zipfile(checkpoint_path):
            raise ValueError(f"Invalid checkpoint file (not a ZIP): {path}")

        # Extract config from checkpoint
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(checkpoint_path, "r") as zipf:
                # Find config file in archive
                config_files = [n for n in zipf.namelist() if n.endswith("_config.json")]
                if not config_files:
                    raise ValueError(f"Checkpoint missing config file: {path}")

                # Extract config
                config_name = config_files[0]
                zipf.extract(config_name, temp_dir)
                config_path = Path(temp_dir) / config_name

                with open(config_path, "r") as f:
                    config_data = json.load(f)

        # Create spec from config and set checkpoint path
        spec = QwenTrainingSpec.from_dict(config_data)
        spec.checkpoint_path = str(checkpoint_path)

        return spec

    @staticmethod
    def _resolve_output_path(path: str, new_folder: bool) -> str:
        """Resolve and create output path."""
        requested_path = os.path.expanduser(path)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if new_folder and os.path.exists(requested_path):
            requested_path = f"{requested_path}_{now_str}"

        os.makedirs(requested_path, exist_ok=True)
        return requested_path

    def to_json(self) -> str:
        """Serialize spec to JSON string."""
        spec_dict = asdict(self)
        return json.dumps(self._serialize(spec_dict), indent=4)

    @staticmethod
    def _serialize(obj: Any) -> Any:
        """Custom serialization for Path objects."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, list):
            return [QwenTrainingSpec._serialize(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: QwenTrainingSpec._serialize(v) for k, v in obj.items()}
        return obj

    @property
    def total_steps(self) -> int:
        """
        Calculate total training steps (optimizer updates).

        Note: With gradient accumulation, an "optimizer step" happens only
        after accumulation_steps forward/backward passes. This property
        returns the number of OPTIMIZER UPDATES, which is what the LR
        scheduler should use.

        Total optimizer steps = (examples_per_epoch // batch_size) * epochs // accumulation_steps
        """
        num_examples = len(self.examples)
        batch_size = self.training_loop.batch_size
        accumulation_steps = self.training_loop.gradient_accumulation_steps

        # Batches per epoch (forward/backward passes)
        batches_per_epoch = max(1, num_examples // batch_size)

        # Total batches across all epochs
        total_batches = batches_per_epoch * self.training_loop.num_epochs

        # Optimizer steps (divide by accumulation since we only step every N batches)
        return max(1, total_batches // accumulation_steps)

    @property
    def total_iterations(self) -> int:
        """
        Calculate total training iterations (forward/backward passes).

        This is the total number of batches processed, regardless of
        gradient accumulation. Use total_steps for LR scheduler.
        """
        num_examples = len(self.examples)
        batch_size = self.training_loop.batch_size
        batches_per_epoch = max(1, num_examples // batch_size)
        return batches_per_epoch * self.training_loop.num_epochs
