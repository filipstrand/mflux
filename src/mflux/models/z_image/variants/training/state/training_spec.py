import datetime
import json
import os
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import List

from mflux.models.z_image.variants.training.state.zip_util import ZipUtil


class TrainingMode(str, Enum):
    """Training mode: LoRA for efficient fine-tuning, full for complete model training."""

    LORA = "lora"
    FULL = "full"


@dataclass
class ExampleSpec:
    """Specification for a single training example (image + caption)."""

    image: Path
    prompt: str

    @classmethod
    def create(cls, param: dict[str, str], absolute_or_relative_path: str, base_path: Path | None) -> "ExampleSpec":
        raw_image_path = param.get("image", "")

        # Security: Reject paths with traversal sequences
        if ".." in raw_image_path or raw_image_path.startswith("/"):
            raise ValueError(f"Security: Invalid image path '{raw_image_path}' - path traversal not allowed")

        image_path = Path(raw_image_path)

        if base_path is not None and not Path(absolute_or_relative_path).is_absolute():
            image_path = Path(base_path).parent / absolute_or_relative_path / image_path
        else:
            image_path = Path(absolute_or_relative_path) / image_path

        # Resolve and validate final path
        resolved = image_path.resolve()

        return cls(
            image=resolved,
            prompt=param.get("prompt", ""),
        )


@dataclass
class TrainingLoopSpec:
    """Training loop configuration."""

    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int = 1
    iterator_state_path: str | None = None


@dataclass
class OptimizerSpec:
    """Optimizer configuration."""

    name: str
    learning_rate: float
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    max_grad_norm: float = 1.0  # Gradient clipping threshold (0 to disable)
    warmup_steps: int = 0  # Learning rate warmup steps
    warmup_ratio: float = 0.0  # Alternative: warmup as ratio of total steps (if > 0, overrides warmup_steps)
    state_path: str | None = None
    # LR Scheduler configuration
    scheduler_type: str = "constant"  # "constant", "cosine", "onecycle", "linear_warmup"
    min_lr: float = 0.0  # Minimum LR for cosine scheduler
    pct_start: float = 0.3  # Warmup percentage for OneCycle scheduler


@dataclass
class SaveSpec:
    """Checkpoint saving configuration.

    Attributes:
        checkpoint_frequency: How often to save checkpoints (in iterations).
        output_path: Directory path for saving checkpoints.
        keep_last_n_checkpoints: Number of recent checkpoints to keep. 0 means keep all (default).
    """

    checkpoint_frequency: int
    output_path: str
    keep_last_n_checkpoints: int = 0  # 0 = keep all (backward compatible)

    def __post_init__(self) -> None:
        """Validate checkpoint saving configuration."""
        if self.checkpoint_frequency < 1:
            raise ValueError(f"checkpoint_frequency must be >= 1, got {self.checkpoint_frequency}")
        if self.keep_last_n_checkpoints < 0:
            raise ValueError(f"keep_last_n_checkpoints must be >= 0, got {self.keep_last_n_checkpoints}")
        # Validate output_path doesn't contain path traversal patterns
        if ".." in self.output_path:
            raise ValueError(f"output_path cannot contain '..': {self.output_path}")


@dataclass
class StatisticsSpec:
    """Loss statistics configuration."""

    state_path: str | None = None


@dataclass
class BlockRange:
    """Flexible block range specification for transformer layers."""

    start: int | None = None
    end: int | None = None
    indices: List[int] | None = None

    def get_blocks(self) -> List[int]:
        if self.indices:
            return self.indices
        if self.start is not None and self.end is not None:
            return list(range(self.start, self.end))
        raise ValueError("Either 'start' and 'end' or 'indices' must be provided.")


@dataclass
class ZImageTransformerBlocks:
    """Z-Image transformer block specification for LoRA training.

    Z-Image has:
    - 30 main layers
    - 2 noise_refiner layers
    - 2 context_refiner layers

    LoRA targets: to_q, to_k, to_v, to_out.0, w1, w2, w3, adaLN_modulation.0
    """

    block_range: BlockRange
    layer_types: List[str]
    lora_rank: int
    layer_type: str = "layers"  # "layers", "noise_refiner", or "context_refiner"


@dataclass
class LoraLayersSpec:
    """LoRA layer configuration for Z-Image training."""

    main_layers: ZImageTransformerBlocks | None = None
    noise_refiner: ZImageTransformerBlocks | None = None
    context_refiner: ZImageTransformerBlocks | None = None
    state_path: str | None = None

    @staticmethod
    def default_full() -> "LoraLayersSpec":
        """Default LoRA configuration targeting all trainable layers."""
        default_layer_types = [
            "attention.to_q",
            "attention.to_k",
            "attention.to_v",
            "attention.to_out.0",
            "feed_forward.w1",
            "feed_forward.w2",
            "feed_forward.w3",
            "adaLN_modulation.0",
        ]
        return LoraLayersSpec(
            main_layers=ZImageTransformerBlocks(
                block_range=BlockRange(start=0, end=30),
                layer_types=default_layer_types,
                lora_rank=32,
                layer_type="layers",
            ),
            noise_refiner=ZImageTransformerBlocks(
                block_range=BlockRange(start=0, end=2),
                layer_types=default_layer_types,
                lora_rank=32,
                layer_type="noise_refiner",
            ),
            context_refiner=ZImageTransformerBlocks(
                block_range=BlockRange(start=0, end=2),
                layer_types=["attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0"],
                lora_rank=32,
                layer_type="context_refiner",
            ),
        )


@dataclass
class DatasetSpec:
    """Dataset augmentation and preprocessing configuration."""

    repeat_count: int = 1  # How many times to repeat each example (for small datasets)
    enable_augmentation: bool = True  # Enable flip/crop augmentations
    random_crop: bool = False  # Apply random crop during encoding
    use_aspect_ratio_bucketing: bool = False  # Group images by aspect ratio for efficient batching


@dataclass
class EMASpec:
    """EMA (Exponential Moving Average) configuration.

    Attributes:
        enabled: Whether to use EMA for weight smoothing during training.
        decay: EMA decay factor in range [0.0, 1.0]. Higher values mean slower
               updates and smoother weights. Typical values: 0.999-0.9999.
        state_path: Path to EMA state in checkpoint (optional).
    """

    enabled: bool = False
    decay: float = 0.9999
    state_path: str | None = None

    def __post_init__(self) -> None:
        """Validate EMA configuration."""
        if not 0.0 <= self.decay <= 1.0:
            raise ValueError(f"EMA decay must be in range [0.0, 1.0], got {self.decay}")
        # Warn about unusual decay values that may indicate configuration errors
        if self.enabled and (self.decay < 0.99 or self.decay > 0.99999):
            import warnings

            warnings.warn(
                f"EMA decay {self.decay} is outside typical range [0.99, 0.99999]. "
                f"Very low values cause rapid weight changes; very high values may not smooth enough.",
                UserWarning,
                stacklevel=2,
            )


@dataclass
class EarlyStoppingSpec:
    """Early stopping configuration.

    Attributes:
        enabled: Whether to use early stopping during training.
        patience: Number of validations without improvement before stopping.
        min_delta: Minimum improvement threshold to consider as improvement.
    """

    enabled: bool = False
    patience: int = 5  # Validations without improvement before stopping
    min_delta: float = 0.0  # Minimum improvement threshold

    def __post_init__(self) -> None:
        """Validate early stopping configuration."""
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if self.min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")


@dataclass
class FullFinetuneSpec:
    """Full fine-tuning configuration."""

    train_transformer: bool = True
    train_vae: bool = False
    train_text_encoder: bool = False
    gradient_checkpointing: bool = False  # Not typically needed with 512GB


@dataclass
class InstrumentationSpec:
    """Validation and monitoring configuration."""

    plot_frequency: int
    generate_image_frequency: int
    validation_prompt: str
    negative_prompt: str = ""
    guidance_scale: float = 3.5  # Z-Image-Base supports CFG


@dataclass
class TrainingSpec:
    """Complete training specification for Z-Image."""

    model: str
    seed: int
    steps: int
    guidance: float
    quantize: int | None
    width: int
    height: int
    mode: TrainingMode
    training_loop: TrainingLoopSpec
    optimizer: OptimizerSpec
    saver: SaveSpec
    instrumentation: InstrumentationSpec | None
    statistics: StatisticsSpec
    examples: List[ExampleSpec]
    lora_layers: LoraLayersSpec | None = None  # For LoRA mode
    full_finetune: FullFinetuneSpec | None = None  # For full mode
    dataset: DatasetSpec | None = None  # Dataset augmentation config
    ema: EMASpec | None = None  # EMA configuration
    early_stopping: EarlyStoppingSpec | None = None  # Early stopping configuration
    config_path: str | None = None
    checkpoint_path: str | None = None

    @staticmethod
    def resolve(config_path: str | None, checkpoint_path: str | None) -> "TrainingSpec":
        if not config_path and not checkpoint_path:
            raise ValueError("Either 'config_path' or 'checkpoint_path' must be provided.")

        if checkpoint_path:
            return TrainingSpec._from_checkpoint(checkpoint_path, new_folder=False)

        return TrainingSpec._from_config(config_path, new_folder=True)

    @staticmethod
    def _from_config(path: str, new_folder: bool = True) -> "TrainingSpec":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Training config file not found: {config_path}")
        if not config_path.is_file():
            raise ValueError(f"Training config path is not a file: {config_path}")

        with open(config_path, "r") as f:
            data = json.load(f)

        return TrainingSpec.from_conf(data, path, new_folder)

    @staticmethod
    def from_conf(config: dict, config_path: str | None, new_folder: bool = True) -> "TrainingSpec":
        absolute_config_path = None if config_path is None else Path(config_path).absolute()
        absolute_or_relative_path = config["examples"]["path"]
        examples = [
            ExampleSpec.create(ex, absolute_or_relative_path, absolute_config_path)
            for ex in config["examples"]["images"]
        ]

        training_loop = TrainingLoopSpec(**config["training_loop"])
        optimizer = OptimizerSpec(**config["optimizer"])
        saver = SaveSpec(
            checkpoint_frequency=config["save"]["checkpoint_frequency"],
            output_path=TrainingSpec._resolve_output_path(config["save"]["output_path"], new_folder),
            keep_last_n_checkpoints=config["save"].get("keep_last_n_checkpoints", 0),
        )
        instrumentation = (
            None if config.get("instrumentation", None) is None else InstrumentationSpec(**config["instrumentation"])
        )
        statistics = (
            StatisticsSpec() if config.get("statistics", None) is None else StatisticsSpec(**config["statistics"])
        )

        # Determine training mode
        mode = TrainingMode(config.get("mode", "lora"))

        # Parse LoRA layers (for LoRA mode)
        lora_layers = None
        if mode == TrainingMode.LORA:
            lora_config = config.get("lora_layers", None)
            if lora_config:
                lora_layers = TrainingSpec._parse_lora_layers(lora_config)
            else:
                # Use default full configuration
                lora_layers = LoraLayersSpec.default_full()

        # Parse full fine-tune spec (for full mode)
        full_finetune = None
        if mode == TrainingMode.FULL:
            full_config = config.get("full_finetune", None)
            if full_config:
                full_finetune = FullFinetuneSpec(**full_config)
            else:
                full_finetune = FullFinetuneSpec()

        # Parse dataset spec
        dataset_config = config.get("dataset", None)
        dataset_spec = DatasetSpec(**dataset_config) if dataset_config else DatasetSpec()

        # Parse EMA spec
        ema_config = config.get("ema", None)
        ema_spec = EMASpec(**ema_config) if ema_config else EMASpec()

        # Parse early stopping spec
        early_stopping_config = config.get("early_stopping", None)
        early_stopping_spec = EarlyStoppingSpec(**early_stopping_config) if early_stopping_config else None

        return TrainingSpec(
            model=config.get("model", "z-image"),
            seed=config["seed"],
            steps=config["steps"],
            guidance=config.get("guidance", 3.5),  # Z-Image-Base default
            quantize=config.get("quantize", None),
            width=config["width"],
            height=config["height"],
            mode=mode,
            training_loop=training_loop,
            optimizer=optimizer,
            saver=saver,
            instrumentation=instrumentation,
            lora_layers=lora_layers,
            full_finetune=full_finetune,
            dataset=dataset_spec,
            ema=ema_spec,
            early_stopping=early_stopping_spec,
            statistics=statistics,
            examples=examples,
            config_path=None if absolute_config_path is None else str(absolute_config_path),
        )

    @staticmethod
    def _parse_lora_layers(lora_config: dict) -> LoraLayersSpec:
        """Parse LoRA layers configuration from dict."""
        main_layers = None
        noise_refiner = None
        context_refiner = None

        if "main_layers" in lora_config:
            main_layers = ZImageTransformerBlocks(
                block_range=BlockRange(**lora_config["main_layers"]["block_range"]),
                layer_types=lora_config["main_layers"]["layer_types"],
                lora_rank=lora_config["main_layers"]["lora_rank"],
                layer_type="layers",
            )

        if "noise_refiner" in lora_config:
            noise_refiner = ZImageTransformerBlocks(
                block_range=BlockRange(**lora_config["noise_refiner"]["block_range"]),
                layer_types=lora_config["noise_refiner"]["layer_types"],
                lora_rank=lora_config["noise_refiner"]["lora_rank"],
                layer_type="noise_refiner",
            )

        if "context_refiner" in lora_config:
            context_refiner = ZImageTransformerBlocks(
                block_range=BlockRange(**lora_config["context_refiner"]["block_range"]),
                layer_types=lora_config["context_refiner"]["layer_types"],
                lora_rank=lora_config["context_refiner"]["lora_rank"],
                layer_type="context_refiner",
            )

        return LoraLayersSpec(
            main_layers=main_layers,
            noise_refiner=noise_refiner,
            context_refiner=context_refiner,
            state_path=lora_config.get("state_path"),
        )

    def to_json(self) -> str:
        spec_dict = asdict(self)
        serialized_dict = TrainingSpec._custom_serializer(spec_dict)
        return json.dumps(serialized_dict, indent=4)

    @staticmethod
    def _custom_serializer(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, datetime.datetime):
            return obj.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(obj, TrainingMode):
            return obj.value
        elif isinstance(obj, list):
            return [TrainingSpec._custom_serializer(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: TrainingSpec._custom_serializer(value) for key, value in obj.items()}
        return obj

    # Resource limits to prevent DoS via malicious configs
    MAX_NUM_EPOCHS = 10_000
    MAX_GRADIENT_ACCUMULATION_STEPS = 1024
    MAX_CHECKPOINT_FILE_SIZE_MB = 100

    @staticmethod
    def _validate_config_values(config: dict) -> None:
        """Validate config values are within safe bounds.

        This provides defense-in-depth against malicious training configs
        or corrupted checkpoint files that could cause resource exhaustion.
        """
        # Validate training loop parameters
        if "training_loop" in config:
            loop = config["training_loop"]
            batch_size = loop.get("batch_size", 1)
            if not isinstance(batch_size, int) or batch_size < 1 or batch_size > 128:
                raise ValueError(f"Invalid batch_size: {batch_size}. Must be 1-128.")

            num_epochs = loop.get("num_epochs")
            if num_epochs is not None:
                if not isinstance(num_epochs, int) or num_epochs < 1:
                    raise ValueError(f"Invalid num_epochs: {num_epochs}. Must be positive integer or null.")
                if num_epochs > TrainingSpec.MAX_NUM_EPOCHS:
                    raise ValueError(f"Invalid num_epochs: {num_epochs}. Must be <= {TrainingSpec.MAX_NUM_EPOCHS}.")

            grad_accum = loop.get("gradient_accumulation_steps", 1)
            if (
                not isinstance(grad_accum, int)
                or grad_accum < 1
                or grad_accum > TrainingSpec.MAX_GRADIENT_ACCUMULATION_STEPS
            ):
                raise ValueError(
                    f"Invalid gradient_accumulation_steps: {grad_accum}. Must be 1-{TrainingSpec.MAX_GRADIENT_ACCUMULATION_STEPS}."
                )

        # Validate optimizer parameters
        if "optimizer" in config:
            opt = config["optimizer"]
            lr = opt.get("learning_rate", 1e-4)
            if not isinstance(lr, (int, float)) or lr <= 0 or lr > 1.0:
                raise ValueError(f"Invalid learning_rate: {lr}. Must be in range (0, 1].")

        # Validate dimensions
        width = config.get("width", 512)
        height = config.get("height", 512)
        if not isinstance(width, int) or width < 64 or width > 4096:
            raise ValueError(f"Invalid width: {width}. Must be 64-4096.")
        if not isinstance(height, int) or height < 64 or height > 4096:
            raise ValueError(f"Invalid height: {height}. Must be 64-4096.")

    @staticmethod
    def _resolve_output_path(path: str, new_folder: bool) -> str:
        # Expand user home directory
        requested_path = os.path.expanduser(path)

        # Security: Reject paths with traversal sequences after expansion
        if ".." in requested_path:
            raise ValueError(f"Security: Path traversal not allowed in output path: {path}")

        # Resolve to absolute path and verify it's valid
        resolved_path = os.path.realpath(requested_path)

        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if new_folder and os.path.exists(resolved_path):
            resolved_path = f"{resolved_path}_{now_str}"

        # Use atomic directory creation to minimize race condition window
        os.makedirs(resolved_path, exist_ok=True)
        return resolved_path

    @staticmethod
    def _from_checkpoint(path: str, new_folder: bool) -> "TrainingSpec":
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint path is not a file: {checkpoint_path}")

        # Defense-in-depth: Validate file size before loading to prevent memory exhaustion
        file_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        if file_size_mb > TrainingSpec.MAX_CHECKPOINT_FILE_SIZE_MB:
            raise ValueError(
                f"Checkpoint file too large: {file_size_mb:.1f}MB > {TrainingSpec.MAX_CHECKPOINT_FILE_SIZE_MB}MB limit"
            )

        # Load checkpoint metadata with error handling
        # Note: RecursionError and MemoryError are not caught - these indicate serious
        # runtime issues that should propagate naturally for proper program termination.
        # The file size check above provides defense against most DoS attacks.
        try:
            checkpoint = ZipUtil.unzip(path, "checkpoint.json", lambda x: json.load(open(x, "r")))
        except (json.JSONDecodeError, KeyError, IOError) as e:
            raise ValueError(f"Invalid or corrupted checkpoint metadata: {e}") from e

        # Validate checkpoint structure
        if not isinstance(checkpoint, dict) or "files" not in checkpoint:
            raise ValueError("Checkpoint metadata missing required 'files' key")

        files = checkpoint["files"]
        required_files = ["config", "optimizer", "iterator", "loss"]
        missing = [f for f in required_files if f not in files]
        if missing:
            raise ValueError(f"Checkpoint missing required files: {missing}")

        # Load config with error handling
        try:
            config = ZipUtil.unzip(path, files["config"], lambda x: json.load(open(x, "r")))
        except (json.JSONDecodeError, KeyError, IOError) as e:
            raise ValueError(f"Invalid or corrupted config in checkpoint: {e}") from e

        # Validate critical config values to prevent malicious checkpoints
        TrainingSpec._validate_config_values(config)

        spec = TrainingSpec.from_conf(config, None, new_folder)
        spec.optimizer.state_path = files["optimizer"]
        if spec.lora_layers:
            spec.lora_layers.state_path = files.get("lora_adapter")
        spec.training_loop.iterator_state_path = files["iterator"]
        spec.statistics.state_path = files["loss"]
        spec.checkpoint_path = path
        spec.config_path = None
        return spec
