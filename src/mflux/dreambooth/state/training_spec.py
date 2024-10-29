import datetime
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass
class ExampleSpec:
    image: Path
    prompt: str

    @classmethod
    def create(cls, param: dict[str, str], base_path: str) -> "ExampleSpec":
        image_path = Path(param["image"])

        # If the path is not absolute, resolve it relative to the base path
        if not image_path.is_absolute():
            image_path = Path(base_path).parent / image_path

        return cls(
            image=image_path,
            prompt=param["prompt"],
        )


@dataclass
class TrainingLoopSpec:
    num_epochs: int
    batch_size: int
    iterator_state_path: str | None = None


@dataclass
class OptimizerSpec:
    name: str
    learning_rate: float
    state_path: str | None = None


@dataclass
class SaveSpec:
    checkpoint_frequency: int
    output_path: str


@dataclass
class StatisticsSpec:
    state_path: str | None = None


@dataclass
class BlockRange:
    start: int
    end: int


@dataclass
class SingleTransformerBlocks:
    block_range: BlockRange
    layer_types: List[str]
    lora_rank: int


@dataclass
class LoraLayersSpec:
    single_transformer_blocks: SingleTransformerBlocks
    state_path: str | None = None


@dataclass
class InstrumentationSpec:
    plot_frequency: int
    generate_image_frequency: int
    validation_prompt: str


@dataclass
class TrainingSpec:
    model: str
    seed: int
    steps: int
    guidance: float
    quantize: int
    width: int
    height: int
    training_loop: TrainingLoopSpec
    optimizer: OptimizerSpec
    saver: SaveSpec
    instrumentation: InstrumentationSpec
    lora_layers: LoraLayersSpec
    statistics: StatisticsSpec
    examples: List[ExampleSpec]
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
        now = datetime.datetime.now()

        with open(Path(path), "r") as f:
            data = json.load(f)

        # Parse nested structures first
        examples = [ExampleSpec.create(ex, path) for ex in data["examples"]]
        training_loop = TrainingLoopSpec(**data["training_loop"])
        optimizer = OptimizerSpec(**data["optimizer"])
        saver = SaveSpec(
            checkpoint_frequency=data["save"]["checkpoint_frequency"],
            output_path=TrainingSpec._resolve_output_path(data["save"]["output_path"], now, new_folder),
        )
        instrumentation = (
            None if data.get("instrumentation", None) is None else InstrumentationSpec(**data["instrumentation"])
        )
        statistics = StatisticsSpec() if data.get("statistics", None) is None else StatisticsSpec(**data["statistics"])

        # Parse LoraLayers structure
        block_range = BlockRange(**data["lora_layers"]["single_transformer_blocks"]["block_range"])
        single_transformer_blocks = SingleTransformerBlocks(
            block_range=block_range,
            layer_types=data["lora_layers"]["single_transformer_blocks"]["layer_types"],
            lora_rank=data["lora_layers"]["single_transformer_blocks"]["lora_rank"],
        )
        lora_layers = LoraLayersSpec(single_transformer_blocks=single_transformer_blocks)

        # Create the training specification
        return TrainingSpec(
            model=data["model"],
            seed=data["seed"],
            steps=data["steps"],
            guidance=data["guidance"],
            quantize=data["quantize"],
            width=data["width"],
            height=data["height"],
            training_loop=training_loop,
            optimizer=optimizer,
            saver=saver,
            instrumentation=instrumentation,
            lora_layers=lora_layers,
            statistics=statistics,
            examples=examples,
            config_path=path,
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
        elif isinstance(obj, list):
            return [TrainingSpec._custom_serializer(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: TrainingSpec._custom_serializer(value) for key, value in obj.items()}
        return obj

    @staticmethod
    def _resolve_output_path(path: str, now: datetime, new_folder: bool) -> str:
        requested_path = os.path.expanduser(path)
        now_str = now.strftime("%Y%m%d_%H%M%S")

        if new_folder and os.path.exists(requested_path):
            requested_path = f"{requested_path}_{now_str}"

        os.makedirs(requested_path, exist_ok=True)
        return requested_path

    @staticmethod
    def _from_checkpoint(path: str, new_folder: bool) -> "TrainingSpec":
        with open(Path(path), "r") as f:
            checkpoint = json.load(f)

        config_path = checkpoint["config"]
        spec = TrainingSpec._from_config(config_path, new_folder)
        spec.optimizer.state_path = checkpoint["optimizer"]
        spec.lora_layers.state_path = checkpoint["adapter"]
        spec.training_loop.iterator_state_path = checkpoint["iterator"]
        spec.statistics.state_path = checkpoint["statistics"]
        return spec
