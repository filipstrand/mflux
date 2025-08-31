import datetime
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

from mflux.models.flux.variants.dreambooth.state.zip_util import ZipUtil


@dataclass
class ExampleSpec:
    image: Path
    prompt: str

    @classmethod
    def create(cls, param: dict[str, str], absolute_or_relative_path: str, base_path: Path) -> "ExampleSpec":
        image_path = Path(param["image"])

        # If the path is not absolute, resolve it relative to the base path
        if not Path(absolute_or_relative_path).is_absolute():
            image_path = Path(base_path).parent / absolute_or_relative_path / image_path
        else:
            image_path = Path(absolute_or_relative_path) / image_path

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
class TransformerBlocks:
    block_range: BlockRange
    layer_types: List[str]
    lora_rank: int


@dataclass
class SingleTransformerBlocks:
    block_range: BlockRange
    layer_types: List[str]
    lora_rank: int


@dataclass
class LoraLayersSpec:
    transformer_blocks: TransformerBlocks
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
        with open(Path(path), "r") as f:
            data = json.load(f)

        return TrainingSpec.from_conf(data, path, new_folder)

    @staticmethod
    def from_conf(config: dict, config_path: str | None, new_folder: bool = True) -> "TrainingSpec":
        absolute_config_path = None if config_path is None else Path(config_path).absolute()
        absolute_or_relative_path = config["examples"]["path"]
        examples = [ExampleSpec.create(ex, absolute_or_relative_path, absolute_config_path) for ex in config["examples"]["images"]]  # fmt: off
        training_loop = TrainingLoopSpec(**config["training_loop"])
        optimizer = OptimizerSpec(**config["optimizer"])
        saver = SaveSpec(
            checkpoint_frequency=config["save"]["checkpoint_frequency"],
            output_path=TrainingSpec._resolve_output_path(config["save"]["output_path"], new_folder),
        )
        instrumentation = (
            None if config.get("instrumentation", None) is None else InstrumentationSpec(**config["instrumentation"])
        )
        statistics = (
            StatisticsSpec() if config.get("statistics", None) is None else StatisticsSpec(**config["statistics"])
        )

        # Parse LoraLayers structure
        transformer_blocks = config["lora_layers"].get("transformer_blocks", None)
        if transformer_blocks:
            transformer_blocks = TransformerBlocks(
                block_range=BlockRange(**transformer_blocks["block_range"]),
                layer_types=transformer_blocks["layer_types"],
                lora_rank=transformer_blocks["lora_rank"],
            )

        single_transformer_blocks = config["lora_layers"].get("single_transformer_blocks", None)
        if single_transformer_blocks:
            single_transformer_blocks = SingleTransformerBlocks(
                block_range=BlockRange(**single_transformer_blocks["block_range"]),
                layer_types=single_transformer_blocks["layer_types"],
                lora_rank=single_transformer_blocks["lora_rank"],
            )

        lora_layers = LoraLayersSpec(
            transformer_blocks=transformer_blocks, single_transformer_blocks=single_transformer_blocks
        )

        # Create the training specification
        return TrainingSpec(
            model=config["model"],
            seed=config["seed"],
            steps=config["steps"],
            guidance=config["guidance"],
            quantize=config["quantize"],
            width=config["width"],
            height=config["height"],
            training_loop=training_loop,
            optimizer=optimizer,
            saver=saver,
            instrumentation=instrumentation,
            lora_layers=lora_layers,
            statistics=statistics,
            examples=examples,
            config_path=None if absolute_config_path is None else str(absolute_config_path),
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
    def _resolve_output_path(path: str, new_folder: bool) -> str:
        requested_path = os.path.expanduser(path)
        now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if new_folder and os.path.exists(requested_path):
            requested_path = f"{requested_path}_{now_str}"

        os.makedirs(requested_path, exist_ok=True)
        return requested_path

    @staticmethod
    def _from_checkpoint(path: str, new_folder: bool) -> "TrainingSpec":
        checkpoint = ZipUtil.unzip(path, "checkpoint.json", lambda x: json.load(open(x, "r")))
        config = ZipUtil.unzip(path, checkpoint["files"]["config"], lambda x: json.load(open(x, "r")))
        spec = TrainingSpec.from_conf(config, None, new_folder)
        spec.optimizer.state_path = checkpoint["files"]["optimizer"]
        spec.lora_layers.state_path = checkpoint["files"]["lora_adapter"]
        spec.training_loop.iterator_state_path = checkpoint["files"]["iterator"]
        spec.statistics.state_path = checkpoint["files"]["loss"]
        spec.checkpoint_path = path
        spec.config_path = None
        return spec
