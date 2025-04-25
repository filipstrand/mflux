import mlx.core.random as random

from mflux import Config, Flux1, ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.dreambooth.dataset.dataset import Dataset
from mflux.dreambooth.dataset.iterator import Iterator
from mflux.dreambooth.lora_layers.lora_layers import LoRALayers
from mflux.dreambooth.optimization.optimizer import Optimizer
from mflux.dreambooth.state.training_spec import TrainingSpec
from mflux.dreambooth.state.training_state import TrainingState
from mflux.dreambooth.statistics.statistics import Statistics


class DreamBoothInitializer:
    @staticmethod
    def initialize(
        config_path: str | None,
        checkpoint_path: str | None,
    ) -> tuple[Flux1, RuntimeConfig, TrainingSpec, TrainingState]:
        # The training specification describing the details of the training process. It is resolved
        # differently depending on if training starts from scratch or resumes from checkpoint.
        training_spec = TrainingSpec.resolve(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
        )

        # Set global random seed to make training deterministic
        random.seed(training_spec.seed)

        # Load the model
        model_config = ModelConfig.from_name(training_spec.model)
        flux = Flux1(
            model_config=model_config,
            quantize=training_spec.quantize,
        )
        runtime_config = RuntimeConfig(
            model_config=model_config,
            config=Config(
                num_inference_steps=training_spec.steps,
                width=training_spec.width,
                height=training_spec.height,
                guidance=training_spec.guidance,
            ),
        )

        # Create the optimizer
        optimizer = Optimizer.from_spec(training_spec)

        # Create the LoRA layers by matching them against the corresponding Flux layers
        lora_layers = LoRALayers.from_spec(flux=flux, training_spec=training_spec)

        # Prepare the fine-tuning dataset and create the iterator
        dataset = Dataset.prepare_dataset(
            flux=flux,
            raw_data=training_spec.examples,
            width=training_spec.width,
            height=training_spec.height,
        )
        iterator = Iterator.from_spec(
            training_spec=training_spec,
            dataset=dataset,
        )

        # Setup loss statistics
        statistics = Statistics.from_spec(training_spec=training_spec)

        # The training state consisting of everything that moves during training
        training_state = TrainingState(
            optimizer=optimizer,
            lora_layers=lora_layers,
            iterator=iterator,
            statistics=statistics,
        )

        return flux, runtime_config, training_spec, training_state
