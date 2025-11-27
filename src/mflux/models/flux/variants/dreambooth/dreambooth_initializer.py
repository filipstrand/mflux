import mlx.core.random as random

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.flux.variants.dreambooth.dataset.dataset import Dataset
from mflux.models.flux.variants.dreambooth.dataset.iterator import Iterator
from mflux.models.flux.variants.dreambooth.lora_layers.lora_layers import LoRALayers
from mflux.models.flux.variants.dreambooth.optimization.optimizer import Optimizer
from mflux.models.flux.variants.dreambooth.state.training_spec import TrainingSpec
from mflux.models.flux.variants.dreambooth.state.training_state import TrainingState
from mflux.models.flux.variants.dreambooth.statistics.statistics import Statistics
from mflux.models.flux.variants.txt2img.flux import Flux1


class DreamBoothInitializer:
    @staticmethod
    def initialize(
        config_path: str | None,
        checkpoint_path: str | None,
    ) -> tuple[Flux1, Config, TrainingSpec, TrainingState]:
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
        config = Config(
            model_config=model_config,
            num_inference_steps=training_spec.steps,
            width=training_spec.width,
            height=training_spec.height,
            guidance=training_spec.guidance,
        )

        # Create the optimizer
        optimizer = Optimizer.from_spec(training_spec)

        # Create and apply the LoRA layers directly to the transformer
        LoRALayers.from_spec(flux=flux, training_spec=training_spec)

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
            iterator=iterator,
            statistics=statistics,
        )

        return flux, config, training_spec, training_state
