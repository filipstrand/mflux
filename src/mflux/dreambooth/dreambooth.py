from mlx import nn
from tqdm import tqdm

from mflux import Flux1
from mflux.config.runtime_config import RuntimeConfig
from mflux.dreambooth.optimization.dreambooth_loss import DreamBoothLoss
from mflux.dreambooth.state.training_spec import TrainingSpec
from mflux.dreambooth.state.training_state import TrainingState
from mflux.dreambooth.statistics.plotter import Plotter
from mflux.weights.weight_handler_lora import WeightHandlerLoRA


class DreamBooth:
    @staticmethod
    def train(
        flux: Flux1,
        runtime_config: RuntimeConfig,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ):
        # Freeze the model and assign the LoRA layers to the model
        flux.freeze()
        WeightHandlerLoRA.set_lora_layers(
            transformer_module=flux.transformer,
            lora_layers=training_state.lora_layers,
        )

        # Define loss computation as a function of a batch 'b'
        train_step_function = nn.value_and_grad(
            model=flux,
            fn=lambda b: DreamBoothLoss.compute_loss(flux, runtime_config, b),
        )

        # Setup progress bar
        batches = tqdm(
            training_state.iterator,
            total=training_state.iterator.total_number_of_steps(),
            initial=training_state.iterator.num_iterations,
        )

        # Training loop
        for batch in batches:
            # Perform one gradient update on the LoRA the weights
            loss, grads = train_step_function(batch)
            training_state.optimizer.optimizer.update(model=flux, gradients=grads)
            del loss, grads

            # Plot loss progress periodically
            if training_state.should_plot_loss(training_spec):
                validation_batch = training_state.iterator.get_validation_batch()
                validation_loss = DreamBoothLoss.compute_loss(flux, runtime_config, validation_batch)
                training_state.statistics.append_values(step=training_state.iterator.num_iterations, loss=validation_loss)  # fmt: off
                Plotter.update_loss_plot(training_spec=training_spec, training_state=training_state)
                del validation_loss

            # Generate a test image from the model periodically
            if training_state.should_generate_image(training_spec):
                image = flux.generate_image(
                    seed=training_spec.seed,
                    config=runtime_config.config,
                    prompt=training_spec.instrumentation.validation_prompt,
                )
                image.save(path=training_state.get_current_validation_image_path(training_spec))
                del image
                flux.prompt_cache = {}

            # Save checkpoint periodically
            if training_state.should_save(training_spec):
                training_state.save(training_spec)

        # Save the final state
        training_state.save(training_spec)
