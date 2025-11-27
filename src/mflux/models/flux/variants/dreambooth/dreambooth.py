from mlx import nn
from tqdm import tqdm

from mflux.models.common.config.config import Config
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.flux.variants.dreambooth.optimization.dreambooth_loss import DreamBoothLoss
from mflux.models.flux.variants.dreambooth.state.training_spec import TrainingSpec
from mflux.models.flux.variants.dreambooth.state.training_state import TrainingState
from mflux.models.flux.variants.dreambooth.statistics.plotter import Plotter
from mflux.models.flux.variants.txt2img.flux import Flux1


class DreamBooth:
    @staticmethod
    def train(
        flux: Flux1,
        config: Config,
        training_spec: TrainingSpec,
        training_state: TrainingState,
    ):
        # Freeze the model (LoRA layers are already applied to transformer in from_spec)
        flux.freeze()

        # Unfreeze LoRA layers so they can be trained
        DreamBooth._unfreeze_lora_layers(flux.transformer)

        # Define loss computation as a function of a batch 'b'
        train_step_function = nn.value_and_grad(
            model=flux,
            fn=lambda b: DreamBoothLoss.compute_loss(flux, config, b),
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
                validation_loss = DreamBoothLoss.compute_loss(flux, config, validation_batch)
                training_state.statistics.append_values(step=training_state.iterator.num_iterations, loss=validation_loss)  # fmt: off
                Plotter.update_loss_plot(training_spec=training_spec, training_state=training_state)
                del validation_loss

            # Generate a test image from the model periodically
            if training_state.should_generate_image(training_spec):
                image = flux.generate_image(
                    seed=training_spec.seed,
                    prompt=training_spec.instrumentation.validation_prompt,
                )
                image.save(path=training_state.get_current_validation_image_path(training_spec))
                del image
                flux.prompt_cache = {}

            # Save checkpoint periodically
            if training_state.should_save(training_spec):
                training_state.save(flux, training_spec)

        # Save the final state
        training_state.save(flux, training_spec)

    @staticmethod
    def _unfreeze_lora_layers(module: nn.Module) -> None:
        for name, child in module.named_modules():
            if isinstance(child, LoRALinear):
                child.unfreeze(keys=["lora_A", "lora_B"], strict=False)
