from mlx import nn

from mflux import Config, Flux1, ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.dreambooth.dreambooth_loss import DreamBoothLoss
from mflux.dreambooth.dreambooth_util import DreamBoothUtil
from mflux.dreambooth.finetuning_dataset import FineTuningDataset
from mflux.dreambooth.optimizer import Optimizer


def main():
    # Set up the model
    runtime_config = RuntimeConfig(
        model_config=ModelConfig.FLUX1_DEV,
        config=Config(
            num_inference_steps=20,
            width=1024,
            height=1024,
            guidance=4.0,
            training_seed=0,
        ),
    )
    flux = Flux1(model_config=runtime_config.model_config, quantize=None)
    flux.freeze()
    flux.set_lora_layer()

    # Get and prepare the fine-tuning dataset
    dataset = FineTuningDataset.load_from_disk("/Users/filipstrand/Desktop/dog")
    dataset.prepare_dataset(flux)

    # Set up the optimizer
    optimizer = Optimizer.setup_optimizer()

    # Define loss computation in terms of the batch item
    train_step_function = nn.value_and_grad(
        model=flux,
        fn=lambda batch_item: DreamBoothLoss.compute_loss(flux, runtime_config, batch_item)
    )  # fmt: off

    # Training loop
    for t, batch in enumerate(dataset, 1):
        loss, grads = train_step_function(batch)
        optimizer.update(flux, grads)
        DreamBoothUtil.track_progress(loss, t)
        DreamBoothUtil.save_incrementally(flux, t)

    # Save the final adapter
    DreamBoothUtil.save_adapter(flux, t)


if __name__ == "__main__":
    main()
