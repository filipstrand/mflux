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
    flux = Flux1(model_config=runtime_config.model_config, quantize=8)
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
    num_epochs = 10
    steps_per_epoch = 100
    batch_size = 1
    global_step = 0
    for epoch in range(num_epochs):
        dataset_iterator = dataset.get_iterator(batch_size)

        for step in range(steps_per_epoch):
            batch = next(dataset_iterator)
            loss, grads = train_step_function(batch)
            optimizer.update(flux, grads)

            global_step += 1
            DreamBoothUtil.track_progress(loss, global_step)
            DreamBoothUtil.save_incrementally(flux, global_step)

        print(f"Completed epoch {epoch + 1}/{num_epochs}")

    # Save the final adapter
    DreamBoothUtil.save_adapter(flux, global_step)


if __name__ == "__main__":
    main()
