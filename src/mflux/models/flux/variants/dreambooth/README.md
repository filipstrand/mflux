# Dreambooth fine-tuning (Flux)
This directory contains MFLUX’s Dreambooth / LoRA fine-tuning implementation for the FLUX family.

## Training configuration
To describe a training run, you need to provide a training configuration file which specifies details such as what training data to use and various parameters.

To try it out, one of the easiest ways is to start from the provided example configuration and modify the `examples` section to point to your dataset:
- `_example/train.json`

## Training example
A complete example (training configuration + dataset) is provided in this repository:
- `_example/train.json`
- `_example/images/`

To start a training run, run:

```sh
mflux-train --train-config _example/train.json
```

By default, this will train an adapter with images of size `512x512` with a batch size of 1 and can take up to several hours to fully complete depending on your machine.

During training, MFLUX will output training checkpoints with artifacts (weights, states) according to what is specified in the configuration file.

As specified in `train.json`, these files will be placed in a folder on the Desktop called `~/Desktop/train`, but this can of course be changed to any other path by adjusting the configuration.

All training artifacts will be saved as a self-contained zip file, which can later be pointed to in order to resume an existing training run.

To find the LoRA weights, simply unzip and look for the `adapter` safetensors file and use it like any other LoRA adapter.

## Resuming a training run
The training process will continue to run until each training example has been used `num_epochs` times. If you interrupt the process, you can resume from a checkpoint zip file, e.g.:

```sh
mflux-train --train-checkpoint 0001000_checkpoint.zip
```

This feature has two helpful properties:
- Fully deterministic (given a specified `seed` in the training configuration)
- The complete training state (including optimizer state) is saved at each checkpoint

Because of these, MFLUX can resume a training run from a previous checkpoint and produce results that are *exactly* identical to a training run which was never interrupted.

*⚠️ Note: Everything but the dataset itself is contained within the zip file. The zip file contains configuration files which point to the original dataset, so make sure that it is in the same place when resuming.*

*⚠️ Note: A training run can only be resumed if it has not yet been completed (only checkpoints from interrupted runs can be resumed).*

## Memory issues
Fine-tuning can be memory intensive. To reduce memory requirements, consider:
- Use a quantized base model by setting `"quantize": 4` or `"quantize": 8`
- Skip some trainable layers
- Use a lower `rank` value for the LoRA matrices
- Don’t train all layers (limit block ranges)
- Use a smaller batch size (e.g. `"batch_size": 1`)
- Close other memory-heavy apps

Applying some of these strategies (like how the example `train.json` is set up) can allow a 32GB M1 Pro to perform a successful fine-tuning run.

