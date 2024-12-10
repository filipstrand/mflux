from mflux.dreambooth.dreambooth import DreamBooth
from mflux.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.error.exceptions import StopTrainingException
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Finetune a LoRA adapter")
    parser.add_model_arguments(require_model_arg=False)
    parser.add_training_arguments()
    args = parser.parse_args()

    flux, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
        config_path=args.train_config,
        checkpoint_path=args.train_checkpoint
    )  # fmt: off

    try:
        DreamBooth.train(
            flux=flux,
            runtime_config=runtime_config,
            training_spec=training_spec,
            training_state=training_state
        )  # fmt: off
    except StopTrainingException as stop_exc:
        training_state.save(training_spec)
        print(stop_exc)


if __name__ == "__main__":
    main()
