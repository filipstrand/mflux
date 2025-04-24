from mflux.dreambooth.dreambooth import DreamBooth
from mflux.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.error.exceptions import StopTrainingException
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Finetune a LoRA adapter")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_training_arguments()
    args = parser.parse_args()

    # 1. Initialize the required resources
    flux, runtime_config, training_spec, training_state = DreamBoothInitializer.initialize(
        config_path=args.train_config,
        checkpoint_path=args.train_checkpoint,
    )

    # 2. Start the training process
    try:
        DreamBooth.train(
            flux=flux,
            runtime_config=runtime_config,
            training_spec=training_spec,
            training_state=training_state,
        )
    except StopTrainingException as stop_exc:
        training_state.save(training_spec)
        print(stop_exc)


if __name__ == "__main__":
    main()
