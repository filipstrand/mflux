from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.flux.variants.dreambooth.dreambooth import DreamBooth
from mflux.models.flux.variants.dreambooth.dreambooth_initializer import DreamBoothInitializer
from mflux.utils.exceptions import StopTrainingException


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Finetune a LoRA adapter")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_training_arguments()
    args = parser.parse_args()

    # 1. Initialize the required resources
    flux, config, training_spec, training_state = DreamBoothInitializer.initialize(
        config_path=args.train_config,
        checkpoint_path=args.train_checkpoint,
    )

    # 2. Start the training process
    try:
        DreamBooth.train(
            flux=flux,
            config=config,
            training_spec=training_spec,
            training_state=training_state,
        )
    except StopTrainingException as stop_exc:
        training_state.save(flux, training_spec)
        print(stop_exc)


if __name__ == "__main__":
    main()
