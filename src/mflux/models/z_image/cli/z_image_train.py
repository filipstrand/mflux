"""CLI for Z-Image-Base training.

Supports both LoRA and full fine-tuning modes.

Usage:
    mflux-train-z-image --train-config config.json
    mflux-train-z-image --train-checkpoint checkpoint.zip

For LoRA training:
    mflux-train-z-image --train-config lora_config.json

For full fine-tuning:
    mflux-train-z-image --train-config full_config.json
"""

from mflux.cli.parser.parsers import CommandLineParser
from mflux.models.z_image.variants.training.trainer import ZImageTrainer
from mflux.models.z_image.variants.training.training_initializer import ZImageTrainingInitializer
from mflux.utils.exceptions import StopTrainingException


def main():
    # Parse command line arguments
    parser = CommandLineParser(description="Train Z-Image-Base with LoRA or full fine-tuning")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_training_arguments()
    args = parser.parse_args()

    # Initialize training resources
    print("Initializing Z-Image training...")
    model, config, training_spec, training_state = ZImageTrainingInitializer.initialize(
        config_path=args.train_config,
        checkpoint_path=args.train_checkpoint,
    )

    # Show memory estimate
    estimates = ZImageTrainingInitializer.estimate_memory_usage(training_spec)
    print("\n=== Memory Estimates ===")
    for key, value in estimates.items():
        print(f"  {key}: {value:.1f} GB")
    print("=" * 30)

    # Start training
    try:
        print("\nStarting training loop...")
        ZImageTrainer.train(
            model=model,
            config=config,
            training_spec=training_spec,
            training_state=training_state,
        )
    except StopTrainingException as stop_exc:
        training_state.save(model, training_spec)
        print(f"\nTraining stopped: {stop_exc}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        training_state.save(model, training_spec)
        print(f"Checkpoint saved to {training_spec.saver.output_path}")


if __name__ == "__main__":
    main()
