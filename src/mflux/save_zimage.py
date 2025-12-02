"""CLI entry point for mflux-save-zimage command.

Saves a quantized version of Z-Image-Turbo to disk for efficient loading.
"""

import argparse

from mflux.config.model_config import ModelConfig
from mflux.zimage import ZImage


def main():
    # Simple argument parser for Z-Image save
    parser = argparse.ArgumentParser(description="Save a quantized version of Z-Image-Turbo to disk.")
    parser.add_argument(
        "--path",
        type=str,
        required=True,
        help="Directory to save the quantized model",
    )
    parser.add_argument(
        "--quantize",
        "-q",
        type=int,
        choices=[3, 4, 5, 6, 8],
        default=None,
        help="Quantization bits (3, 4, 5, 6, 8). Default: None (fp16)",
    )
    args = parser.parse_args()

    # Status message
    if args.quantize is None:
        print("Saving Z-Image-Turbo (fp16, no quantization)...")
    else:
        print(f"Saving Z-Image-Turbo ({args.quantize}-bit quantization)...")

    # Load and quantize model
    zimage = ZImage(
        model_config=ModelConfig.zimage_turbo(),
        quantize=args.quantize,
    )

    # Save to disk
    print(f"Saving model to {args.path}...")
    zimage.save_model(args.path)
    print(f"Model saved successfully to {args.path}")


if __name__ == "__main__":
    main()
