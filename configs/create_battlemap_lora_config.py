#!/usr/bin/env python3
"""Create battlemap LoRA training config with examples embedded."""

import json
from pathlib import Path

# Load examples
examples_path = Path(__file__).parent / "battlemap_examples.json"
with open(examples_path) as f:
    examples_data = json.load(f)

# Create the full config
config = {
    "model": "z-image-base",
    "mode": "lora",
    "seed": 42,
    "steps": 50,
    "guidance": 3.5,
    "quantize": 8,  # 8-bit quantization for memory efficiency
    "width": 1024,
    "height": 1024,
    "training_loop": {
        "num_epochs": 2,  # Reduced from 15 due to 7.5x caption potency
        "batch_size": 16,  # Leverage 512GB unified memory
        "gradient_accumulation_steps": 2,  # Effective batch size 32
    },
    "optimizer": {
        "name": "AdamW",
        "learning_rate": 5e-5,  # Reduced to prevent overfitting on rich data
        "weight_decay": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.05,
        "scheduler_type": "cosine",
        "min_lr": 1e-6,
    },
    "lora_layers": {
        "main_layers": {
            "block_range": {"start": 0, "end": 30},
            "layer_types": [
                "attention.to_q",
                "attention.to_k",
                "attention.to_v",
                "attention.to_out.0",
                "feed_forward.w1",
                "feed_forward.w2",
                "feed_forward.w3",
            ],
            "lora_rank": 64,  # Higher rank for style/composition learning
        },
        "noise_refiner": {
            "block_range": {"start": 0, "end": 2},
            "layer_types": ["attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0"],
            "lora_rank": 32,
        },
        "context_refiner": {
            "block_range": {"start": 0, "end": 2},
            "layer_types": ["attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0"],
            "lora_rank": 32,
        },
    },
    "dataset": {
        "repeat_count": 1,  # No repetition needed - rich captions
        "enable_augmentation": True,
        "use_aspect_ratio_bucketing": True,
        "augmentation": {
            "enable_flip": True,  # Battlemaps can be flipped horizontally
            "enable_brightness": False,  # Keep lighting consistent
            "enable_contrast": False,
            "enable_color_jitter": False,
            "enable_rotation": False,  # Don't rotate battlemaps
        },
    },
    "validation": {
        "enabled": True,
        "validation_split": 0.05,  # 5% holdout (~296 images)
        "validation_seed": 42,
        "validation_frequency": 500,
    },
    "ema": {"enabled": True, "decay": 0.9999},
    "early_stopping": {"enabled": True, "patience": 3, "min_delta": 0.001},
    "save": {
        "checkpoint_frequency": 500,
        "output_path": "~/z_image_training/battlemap_lora",
        "keep_last_n_checkpoints": 3,
        "keep_best_n_checkpoints": 2,
        "best_checkpoint_metric": "validation_loss",
    },
    "instrumentation": {
        "plot_frequency": 100,
        "generate_image_frequency": 500,
        "validation_prompt": "top-down fantasy battlemap, stone dungeon with lava river, glowing crystals, gridless, high detail, vtt map",
        "negative_prompt": "blurry, low quality, distorted, grid lines",
        "guidance_scale": 3.5,
        "enable_memory_monitoring": True,
        "compute_clip_score": False,
    },
    # Embed the examples directly
    "examples": examples_data,
}

# Write config
output_path = Path(__file__).parent / "battlemap_lora_config.json"
with open(output_path, "w") as f:
    json.dump(config, f, indent=2)

print(f"Created config at: {output_path}")
print(f"File size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
print(f"Number of examples: {len(examples_data['images'])}")
print("\nKey settings:")
print(f"  - Model: {config['model']}")
print(f"  - Quantize: {config['quantize']}-bit")
print(f"  - Epochs: {config['training_loop']['num_epochs']}")
print(f"  - Batch size: {config['training_loop']['batch_size']}")
print(f"  - Gradient accumulation: {config['training_loop']['gradient_accumulation_steps']}")
print(
    f"  - Effective batch: {config['training_loop']['batch_size'] * config['training_loop']['gradient_accumulation_steps']}"
)
print(f"  - Learning rate: {config['optimizer']['learning_rate']}")
print(f"  - LoRA rank (main): {config['lora_layers']['main_layers']['lora_rank']}")
