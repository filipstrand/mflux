# Z-Image-Base Training Module

Native MLX training for Z-Image-Base on Apple Silicon, optimized for Mac Studio M3 Ultra with 512GB unified memory.

## Features

- **LoRA Training**: Efficient adapter training with configurable rank and target layers
- **Full Fine-tuning**: Complete model training leveraging 512GB unified memory
- **CFG Support**: Training with classifier-free guidance (3.0-5.0)
- **Memory Optimization**: Automatic batch size calculation and gradient accumulation
- **Pre-computed Embeddings**: ~30-40% speedup by caching VAE and text embeddings
- **Checkpoint Resume**: Full state persistence for training resumption

## Quick Start

### LoRA Training

```bash
mflux-train-z-image --train-config lora_config.json
```

Example `lora_config.json`:
```json
{
  "model": "z-image",
  "mode": "lora",
  "seed": 42,
  "steps": 50,
  "guidance": 3.5,
  "width": 1024,
  "height": 1024,
  "training_loop": {
    "num_epochs": 10,
    "batch_size": 16
  },
  "optimizer": {
    "name": "AdamW",
    "learning_rate": 1e-4
  },
  "lora_layers": {
    "main_layers": {
      "block_range": {"start": 0, "end": 30},
      "layer_types": ["attention.to_q", "attention.to_k", "attention.to_v", "attention.to_out.0"],
      "lora_rank": 32
    }
  },
  "save": {
    "checkpoint_frequency": 500,
    "output_path": "~/z_image_lora"
  },
  "examples": {
    "path": "./images",
    "images": [
      {"image": "img1.jpg", "prompt": "description 1"},
      {"image": "img2.jpg", "prompt": "description 2"}
    ]
  }
}
```

### Full Fine-tuning

```bash
mflux-train-z-image --train-config full_config.json
```

Example `full_config.json`:
```json
{
  "model": "z-image",
  "mode": "full",
  "seed": 42,
  "training_loop": {
    "num_epochs": 5,
    "batch_size": 8,
    "gradient_accumulation_steps": 4
  },
  "optimizer": {
    "name": "AdamW",
    "learning_rate": 1e-5
  },
  "full_finetune": {
    "train_transformer": true,
    "train_vae": false,
    "train_text_encoder": false
  }
}
```

### Resuming Training

```bash
mflux-train-z-image --train-checkpoint path/to/checkpoint.zip
```

## Memory Usage

### Mac Studio M3 Ultra (512GB)

| Mode | Batch Size | Memory Usage | Notes |
|------|------------|--------------|-------|
| LoRA (bf16) | 16 | ~100 GB | Recommended for style/character |
| LoRA (bf16) | 32 | ~170 GB | Maximum throughput |
| Full | 8 | ~200 GB | Recommended for domain adaptation |
| Full | 16 | ~320 GB | Maximum for full fine-tuning |

### Memory Optimization API

```python
from mflux.models.z_image.variants.training.optimization import MemoryOptimizer

# Get recommendations
MemoryOptimizer.print_recommendations(mode="lora", available_memory_gb=512)

# Calculate optimal batch size
optimal_batch, estimate = MemoryOptimizer.calculate_optimal_batch_size(
    mode="lora",
    available_memory_gb=512,
    quantize=None
)
```

## Architecture

Z-Image-Base architecture:
- **Transformer**: S3-DiT (Scalable Single-Stream DiT), 30 layers
- **Text Encoder**: Qwen3-4B
- **VAE**: Flux-derived, 16-channel latent space

### LoRA Targets

Available layer types for LoRA:
- `attention.to_q` - Query projection
- `attention.to_k` - Key projection
- `attention.to_v` - Value projection
- `attention.to_out.0` - Output projection
- `feed_forward.w1` - FFN gate projection
- `feed_forward.w2` - FFN down projection
- `feed_forward.w3` - FFN up projection
- `adaLN_modulation.0` - AdaLN modulation

Block types:
- `layers` (30 blocks) - Main transformer layers
- `noise_refiner` (2 blocks) - Noise refinement
- `context_refiner` (2 blocks) - Context refinement

## Training Loop

The training uses flow matching (rectified flow) loss:

1. Sample random timestep t ∈ [0, T]
2. Interpolate: `latent_t = (1-σ_t) * clean + σ_t * noise`
3. Predict noise: `predicted = transformer(latent_t, t, text_embeddings)`
4. Loss: `||clean + predicted - noise||²`

## Checkpoints

Checkpoints are saved as ZIP files containing:
- `adapter.safetensors` - LoRA weights (or full model for full fine-tuning)
- `optimizer.safetensors` - Optimizer state
- `iterator.json` - Dataset iteration state
- `loss.json` - Training statistics
- `config.json` - Training configuration
- `checkpoint.json` - Metadata

## CLI Reference

### mflux-train-z-image

Train Z-Image-Base model.

```
mflux-train-z-image [OPTIONS]

Options:
  --train-config PATH      Path to training config JSON
  --train-checkpoint PATH  Path to checkpoint ZIP (for resuming)
  --low-ram               Enable low-RAM mode (not typically needed with 512GB)
  -B, --battery-percentage-stop-limit INT
                          Stop on battery percentage (laptop mode)
```

### mflux-generate-z-image-base

Generate images with Z-Image-Base (CFG support).

```
mflux-generate-z-image-base [OPTIONS]

Options:
  --prompt TEXT           Text prompt
  --negative-prompt TEXT  Negative prompt (Z-Image-Base feature)
  --guidance FLOAT        CFG guidance scale (default: 3.5)
  --steps INT            Inference steps (default: 50)
  --width INT            Image width
  --height INT           Image height
  --lora-paths PATH      LoRA weights path
  --lora-scales FLOAT    LoRA scale (default: 1.0)
```

## Training Time Estimates

With optimized batch sizes on Mac Studio M3 Ultra:

| Dataset | LoRA (batch=16) | Full (batch=8) |
|---------|-----------------|----------------|
| 3k images | ~20-30 hours | ~2-3 days |
| 10k images | ~3-4 days | ~6-8 days |
| 100k images | ~29-43 days | ~2-3 months |

## Comparison with Cloud

| Dataset | Mac M3 Ultra | 8x H200 Cloud | Cost Savings |
|---------|--------------|---------------|--------------|
| 3k images | 1-2 days | 53 min | $25 saved |
| 10k images | 3-4 days | 2.75 hrs | $79 saved |
| 100k images | 29-43 days | 27 hrs | $775 saved |

Your Mac trades time for $0 training cost - ideal for iteration and smaller datasets.
