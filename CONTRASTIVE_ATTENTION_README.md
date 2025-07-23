# Contrastive Concept Attention

This implementation adds **contrastive concept attention** to the FLUX concept attention system, designed to generate **sharper, more precise attention heatmaps** by using "background" as an anti-concept.

## The Problem

Standard concept attention often suffers from **attention leakage** - the heatmap shows not just the target concept (e.g., "dragon") but also activates surrounding areas, creating blurry boundaries and less precise localization.

## The Solution

**Contrastive attention** computes attention for both:
1. **Target concept** (e.g., "dragon") 
2. **Anti-concept** ("background")

Then subtracts them: `dragon_attention - background_attention`

This creates **sharper boundaries** by explicitly contrasting foreground vs background, enhanced with:
- **Spectral sharpening** using SVD to emphasize dominant attention modes
- **Temperature-scaled softmax** for sharper probability distributions

## Mathematical Foundation

The core operation is:
```python
# Semantic subtraction in embedding space
contrastive_scores = concept_similarities - background_similarities

# Spectral sharpening via SVD
U, Σ, V = svd(contrastive_scores)  
Σ_sharp = Σ ** sharpening_exponent  # Emphasize dominant modes
sharp_scores = U @ diag(Σ_sharp) @ V.T

# Temperature sharpening
final_attention = softmax(sharp_scores / temperature)
```

## Usage

### Basic Usage

```python
from mflux.community.concept_attention.flux_concept import Flux1Concept
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig

# Initialize model
flux_model = Flux1Concept(model_config=ModelConfig.dev())

# Generate with contrastive attention
result = flux_model.generate_contrastive_image(
    seed=42,
    prompt="a dragon breathing fire on a mountain",
    concept="dragon",
    config=Config(height=512, width=512, num_inference_steps=20),
    sharpening_exponent=2.0,  # Higher = sharper (try 1.5-5.0)
    temperature=0.1,          # Lower = sharper (try 0.01-1.0)
)

# Save results
result.image.save("generated_image.png")
result.concept_heatmap.save("sharp_dragon_heatmap.png")
```

### Test Script

Run the included test script to compare standard vs contrastive attention:

```bash
# Basic test
python test_contrastive_attention.py \
    --prompt "a dragon breathing fire" \
    --concept "dragon" \
    --seed 42

# Parameter sensitivity test
python test_contrastive_attention.py --test-params
```

## Parameters

### `sharpening_exponent` (default: 2.0)
Controls spectral sharpening intensity:
- **1.0**: No sharpening (standard SVD)
- **1.5-3.0**: Moderate sharpening (recommended)
- **>3.0**: Aggressive sharpening (may over-sharpen)

### `temperature` (default: 0.1)  
Controls softmax sharpening:
- **1.0**: Standard softmax
- **0.1-0.5**: Moderate sharpening (recommended)
- **<0.1**: Very sharp (may create artifacts)

## API Reference

### New Methods

Both `Flux1Concept` and `Flux1ConceptFromImage` now support:

```python
def generate_contrastive_image(
    self,
    seed: int,
    prompt: str,
    concept: str,
    config: Config,
    heatmap_layer_indices: list[int] | None = None,
    heatmap_timesteps: list[int] | None = None, 
    sharpening_exponent: float = 2.0,
    temperature: float = 0.1,
) -> GeneratedImage
```

### New Utility Methods

```python
from mflux.community.concept_attention.concept_util import ConceptUtil

# Create contrastive heatmap from attention data
heatmap = ConceptUtil.create_contrastive_heatmap(
    concept="dragon",
    attention_data=concept_attention_data,
    background_attention_data=background_attention_data,
    height=512, width=512,
    layer_indices=list(range(15, 19)),
    timesteps=list(range(20)),
    sharpening_exponent=2.0,
    temperature=0.1,
)
```

## Expected Results

The contrastive heatmaps should show:
- ✅ **Sharper boundaries** around the target concept
- ✅ **Reduced leakage** into surrounding areas  
- ✅ **Higher contrast** between foreground and background
- ✅ **More precise localization** of semantic concepts

## Implementation Details

The contrastive approach addresses three mathematical sources of attention leakage:

1. **Dot product similarity**: Creates natural correlation between similar patches
2. **Softmax distribution**: Inherently smooth, never produces hard zeros
3. **Multi-scale averaging**: Blurs across different layers and timesteps

By computing explicit contrast against "background" and applying spectral + temperature sharpening, we work **with** the model's semantic understanding rather than against it.

## Future Extensions

This approach could be extended to other concept pairs:
- "person" vs "empty street"
- "car" vs "road surface"  
- "bird" vs "sky"
- Custom anti-concepts for specific use cases 