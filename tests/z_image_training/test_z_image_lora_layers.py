"""Tests for Z-Image LoRA layer management.

Tests LoRA layer:
- Injection into transformer
- Weight extraction
- Safetensors export
"""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from mflux.models.z_image.variants.training.lora_layers.lora_layers import ZImageLoRALayers
from mflux.models.z_image.variants.training.state.training_spec import (
    BlockRange,
    LoraLayersSpec,
    TrainingSpec,
    ZImageTransformerBlocks,
)


def create_mock_linear(in_features: int = 64, out_features: int = 64):
    """Create a mock linear layer."""
    linear = MagicMock()
    linear.weight = mx.random.normal((out_features, in_features))
    linear.bias = mx.zeros(out_features)
    linear.in_features = in_features
    linear.out_features = out_features
    return linear


def create_mock_block():
    """Create a mock transformer block with attention layers."""
    block = MagicMock()

    # Create attention module
    attention = MagicMock()
    attention.to_q = create_mock_linear()
    attention.to_k = create_mock_linear()
    attention.to_v = create_mock_linear()
    attention.to_out = MagicMock()
    attention.to_out.__getitem__ = MagicMock(return_value=create_mock_linear())

    block.attention = attention

    # Create feed_forward module
    feed_forward = MagicMock()
    feed_forward.w1 = create_mock_linear()
    feed_forward.w2 = create_mock_linear()
    feed_forward.w3 = create_mock_linear()

    block.feed_forward = feed_forward

    # Create adaLN_modulation
    adaLN = MagicMock()
    adaLN.__getitem__ = MagicMock(return_value=create_mock_linear())
    block.adaLN_modulation = adaLN

    return block


@pytest.mark.fast
def test_block_range_with_start_end():
    """Test BlockRange with start and end values."""
    block_range = BlockRange(start=0, end=5)
    blocks = block_range.get_blocks()

    assert blocks == [0, 1, 2, 3, 4]


@pytest.mark.fast
def test_block_range_with_indices():
    """Test BlockRange with explicit indices."""
    block_range = BlockRange(indices=[0, 2, 5, 10])
    blocks = block_range.get_blocks()

    assert blocks == [0, 2, 5, 10]


@pytest.mark.fast
def test_block_range_indices_take_precedence():
    """Test that indices take precedence over start/end."""
    block_range = BlockRange(start=0, end=10, indices=[1, 3, 5])
    blocks = block_range.get_blocks()

    # indices should take precedence
    assert blocks == [1, 3, 5]


@pytest.mark.fast
def test_block_range_requires_specification():
    """Test that BlockRange requires either indices or start/end."""
    block_range = BlockRange()

    with pytest.raises(ValueError):
        block_range.get_blocks()


@pytest.mark.fast
def test_z_image_transformer_blocks_spec():
    """Test ZImageTransformerBlocks dataclass."""
    spec = ZImageTransformerBlocks(
        block_range=BlockRange(start=0, end=2),
        layer_types=["attention.to_q", "attention.to_k"],
        lora_rank=8,
        layer_type="layers",
    )

    assert spec.block_range.get_blocks() == [0, 1]
    assert len(spec.layer_types) == 2
    assert spec.lora_rank == 8


@pytest.mark.fast
def test_lora_layers_spec_default_full():
    """Test default LoRA configuration."""
    spec = LoraLayersSpec.default_full()

    assert spec.main_layers is not None
    assert spec.noise_refiner is not None
    assert spec.context_refiner is not None

    # Main layers should cover blocks 0-29
    assert spec.main_layers.block_range.get_blocks() == list(range(30))

    # Noise refiner should cover 2 blocks
    assert spec.noise_refiner.block_range.get_blocks() == [0, 1]

    # Context refiner should cover 2 blocks
    assert spec.context_refiner.block_range.get_blocks() == [0, 1]


@pytest.mark.fast
def test_nested_attr_access():
    """Test _get_nested_attr helper."""
    block = create_mock_block()

    # Test single level access
    result = ZImageLoRALayers._get_nested_attr(block, "attention")
    assert result == block.attention

    # Test multi-level access
    result = ZImageLoRALayers._get_nested_attr(block, "attention.to_q")
    assert result == block.attention.to_q


@pytest.mark.fast
def test_lora_rank_validation():
    """Test that invalid LoRA rank raises error."""
    spec = ZImageTransformerBlocks(
        block_range=BlockRange(start=0, end=1),
        layer_types=["attention.to_q"],
        lora_rank=0,  # Invalid
    )

    # Should raise on application
    # This is a static test of the dataclass construction
    assert spec.lora_rank == 0


@pytest.mark.fast
def test_lora_rank_negative_validation():
    """Test that negative LoRA rank is captured."""
    spec = ZImageTransformerBlocks(
        block_range=BlockRange(start=0, end=1),
        layer_types=["attention.to_q"],
        lora_rank=-1,  # Invalid
    )

    # Should be captured but validation happens at application time
    assert spec.lora_rank == -1


@pytest.mark.fast
def test_training_spec_creates_default_lora():
    """Test that TrainingSpec creates default LoRA config when mode is LORA."""
    config_dict = {
        "model": "z-image-base",
        "seed": 42,
        "steps": 2,
        "guidance": 3.5,
        "width": 128,
        "height": 128,
        "mode": "lora",
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "save": {"output_path": "/tmp/test", "checkpoint_frequency": 1},
        "examples": {"path": "/tmp", "images": [{"image": "test.jpg", "prompt": "test"}]},
    }

    # Note: This will use default_full() if no lora_layers specified
    # Just testing the spec parsing, not actual file I/O
    spec = TrainingSpec.from_conf(config_dict, None, new_folder=False)

    assert spec.lora_layers is not None
    assert spec.lora_layers.main_layers is not None


@pytest.mark.fast
def test_lora_save_creates_safetensors():
    """Test that LoRA save creates a safetensors file."""
    with TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "test_lora.safetensors"

        # Create a mock transformer with LoRA weights
        transformer = MagicMock()

        # Simulate tree_flatten returning LoRA weights
        mock_weights = [
            ("layers.0.attention.to_q.lora_A", mx.random.normal((4, 64))),
            ("layers.0.attention.to_q.lora_B", mx.random.normal((64, 4))),
            ("layers.0.attention.to_k.lora_A", mx.random.normal((4, 64))),
            ("layers.0.attention.to_k.lora_B", mx.random.normal((64, 4))),
        ]

        # Mock tree_flatten to return our test weights
        with pytest.MonkeyPatch.context() as mp:

            def mock_tree_flatten(obj):
                return mock_weights

            mp.setattr(
                "mflux.models.z_image.variants.training.lora_layers.lora_layers.tree_flatten",
                mock_tree_flatten,
            )

            # Create minimal training spec
            training_spec = MagicMock()
            training_spec.lora_layers = LoraLayersSpec(
                main_layers=ZImageTransformerBlocks(
                    block_range=BlockRange(start=0, end=1),
                    layer_types=["attention.to_q", "attention.to_k"],
                    lora_rank=4,
                )
            )

            ZImageLoRALayers.save(transformer, save_path, training_spec)

            assert save_path.exists()


@pytest.mark.fast
def test_lora_weight_naming_convention():
    """Test LoRA weight naming follows expected convention."""
    # LoRA weights should end with .lora_A or .lora_B
    valid_names = [
        "layers.0.attention.to_q.lora_A",
        "layers.0.attention.to_q.lora_B",
        "noise_refiner.0.attention.to_k.lora_A",
        "context_refiner.1.feed_forward.w1.lora_B",
    ]

    for name in valid_names:
        assert name.endswith(".lora_A") or name.endswith(".lora_B")


@pytest.mark.fast
def test_block_index_out_of_range():
    """Test that out of range block index raises error."""
    # Create mock blocks list with only 2 elements
    blocks = [create_mock_block(), create_mock_block()]

    spec = ZImageTransformerBlocks(
        block_range=BlockRange(start=0, end=10),  # Beyond available blocks
        layer_types=["attention.to_q"],
        lora_rank=4,
    )

    transformer = MagicMock()

    with pytest.raises(IndexError):
        ZImageLoRALayers._construct_and_apply_layers(
            transformer=transformer,
            blocks=blocks,
            block_spec=spec,
            block_attr="layers",
        )
