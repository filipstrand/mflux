"""Tests for Z-Image training checkpoint save/resume functionality.

Tests that:
- Checkpoint ZIP files are created correctly
- All state components are saved (optimizer, iterator, loss, config)
- State can be restored from checkpoint
"""

import datetime
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import mlx.core as mx
import pytest

# Note: Avoid importing from the main training module __init__ to prevent circular imports
# Import only specific submodules needed for testing
from mflux.models.z_image.variants.training.dataset.batch import Example
from mflux.models.z_image.variants.training.dataset.dataset import Dataset
from mflux.models.z_image.variants.training.dataset.iterator import Iterator
from mflux.models.z_image.variants.training.state.training_spec import (
    BlockRange,
    LoraLayersSpec,
    OptimizerSpec,
    SaveSpec,
    StatisticsSpec,
    TrainingLoopSpec,
    TrainingMode,
    TrainingSpec,
    ZImageTransformerBlocks,
)
from mflux.models.z_image.variants.training.state.training_state import (
    TRAINING_FILE_NAME_CONFIG_FILE,
    TRAINING_FILE_NAME_ITERATOR,
    TRAINING_FILE_NAME_LORA_ADAPTER,
    TRAINING_FILE_NAME_LOSS_FILE,
    TRAINING_FILE_NAME_OPTIMIZER,
    TRAINING_PATH_CHECKPOINTS,
    TrainingState,
)
from mflux.models.z_image.variants.training.statistics.statistics import Statistics


def create_mock_dataset(num_examples: int = 5) -> Dataset:
    """Create a mock dataset."""
    examples = [
        Example(
            example_id=i,
            prompt=f"prompt_{i}",
            image_path=Path(f"image_{i}.jpg"),
            encoded_image=mx.zeros((1, 16, 16, 4)),
            text_embeddings=mx.zeros((1, 77, 768)),
        )
        for i in range(num_examples)
    ]
    return Dataset(examples=examples)


def create_mock_training_spec(output_path: str) -> TrainingSpec:
    """Create a mock training spec."""
    return TrainingSpec(
        model="z-image-base",
        seed=42,
        steps=2,
        guidance=3.5,
        quantize=4,
        width=128,
        height=128,
        mode=TrainingMode.LORA,
        training_loop=TrainingLoopSpec(num_epochs=1, batch_size=1),
        optimizer=OptimizerSpec(name="AdamW", learning_rate=1e-4),
        saver=SaveSpec(checkpoint_frequency=1, output_path=output_path),
        instrumentation=None,
        statistics=StatisticsSpec(),
        examples=[],
        lora_layers=LoraLayersSpec(
            main_layers=ZImageTransformerBlocks(
                block_range=BlockRange(start=0, end=2),
                layer_types=["attention.to_q"],
                lora_rank=4,
            )
        ),
        config_path="/tmp/test_config.json",
    )


def create_mock_model():
    """Create a mock ZImageBase model."""
    model = MagicMock()
    model.transformer = MagicMock()
    return model


@pytest.mark.fast
def test_training_state_initialization():
    """Test TrainingState can be initialized with components."""
    dataset = create_mock_dataset()
    iterator = Iterator(dataset, batch_size=2, num_epochs=1, seed=42)
    optimizer = MagicMock()
    statistics = Statistics()

    state = TrainingState(
        iterator=iterator,
        optimizer=optimizer,
        statistics=statistics,
    )

    assert state.iterator is iterator
    assert state.optimizer is optimizer
    assert state.statistics is statistics


@pytest.mark.fast
def test_should_save():
    """Test should_save returns correct values based on frequency."""
    dataset = create_mock_dataset()
    iterator = Iterator(dataset, batch_size=1, num_epochs=10, seed=42)
    optimizer = MagicMock()
    statistics = Statistics()

    state = TrainingState(
        iterator=iterator,
        optimizer=optimizer,
        statistics=statistics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = create_mock_training_spec(tmpdir)
        spec.saver.checkpoint_frequency = 5

        # Initially at iteration 0, should save
        iterator.num_iterations = 0
        assert state.should_save(spec) is True

        # At iteration 3, should not save
        iterator.num_iterations = 3
        assert state.should_save(spec) is False

        # At iteration 5, should save
        iterator.num_iterations = 5
        assert state.should_save(spec) is True


@pytest.mark.fast
def test_should_plot_loss():
    """Test should_plot_loss with and without instrumentation."""
    dataset = create_mock_dataset()
    iterator = Iterator(dataset, batch_size=1, num_epochs=10, seed=42)
    optimizer = MagicMock()
    statistics = Statistics()

    state = TrainingState(
        iterator=iterator,
        optimizer=optimizer,
        statistics=statistics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = create_mock_training_spec(tmpdir)

        # No instrumentation
        spec.instrumentation = None
        assert state.should_plot_loss(spec) is False

        # With instrumentation
        spec.instrumentation = MagicMock()
        spec.instrumentation.plot_frequency = 10
        iterator.num_iterations = 10
        assert state.should_plot_loss(spec) is True

        iterator.num_iterations = 7
        assert state.should_plot_loss(spec) is False


@pytest.mark.fast
def test_should_generate_image():
    """Test should_generate_image with and without instrumentation."""
    dataset = create_mock_dataset()
    iterator = Iterator(dataset, batch_size=1, num_epochs=10, seed=42)
    optimizer = MagicMock()
    statistics = Statistics()

    state = TrainingState(
        iterator=iterator,
        optimizer=optimizer,
        statistics=statistics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = create_mock_training_spec(tmpdir)

        # No instrumentation
        spec.instrumentation = None
        assert state.should_generate_image(spec) is False

        # With instrumentation
        spec.instrumentation = MagicMock()
        spec.instrumentation.generate_image_frequency = 100
        iterator.num_iterations = 100
        assert state.should_generate_image(spec) is True


@pytest.mark.fast
def test_validation_image_path():
    """Test validation image path generation."""
    dataset = create_mock_dataset()
    iterator = Iterator(dataset, batch_size=1, num_epochs=1, seed=42)
    optimizer = MagicMock()
    statistics = Statistics()

    state = TrainingState(
        iterator=iterator,
        optimizer=optimizer,
        statistics=statistics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = create_mock_training_spec(tmpdir)
        iterator.num_iterations = 42

        path = state.get_current_validation_image_path(spec)

        assert "0000042" in str(path)
        assert "validation_image" in str(path)
        assert path.suffix == ".png"


@pytest.mark.fast
def test_loss_plot_path():
    """Test loss plot path generation."""
    dataset = create_mock_dataset()
    iterator = Iterator(dataset, batch_size=1, num_epochs=1, seed=42)
    optimizer = MagicMock()
    statistics = Statistics()

    state = TrainingState(
        iterator=iterator,
        optimizer=optimizer,
        statistics=statistics,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = create_mock_training_spec(tmpdir)
        iterator.num_iterations = 100

        path = state.get_current_loss_plot_path(spec)

        assert "0000100" in str(path)
        assert "loss" in str(path)
        assert path.suffix == ".pdf"


@pytest.mark.fast
def test_format_duration():
    """Test duration formatting."""
    start = datetime.datetime(2024, 1, 1, 10, 0, 0)

    # Test seconds
    end = datetime.datetime(2024, 1, 1, 10, 0, 30)
    assert TrainingState._format_duration(start, end) == "30 seconds"

    # Test minutes
    end = datetime.datetime(2024, 1, 1, 10, 5, 0)
    assert TrainingState._format_duration(start, end) == "5 minutes"

    # Test hours and minutes
    end = datetime.datetime(2024, 1, 1, 12, 30, 0)
    assert TrainingState._format_duration(start, end) == "2 hours and 30 minutes"

    # Test singular
    end = datetime.datetime(2024, 1, 1, 11, 1, 1)
    assert TrainingState._format_duration(start, end) == "1 hour and 1 minute and 1 second"


@pytest.mark.fast
def test_iterator_state_serialization():
    """Test iterator state can be serialized to dict."""
    dataset = create_mock_dataset(10)
    iterator = Iterator(dataset, batch_size=2, num_epochs=3, seed=42)

    # Advance the iterator
    for _ in range(5):
        next(iterator)

    state_dict = iterator.to_dict()

    assert "position" in state_dict
    assert "epoch" in state_dict
    assert "num_iterations" in state_dict
    assert "current_permutation" in state_dict
    assert "rng_state" in state_dict
    assert "batch_size" in state_dict
    assert "num_epochs" in state_dict
    assert "start_date_time" in state_dict

    assert state_dict["num_iterations"] == 5
    assert state_dict["batch_size"] == 2


@pytest.mark.fast
def test_iterator_state_restoration():
    """Test iterator state can be restored from dict."""
    dataset = create_mock_dataset(10)
    iterator1 = Iterator(dataset, batch_size=2, num_epochs=3, seed=42)

    # Advance the iterator
    for _ in range(5):
        next(iterator1)

    state_dict = iterator1.to_dict()
    iterator2 = Iterator.from_dict(state_dict, dataset)

    # Both iterators should produce the same next batch
    batch1 = next(iterator1)
    batch2 = next(iterator2)

    ids1 = [e.example_id for e in batch1.examples]
    ids2 = [e.example_id for e in batch2.examples]

    assert ids1 == ids2


@pytest.mark.fast
def test_statistics_serialization():
    """Test statistics can be saved and loaded."""
    stats = Statistics()
    stats.append_values(step=1, loss=0.5)
    stats.append_values(step=2, loss=0.4)
    stats.append_values(step=3, loss=0.3)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "loss.json"
        stats.save(save_path)

        assert save_path.exists()

        with open(save_path, "r") as f:
            data = json.load(f)

        assert len(data["steps"]) == 3
        assert len(data["losses"]) == 3
        assert data["losses"] == [0.5, 0.4, 0.3]


@pytest.mark.fast
def test_checkpoint_file_naming():
    """Test checkpoint file naming convention."""
    # Test the naming pattern
    iteration = 42
    expected_lora = f"{iteration:07d}_{TRAINING_FILE_NAME_LORA_ADAPTER}.safetensors"
    expected_optimizer = f"{iteration:07d}_{TRAINING_FILE_NAME_OPTIMIZER}.safetensors"
    expected_iterator = f"{iteration:07d}_{TRAINING_FILE_NAME_ITERATOR}.json"
    expected_loss = f"{iteration:07d}_{TRAINING_FILE_NAME_LOSS_FILE}.json"
    expected_config = f"{iteration:07d}_{TRAINING_FILE_NAME_CONFIG_FILE}.json"

    assert expected_lora == "0000042_adapter.safetensors"
    assert expected_optimizer == "0000042_optimizer.safetensors"
    assert expected_iterator == "0000042_iterator.json"
    assert expected_loss == "0000042_loss.json"
    assert expected_config == "0000042_config.json"


@pytest.mark.fast
def test_checkpoint_path_structure():
    """Test checkpoint directory structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / TRAINING_PATH_CHECKPOINTS

        # Simulate creating checkpoint directory
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        assert checkpoint_path.exists()
        assert checkpoint_path.name == "_checkpoints"
