"""Tests for Z-Image training dataset iterator.

Tests iterator state management, batching, epoch transitions, and state serialization.
Follows patterns from tests/dreambooth/test_dreambooth_dataset_iterator.py.
"""

from pathlib import Path
from typing import Set

import mlx.core as mx
import pytest

from mflux.models.z_image.variants.training.dataset.batch import Example
from mflux.models.z_image.variants.training.dataset.dataset import Dataset
from mflux.models.z_image.variants.training.dataset.iterator import Iterator


def get_dataset(num_examples: int) -> Dataset:
    """Create a mock dataset with the specified number of examples."""
    examples = [
        Example(
            example_id=i,
            prompt=f"prompt_{i}",
            image_path=Path(f"image_{i}.jpg"),
            encoded_image=mx.zeros((1, 16, 16, 4)),  # Simulated latent
            text_embeddings=mx.zeros((1, 77, 768)),  # Simulated text embedding
        )
        for i in range(num_examples)
    ]
    return Dataset(examples=examples)


@pytest.fixture(params=[4, 5, 8, 25, 103])
def dataset(request):
    return get_dataset(request.param)


@pytest.fixture(params=[1, 2, 3])
def batch_size(request):
    return request.param


@pytest.mark.fast
def test_batch_size_consistency(dataset, batch_size):
    """Test that all batches except the last have the specified size."""
    iterator = Iterator(dataset, batch_size)

    batch_sizes = []
    example_ids = set()

    for batch in iterator:
        batch_sizes.append(len(batch.examples))
        for example in batch.examples:
            example_ids.add(example.example_id)

        if len(example_ids) == dataset.size():
            break

    # Check all batches except the last are of specified size
    assert all(size == batch_size for size in batch_sizes[:-1])
    # Check last batch handles remainder
    assert batch_sizes[-1] == dataset.size() % batch_size or batch_size


@pytest.mark.fast
def test_complete_coverage(dataset, batch_size):
    """Test that all examples are covered in one epoch."""
    iterator = Iterator(dataset, batch_size)
    seen_examples: Set[int] = set()

    for batch in iterator:
        for example in batch.examples:
            seen_examples.add(example.example_id)
        if len(seen_examples) == dataset.size():
            break

    expected_examples = set(range(dataset.size()))
    assert seen_examples == expected_examples


@pytest.mark.fast
def test_state_restoration(dataset, batch_size):
    """Test that iterator state can be saved and restored."""
    iterator1 = Iterator(dataset, batch_size)

    first_batches = []
    num_batches_to_compare = 3
    for _ in range(num_batches_to_compare):
        batch = next(iterator1)
        first_batches.append([e.example_id for e in batch.examples])

    state_dict = iterator1.to_dict()
    iterator2 = Iterator.from_dict(state_dict, dataset)

    for _ in range(num_batches_to_compare):
        batch1 = next(iterator1)
        batch2 = next(iterator2)
        ids1 = [e.example_id for e in batch1.examples]
        ids2 = [e.example_id for e in batch2.examples]
        assert ids1 == ids2


@pytest.mark.fast
def test_state_restoration_across_epochs(dataset, batch_size):
    """Test state restoration works across epoch boundaries."""
    iterator1 = Iterator(dataset, batch_size)

    num_iterations = (dataset.size() + batch_size - 1) // batch_size + 1
    for _ in range(num_iterations):
        next(iterator1)

    state_dict = iterator1.to_dict()
    iterator2 = Iterator.from_dict(state_dict, dataset)

    batch1 = next(iterator1)
    batch2 = next(iterator2)
    ids1 = [e.example_id for e in batch1.examples]
    ids2 = [e.example_id for e in batch2.examples]
    assert ids1 == ids2


@pytest.mark.fast
def test_randomization(dataset, batch_size):
    """Test that different seeds produce different orderings."""
    iterator1 = Iterator(dataset, batch_size=2, seed=1)
    iterator2 = Iterator(dataset, batch_size=2, seed=2)

    batch1 = next(iterator1)
    batch2 = next(iterator2)

    ids1 = [e.example_id for e in batch1.examples]
    ids2 = [e.example_id for e in batch2.examples]

    assert ids1 != ids2


@pytest.mark.fast
def test_epoch_transition(dataset, batch_size):
    """Test that epoch transition resets position and reshuffles."""
    iterator = Iterator(dataset, batch_size)

    num_batches_in_epoch = (dataset.size() + batch_size - 1) // batch_size
    first_epoch_examples = []
    for _ in range(num_batches_in_epoch):
        batch = next(iterator)
        first_epoch_examples.append([e.example_id for e in batch.examples])

    second_epoch_batch = next(iterator)
    second_epoch_ids = [e.example_id for e in second_epoch_batch.examples]

    assert all(0 <= id < dataset.size() for id in second_epoch_ids)
    assert len(second_epoch_ids) == min(batch_size, dataset.size())


@pytest.mark.fast
def test_state_consistency(dataset, batch_size):
    """Test that state changes after iteration."""
    iterator = Iterator(dataset, batch_size)

    initial_state = iterator.to_dict()
    next(iterator)
    new_state = iterator.to_dict()

    assert initial_state != new_state

    iterator2 = Iterator.from_dict(initial_state, dataset)
    next(iterator2)
    restored_state = iterator2.to_dict()

    assert restored_state["position"] == new_state["position"]
    assert restored_state["epoch"] == new_state["epoch"]
    assert restored_state["num_iterations"] == new_state["num_iterations"]
    assert restored_state["current_permutation"] == new_state["current_permutation"]


@pytest.mark.fast
def test_fixed_num_epochs(dataset, batch_size):
    """Test that iterator stops after specified number of epochs."""
    num_epochs = 100
    iterator = Iterator(dataset, batch_size, num_epochs=num_epochs)

    total_examples_seen = 0
    epochs_completed = 0
    examples_per_epoch = set()

    try:
        while True:
            batch = next(iterator)
            batch_examples = {e.example_id for e in batch.examples}
            total_examples_seen += len(batch.examples)

            examples_per_epoch.update(batch_examples)

            if len(examples_per_epoch) == dataset.size():
                epochs_completed += 1
                examples_per_epoch = set()

    except StopIteration:
        pass

    assert epochs_completed == num_epochs
    assert total_examples_seen == dataset.size() * num_epochs


@pytest.mark.fast
def test_iteration_counting(dataset, batch_size):
    """Test that iteration counter increments correctly."""
    iterator = Iterator(dataset, batch_size)

    num_iterations = 5
    for _ in range(num_iterations):
        next(iterator)

    assert iterator.num_iterations == num_iterations


@pytest.mark.fast
def test_iteration_counting_across_epochs(dataset, batch_size):
    """Test iteration counting persists across epochs."""
    iterator = Iterator(dataset, batch_size)

    batches_per_epoch = (dataset.size() + batch_size - 1) // batch_size
    total_iterations = int(batches_per_epoch * 1.5)

    for _ in range(total_iterations):
        next(iterator)

    assert iterator.num_iterations == total_iterations


@pytest.mark.fast
def test_random_seed_consistency(dataset, batch_size):
    """Test that same seed produces same ordering."""
    seed = 42
    iterator1 = Iterator(dataset, batch_size, seed=seed)
    iterator2 = Iterator(dataset, batch_size, seed=seed)

    for _ in range(5):
        batch1 = next(iterator1)
        batch2 = next(iterator2)

        ids1 = [e.example_id for e in batch1.examples]
        ids2 = [e.example_id for e in batch2.examples]
        assert ids1 == ids2


@pytest.mark.fast
def test_state_serialization(dataset, batch_size):
    """Test state can be serialized and restored from dict."""
    iterator = Iterator(dataset, batch_size)
    next(iterator)

    state_dict = iterator.to_dict()
    restored_iterator = Iterator.from_dict(state_dict, dataset)

    batch = next(restored_iterator)
    assert len(batch.examples) > 0


@pytest.mark.fast
def test_state_restoration_after_exhaustion(dataset, batch_size):
    """Test that exhausted iterator state can be restored."""
    iterator = Iterator(dataset, batch_size, num_epochs=10)
    try:
        while True:
            next(iterator)
    except StopIteration:
        pass

    state_dict = iterator.to_dict()
    restored_iterator = Iterator.from_dict(state_dict, dataset)

    with pytest.raises(StopIteration):
        next(restored_iterator)


@pytest.mark.fast
def test_total_number_of_steps():
    """Test total_number_of_steps calculation."""
    dataset = get_dataset(10)
    iterator = Iterator(dataset, batch_size=2, num_epochs=3)

    # 10 examples * 3 epochs = 30 total examples to process
    assert iterator.total_number_of_steps() == 30


@pytest.mark.fast
def test_validation_batch():
    """Test get_validation_batch returns a subset of examples."""
    dataset = get_dataset(20)
    iterator = Iterator(dataset, batch_size=2, num_epochs=1)

    validation_batch = iterator.get_validation_batch()

    # Should be capped at VALIDATION_BATCH_MAX_SIZE (10)
    assert len(validation_batch.examples) <= 10
    assert all(isinstance(e, Example) for e in validation_batch.examples)
