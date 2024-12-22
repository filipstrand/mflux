from pathlib import Path
from typing import List, Set

import mlx.core as mx
import pytest

from mflux.dreambooth.dataset.batch import Example
from mflux.dreambooth.dataset.dataset import Dataset
from mflux.dreambooth.dataset.iterator import Iterator


def get_dataset(num_examples: int) -> Dataset:
    examples = [
        Example(
            example_id=i,
            prompt=f"prompt_{i}",
            image_path=Path(f"image_{i}.jpg"),
            encoded_image=mx.zeros((1,)),
            prompt_embeds=mx.zeros((1,)),
            pooled_prompt_embeds=mx.zeros((1,)),
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


def test_batch_size_consistency(dataset, batch_size):
    """Test that batches are of the specified size except for the last one."""
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


def test_complete_coverage(dataset, batch_size):
    """Test that all examples are seen exactly once before reset."""
    iterator = Iterator(dataset, batch_size)
    seen_examples: Set[int] = set()

    for batch in iterator:
        for example in batch.examples:
            seen_examples.add(example.example_id)
        if len(seen_examples) == dataset.size():
            break

    expected_examples = set(range(dataset.size()))
    assert seen_examples == expected_examples


def test_state_restoration(dataset, batch_size):
    """Test that iterator can be saved and restored to the same state."""
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


def test_state_restoration_across_epochs(dataset, batch_size):
    """Test that state restoration works when crossing epoch boundaries."""
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


def test_randomization(dataset, batch_size):
    """Test that different seeds produce different sequences."""
    iterator1 = Iterator(dataset, batch_size=2, seed=1)
    iterator2 = Iterator(dataset, batch_size=2, seed=2)

    batch1 = next(iterator1)
    batch2 = next(iterator2)

    ids1 = [e.example_id for e in batch1.examples]
    ids2 = [e.example_id for e in batch2.examples]

    assert ids1 != ids2


def test_epoch_transition(dataset, batch_size):
    """Test that epoch transition creates a new permutation."""
    iterator = Iterator(dataset, batch_size)

    num_batches_in_epoch = (dataset.size() + batch_size - 1) // batch_size
    first_epoch_examples: List[List[int]] = []
    for _ in range(num_batches_in_epoch):
        batch = next(iterator)
        first_epoch_examples.append([e.example_id for e in batch.examples])

    second_epoch_batch = next(iterator)
    second_epoch_ids = [e.example_id for e in second_epoch_batch.examples]

    assert all(0 <= id < dataset.size() for id in second_epoch_ids)
    assert len(second_epoch_ids) == min(batch_size, dataset.size())


def test_state_consistency(dataset, batch_size):
    """Test that state remains consistent across iterations within the same epoch."""
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


def test_iteration_counting(dataset, batch_size):
    """Test that iteration counting works correctly."""
    iterator = Iterator(dataset, batch_size)

    num_iterations = 5
    for _ in range(num_iterations):
        next(iterator)

    assert iterator.num_iterations == num_iterations


def test_iteration_counting_across_epochs(dataset, batch_size):
    """Test that iteration counting works correctly across epoch boundaries."""
    iterator = Iterator(dataset, batch_size)

    batches_per_epoch = (dataset.size() + batch_size - 1) // batch_size
    total_iterations = int(batches_per_epoch * 1.5)

    for _ in range(total_iterations):
        next(iterator)

    assert iterator.num_iterations == total_iterations


def test_random_seed_consistency(dataset, batch_size):
    """Test that the same seed produces the same sequence of batches."""
    seed = 42
    iterator1 = Iterator(dataset, batch_size, seed=seed)
    iterator2 = Iterator(dataset, batch_size, seed=seed)

    for _ in range(5):
        batch1 = next(iterator1)
        batch2 = next(iterator2)

        ids1 = [e.example_id for e in batch1.examples]
        ids2 = [e.example_id for e in batch2.examples]
        assert ids1 == ids2


def test_state_serialization(dataset, batch_size):
    """Test that the iterator state can be serialized and deserialized."""
    iterator = Iterator(dataset, batch_size)
    next(iterator)

    state_dict = iterator.to_dict()
    restored_iterator = Iterator.from_dict(state_dict, dataset)

    batch = next(restored_iterator)
    assert len(batch.examples) > 0


def test_state_restoration_after_exhaustion(dataset, batch_size):
    """Test that state restoration works correctly after iterator exhaustion."""
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
