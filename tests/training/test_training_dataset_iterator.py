from pathlib import Path
from typing import List, Set

import mlx.core as mx
import pytest

from mflux.models.common.training.dataset.batch import DataItem
from mflux.models.common.training.dataset.dataset import Dataset
from mflux.models.common.training.dataset.iterator import Iterator


def get_dataset(num_items: int) -> Dataset:
    data_items = [
        DataItem(
            data_id=i,
            prompt=f"prompt_{i}",
            image_path=Path(f"image_{i}.jpg"),
            clean_latents=mx.zeros((1,)),
            cond={"dummy": mx.zeros((1,))},
            width=512,
            height=512,
        )
        for i in range(num_items)
    ]
    return Dataset(data=data_items)


@pytest.fixture(params=[4, 5, 8, 25, 103])
def dataset(request):
    return get_dataset(request.param)


@pytest.fixture(params=[1, 2, 3])
def batch_size(request):
    return request.param


@pytest.mark.fast
def test_batch_size_consistency(dataset, batch_size):
    iterator = Iterator(dataset, batch_size)

    batch_sizes = []
    data_ids = set()

    for batch in iterator:
        batch_sizes.append(len(batch.data))
        for item in batch.data:
            data_ids.add(item.data_id)

        if len(data_ids) == dataset.size():
            break

    # Check all batches except the last are of specified size
    assert all(size == batch_size for size in batch_sizes[:-1])
    # Check last batch handles remainder
    assert batch_sizes[-1] == dataset.size() % batch_size or batch_size


@pytest.mark.fast
def test_complete_coverage(dataset, batch_size):
    iterator = Iterator(dataset, batch_size)
    seen_data: Set[int] = set()

    for batch in iterator:
        for item in batch.data:
            seen_data.add(item.data_id)
        if len(seen_data) == dataset.size():
            break

    expected_data = set(range(dataset.size()))
    assert seen_data == expected_data


@pytest.mark.fast
def test_state_restoration(dataset, batch_size):
    iterator1 = Iterator(dataset, batch_size)

    first_batches = []
    num_batches_to_compare = 3
    for _ in range(num_batches_to_compare):
        batch = next(iterator1)
        first_batches.append([e.data_id for e in batch.data])

    state_dict = iterator1.to_dict()
    iterator2 = Iterator.from_dict(state_dict, dataset)

    for _ in range(num_batches_to_compare):
        batch1 = next(iterator1)
        batch2 = next(iterator2)
        ids1 = [e.data_id for e in batch1.data]
        ids2 = [e.data_id for e in batch2.data]
        assert ids1 == ids2


@pytest.mark.fast
def test_state_restoration_across_epochs(dataset, batch_size):
    iterator1 = Iterator(dataset, batch_size)

    num_iterations = (dataset.size() + batch_size - 1) // batch_size + 1
    for _ in range(num_iterations):
        next(iterator1)

    state_dict = iterator1.to_dict()
    iterator2 = Iterator.from_dict(state_dict, dataset)

    batch1 = next(iterator1)
    batch2 = next(iterator2)
    ids1 = [e.data_id for e in batch1.data]
    ids2 = [e.data_id for e in batch2.data]
    assert ids1 == ids2


@pytest.mark.fast
def test_randomization(dataset, batch_size):
    iterator1 = Iterator(dataset, batch_size=2, seed=1)
    iterator2 = Iterator(dataset, batch_size=2, seed=2)

    batch1 = next(iterator1)
    batch2 = next(iterator2)

    ids1 = [e.data_id for e in batch1.data]
    ids2 = [e.data_id for e in batch2.data]

    assert ids1 != ids2


@pytest.mark.fast
def test_epoch_transition(dataset, batch_size):
    iterator = Iterator(dataset, batch_size)

    num_batches_in_epoch = (dataset.size() + batch_size - 1) // batch_size
    first_epoch_data: List[List[int]] = []
    for _ in range(num_batches_in_epoch):
        batch = next(iterator)
        first_epoch_data.append([e.data_id for e in batch.data])

    second_epoch_batch = next(iterator)
    second_epoch_ids = [e.data_id for e in second_epoch_batch.data]

    assert all(0 <= id < dataset.size() for id in second_epoch_ids)
    assert len(second_epoch_ids) == min(batch_size, dataset.size())


@pytest.mark.fast
def test_state_consistency(dataset, batch_size):
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
    num_epochs = 100
    iterator = Iterator(dataset, batch_size, num_epochs=num_epochs)

    total_items_seen = 0
    epochs_completed = 0
    data_per_epoch = set()

    try:
        while True:
            batch = next(iterator)
            batch_data = {e.data_id for e in batch.data}
            total_items_seen += len(batch.data)

            data_per_epoch.update(batch_data)

            if len(data_per_epoch) == dataset.size():
                epochs_completed += 1
                data_per_epoch = set()

    except StopIteration:
        pass

    assert epochs_completed == num_epochs
    assert total_items_seen == dataset.size() * num_epochs


@pytest.mark.fast
def test_iteration_counting(dataset, batch_size):
    iterator = Iterator(dataset, batch_size)

    num_iterations = 5
    for _ in range(num_iterations):
        next(iterator)

    assert iterator.num_iterations == num_iterations


@pytest.mark.fast
def test_iteration_counting_across_epochs(dataset, batch_size):
    iterator = Iterator(dataset, batch_size)

    batches_per_epoch = (dataset.size() + batch_size - 1) // batch_size
    total_iterations = int(batches_per_epoch * 1.5)

    for _ in range(total_iterations):
        next(iterator)

    assert iterator.num_iterations == total_iterations


@pytest.mark.fast
def test_random_seed_consistency(dataset, batch_size):
    seed = 42
    iterator1 = Iterator(dataset, batch_size, seed=seed)
    iterator2 = Iterator(dataset, batch_size, seed=seed)

    for _ in range(5):
        batch1 = next(iterator1)
        batch2 = next(iterator2)

        ids1 = [e.data_id for e in batch1.data]
        ids2 = [e.data_id for e in batch2.data]
        assert ids1 == ids2


@pytest.mark.fast
def test_state_serialization(dataset, batch_size):
    iterator = Iterator(dataset, batch_size)
    next(iterator)

    state_dict = iterator.to_dict()
    restored_iterator = Iterator.from_dict(state_dict, dataset)

    batch = next(restored_iterator)
    assert len(batch.data) > 0


@pytest.mark.fast
def test_state_restoration_after_exhaustion(dataset, batch_size):
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
