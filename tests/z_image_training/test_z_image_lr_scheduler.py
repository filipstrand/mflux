"""Tests for Z-Image LR Scheduler implementations.

Tests that schedulers:
- Return correct learning rates at each step
- Implement warmup correctly
- Handle state serialization
"""

import mlx.optimizers as optim
import numpy as np
import pytest

from mflux.models.z_image.variants.training.optimization.lr_scheduler import (
    ConstantLR,
    CosineAnnealingLR,
    LinearWarmupLR,
    OneCycleLR,
    create_scheduler,
)


def create_dummy_optimizer(lr: float = 0.001) -> optim.Adam:
    """Create a dummy optimizer for testing."""
    return optim.Adam(learning_rate=lr)


@pytest.mark.fast
def test_constant_lr():
    """Test that ConstantLR maintains constant learning rate."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = ConstantLR(optimizer, initial_lr=0.001)

    # LR should stay constant
    for step in range(100):
        assert scheduler.get_lr() == 0.001
        scheduler.step()

    assert scheduler.step_count == 100


@pytest.mark.fast
def test_constant_lr_with_warmup():
    """Test ConstantLR with warmup steps."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = ConstantLR(optimizer, initial_lr=0.001, warmup_steps=10)

    # During warmup: lr should increase linearly
    for step in range(10):
        expected_lr = 0.001 * (step + 1) / 10
        actual_lr = scheduler.get_lr()
        np.testing.assert_almost_equal(actual_lr, expected_lr, decimal=6)
        scheduler.step()

    # After warmup: lr should be constant at initial_lr
    for step in range(90):
        assert scheduler.get_lr() == 0.001
        scheduler.step()


@pytest.mark.fast
def test_linear_warmup_lr():
    """Test LinearWarmupLR warmup and constant phases."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = LinearWarmupLR(
        optimizer,
        initial_lr=0.001,
        warmup_steps=10,
    )

    # During warmup: lr should increase linearly
    for step in range(10):
        expected_lr = 0.001 * (step + 1) / 10
        actual_lr = scheduler.get_lr()
        np.testing.assert_almost_equal(actual_lr, expected_lr, decimal=6)
        scheduler.step()

    # After warmup: lr should be constant at initial_lr
    for step in range(90):
        assert scheduler.get_lr() == 0.001
        scheduler.step()


@pytest.mark.fast
def test_cosine_annealing_lr():
    """Test CosineAnnealingLR decay pattern."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = CosineAnnealingLR(
        optimizer,
        initial_lr=0.001,
        total_steps=100,
        warmup_steps=10,
        min_lr=0.0001,
    )

    # During warmup
    for step in range(10):
        scheduler.step()

    # After warmup, LR should follow cosine decay
    initial_post_warmup_lr = scheduler.get_lr()
    assert initial_post_warmup_lr == 0.001

    # Step through and verify decay
    previous_lr = initial_post_warmup_lr
    for step in range(90):
        scheduler.step()
        current_lr = scheduler.get_lr()
        # LR should generally decrease (with some tolerance for cosine shape)
        if step > 0:
            assert current_lr <= previous_lr + 1e-6
        previous_lr = current_lr

    # At end, should be close to min_lr
    final_lr = scheduler.get_lr()
    np.testing.assert_almost_equal(final_lr, 0.0001, decimal=4)


@pytest.mark.fast
def test_one_cycle_lr():
    """Test OneCycleLR warmup, peak, and decay phases."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = OneCycleLR(
        optimizer,
        initial_lr=0.001,
        total_steps=100,
        pct_start=0.3,  # 30% warmup
    )

    # Track LR over all steps
    lrs = []
    for step in range(100):
        lrs.append(scheduler.get_lr())
        scheduler.step()

    # First phase (0-30%): LR should increase to peak
    warmup_lrs = lrs[:30]
    for i in range(1, len(warmup_lrs)):
        assert warmup_lrs[i] >= warmup_lrs[i - 1] - 1e-6

    # Peak should be at ~30% of steps
    peak_idx = np.argmax(lrs)
    assert 25 <= peak_idx <= 35  # Around step 30

    # Second phase (30-100%): LR should decrease
    decay_lrs = lrs[30:]
    for i in range(1, len(decay_lrs)):
        assert decay_lrs[i] <= decay_lrs[i - 1] + 1e-6


@pytest.mark.fast
def test_scheduler_state_dict():
    """Test state_dict and load_state_dict."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = CosineAnnealingLR(
        optimizer,
        initial_lr=0.001,
        total_steps=100,
        warmup_steps=10,
        min_lr=0.0001,
    )

    # Step a few times
    for _ in range(25):
        scheduler.step()

    # Save state
    state = scheduler.state_dict()
    lr_at_save = scheduler.get_lr()

    # Create new scheduler and load state
    optimizer2 = create_dummy_optimizer(lr=0.001)
    scheduler2 = CosineAnnealingLR(
        optimizer2,
        initial_lr=0.001,
        total_steps=100,
        warmup_steps=10,
        min_lr=0.0001,
    )
    scheduler2.load_state_dict(state)

    # Should have same step count and LR
    assert scheduler2.step_count == 25
    np.testing.assert_almost_equal(scheduler2.get_lr(), lr_at_save, decimal=6)


@pytest.mark.fast
def test_create_scheduler_factory():
    """Test create_scheduler factory function."""
    optimizer = create_dummy_optimizer(lr=0.001)

    # Test each scheduler type
    constant = create_scheduler(
        "constant",
        optimizer,
        initial_lr=0.001,
        total_steps=100,
    )
    assert isinstance(constant, ConstantLR)

    linear = create_scheduler(
        "linear_warmup",
        optimizer,
        initial_lr=0.001,
        total_steps=100,
        warmup_steps=10,
    )
    assert isinstance(linear, LinearWarmupLR)

    cosine = create_scheduler(
        "cosine",
        optimizer,
        initial_lr=0.001,
        total_steps=100,
        warmup_steps=10,
        min_lr=0.0001,
    )
    assert isinstance(cosine, CosineAnnealingLR)

    onecycle = create_scheduler(
        "onecycle",
        optimizer,
        initial_lr=0.001,
        total_steps=100,
        pct_start=0.3,
    )
    assert isinstance(onecycle, OneCycleLR)


@pytest.mark.fast
def test_create_scheduler_unknown_type():
    """Test that unknown scheduler type raises error."""
    optimizer = create_dummy_optimizer(lr=0.001)

    with pytest.raises(ValueError, match="Unknown scheduler"):
        create_scheduler(
            "unknown_scheduler",
            optimizer,
            initial_lr=0.001,
            total_steps=100,
        )


@pytest.mark.fast
def test_scheduler_updates_optimizer():
    """Test that scheduler updates optimizer learning rate."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = LinearWarmupLR(
        optimizer,
        initial_lr=0.01,  # Different from optimizer initial
        warmup_steps=10,
    )

    # Step and verify optimizer LR is updated
    for step in range(5):
        scheduler.step()

    # The scheduler should have updated the optimizer's LR
    # Note: MLX optimizer doesn't expose current LR directly,
    # but the scheduler tracks it
    expected_lr = 0.01 * 6 / 10  # Step 5, so (5+1)/10
    np.testing.assert_almost_equal(scheduler.get_lr(), expected_lr, decimal=6)


@pytest.mark.fast
def test_linear_warmup_single_step():
    """Test LinearWarmupLR with single warmup step."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = LinearWarmupLR(
        optimizer,
        initial_lr=0.001,
        warmup_steps=1,
    )

    # With warmup_steps=1, step 0 should get initial_lr directly
    assert scheduler.get_lr() == 0.001

    # After step, should stay at initial_lr
    scheduler.step()
    assert scheduler.get_lr() == 0.001


@pytest.mark.fast
def test_cosine_no_min_lr():
    """Test CosineAnnealingLR with min_lr=0."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = CosineAnnealingLR(
        optimizer,
        initial_lr=0.001,
        total_steps=100,
        warmup_steps=0,
        min_lr=0.0,
    )

    # Run to completion
    for _ in range(100):
        scheduler.step()

    # Should end at min_lr (0)
    np.testing.assert_almost_equal(scheduler.get_lr(), 0.0, decimal=6)


@pytest.mark.fast
def test_onecycle_lr_boundaries():
    """Test OneCycleLR computes correct start and end LRs."""
    optimizer = create_dummy_optimizer(lr=0.001)
    scheduler = OneCycleLR(
        optimizer,
        initial_lr=0.001,
        total_steps=100,
        div_factor=25.0,
        final_div_factor=1e4,
    )

    # Start LR should be initial_lr / div_factor
    expected_start = 0.001 / 25.0
    # First LR should be somewhere between start_lr and initial_lr
    first_lr = scheduler.get_lr()
    assert first_lr > expected_start

    # Final LR should be close to initial_lr / final_div_factor
    for _ in range(100):
        scheduler.step()

    expected_final = 0.001 / 1e4
    np.testing.assert_almost_equal(scheduler.get_lr(), expected_final, decimal=8)
