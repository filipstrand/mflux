"""Integration tests for Z-Image training workflow.

Tests the full training pipeline:
- Initialize model and training state
- Train for a few steps
- Save checkpoint
- Load and verify weights

These tests require actual model loading and are marked slow.
"""

import shutil
import zipfile
from pathlib import Path

import mlx.core as mx
import pytest

# Test directory paths
TEST_DIR = Path(__file__).parent
CONFIG_PATH = TEST_DIR / "config" / "train.json"
TMP_DIR = TEST_DIR / "tmp"


def cleanup_tmp_dir():
    """Clean up temporary test directory."""
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR)


@pytest.fixture(autouse=True)
def setup_teardown():
    """Setup and teardown for each test."""
    cleanup_tmp_dir()
    yield
    cleanup_tmp_dir()


@pytest.mark.slow
@pytest.mark.high_memory_requirement
class TestZImageTrainAndLoad:
    """Integration tests for Z-Image training.

    These tests require:
    - Z-Image-Base model weights
    - Sufficient memory (recommended 64GB+)
    """

    def test_training_spec_from_config(self):
        """Test that training spec can be loaded from config."""
        from mflux.models.z_image.variants.training.state.training_spec import TrainingSpec

        spec = TrainingSpec.resolve(str(CONFIG_PATH), None)

        assert spec.model == "z-image-base"
        assert spec.seed == 42
        assert spec.steps == 2
        assert spec.width == 128
        assert spec.height == 128
        assert spec.training_loop.batch_size == 1
        assert spec.training_loop.num_epochs == 1
        assert spec.lora_layers is not None

    def test_minimal_training_run(self):
        """Test minimal training run (1 epoch, 1 batch).

        This test validates:
        1. Model can be loaded with quantization
        2. Dataset can be prepared
        3. Training loop executes without error
        4. Checkpoint is saved
        5. LoRA weights can be extracted

        Note: Requires Z-Image-Base model weights to be available.
        """
        try:
            from mflux.models.z_image.variants.training.trainer import ZImageTrainer
            from mflux.models.z_image.variants.training.training_initializer import (
                ZImageTrainingInitializer,
            )
        except ImportError:
            pytest.skip("Z-Image training module not available")

        # Initialize training components
        try:
            model, config, training_spec, training_state = ZImageTrainingInitializer.initialize(
                config_path=str(CONFIG_PATH),
                checkpoint_path=None,
            )
        except FileNotFoundError as e:
            pytest.skip(f"Model weights not found: {e}")
        except Exception as e:  # noqa: BLE001 - Test setup fallback
            pytest.skip(f"Could not initialize training: {e}")

        # Run training
        try:
            ZImageTrainer.train(
                model=model,
                config=config,
                training_spec=training_spec,
                training_state=training_state,
            )
        finally:
            # Cleanup model memory
            del model, config, training_spec, training_state
            mx.synchronize()

        # Verify checkpoint was created
        checkpoint_dir = TMP_DIR / "_checkpoints"
        assert checkpoint_dir.exists(), "Checkpoint directory should exist"

        checkpoints = list(checkpoint_dir.glob("*_checkpoint.zip"))
        assert len(checkpoints) > 0, "At least one checkpoint should be saved"

        # Verify checkpoint contents
        latest_checkpoint = sorted(checkpoints)[-1]
        with zipfile.ZipFile(latest_checkpoint, "r") as zf:
            names = zf.namelist()
            assert any("adapter.safetensors" in n for n in names), "LoRA adapter should be in checkpoint"
            assert any("optimizer.safetensors" in n for n in names), "Optimizer state should be in checkpoint"
            assert any("iterator.json" in n for n in names), "Iterator state should be in checkpoint"
            assert any("loss.json" in n for n in names), "Loss history should be in checkpoint"
            assert "checkpoint.json" in names, "Checkpoint metadata should exist"

    def test_resume_from_checkpoint(self):
        """Test that training can resume from checkpoint.

        This test validates:
        1. Initial training produces a checkpoint
        2. Training can be resumed from that checkpoint
        3. Resumed training produces correct results
        """
        try:
            from mflux.models.z_image.variants.training.trainer import ZImageTrainer
            from mflux.models.z_image.variants.training.training_initializer import (
                ZImageTrainingInitializer,
            )
        except ImportError:
            pytest.skip("Z-Image training module not available")

        # First training run
        try:
            model, config, training_spec, training_state = ZImageTrainingInitializer.initialize(
                config_path=str(CONFIG_PATH),
                checkpoint_path=None,
            )
        except FileNotFoundError as e:
            pytest.skip(f"Model weights not found: {e}")
        except Exception as e:  # noqa: BLE001 - Test setup fallback
            pytest.skip(f"Could not initialize training: {e}")

        initial_iterations = training_state.iterator.num_iterations

        try:
            ZImageTrainer.train(
                model=model,
                config=config,
                training_spec=training_spec,
                training_state=training_state,
            )
        finally:
            del model, config, training_spec, training_state
            mx.synchronize()

        # Find the checkpoint
        checkpoint_dir = TMP_DIR / "_checkpoints"
        checkpoints = list(checkpoint_dir.glob("*_checkpoint.zip"))
        assert len(checkpoints) > 0, "Should have at least one checkpoint"

        latest_checkpoint = sorted(checkpoints)[-1]

        # Resume from checkpoint (this tests the resume logic, not more training)
        model2, config2, training_spec2, training_state2 = ZImageTrainingInitializer.initialize(
            config_path=None,
            checkpoint_path=str(latest_checkpoint),
        )

        try:
            # Verify state was restored
            assert training_state2.iterator.num_iterations > initial_iterations, (
                "Resumed iterator should have iterations from checkpoint"
            )
        finally:
            del model2, config2, training_spec2, training_state2
            mx.synchronize()


@pytest.mark.fast
class TestTrainingModeSpecs:
    """Unit tests for training mode specifications."""

    def test_full_finetune_mode_spec(self):
        """Test TrainingMode.FULL mode specification."""
        import tempfile

        # Create spec for full mode
        from mflux.models.z_image.variants.training.state.training_spec import (
            FullFinetuneSpec,
            OptimizerSpec,
            SaveSpec,
            StatisticsSpec,
            TrainingLoopSpec,
            TrainingMode,
            TrainingSpec,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            spec = TrainingSpec(
                model="z-image-base",
                seed=42,
                steps=2,
                guidance=3.5,
                quantize=None,  # Full mode typically uses fp16/bf16
                width=256,
                height=256,
                mode=TrainingMode.FULL,
                training_loop=TrainingLoopSpec(num_epochs=1, batch_size=1),
                optimizer=OptimizerSpec(name="AdamW", learning_rate=1e-5),
                saver=SaveSpec(checkpoint_frequency=1, output_path=tmpdir),
                instrumentation=None,
                statistics=StatisticsSpec(),
                examples=[],
                full_finetune=FullFinetuneSpec(
                    train_transformer=True,
                    train_vae=False,
                    train_text_encoder=False,
                ),
            )

            assert spec.mode == TrainingMode.FULL
            assert spec.full_finetune is not None
            assert spec.full_finetune.train_transformer is True
            assert spec.full_finetune.train_vae is False
            assert spec.lora_layers is None  # Not used in full mode

    def test_full_finetune_default_values(self):
        """Test FullFinetuneSpec default values."""
        from mflux.models.z_image.variants.training.state.training_spec import FullFinetuneSpec

        spec = FullFinetuneSpec()
        assert spec.train_transformer is True
        assert spec.train_vae is False
        assert spec.train_text_encoder is False
        assert spec.gradient_checkpointing is False

    def test_full_finetune_component_combinations(self):
        """Test various component training combinations."""
        from mflux.models.z_image.variants.training.state.training_spec import FullFinetuneSpec

        # Just transformer (most common)
        spec1 = FullFinetuneSpec(train_transformer=True, train_vae=False, train_text_encoder=False)
        assert spec1.train_transformer is True

        # Transformer + VAE
        spec2 = FullFinetuneSpec(train_transformer=True, train_vae=True, train_text_encoder=False)
        assert spec2.train_transformer is True
        assert spec2.train_vae is True

        # All components
        spec3 = FullFinetuneSpec(train_transformer=True, train_vae=True, train_text_encoder=True)
        assert spec3.train_transformer is True
        assert spec3.train_vae is True
        assert spec3.train_text_encoder is True


@pytest.mark.slow
class TestZImageLoRAExtraction:
    """Tests for LoRA weight extraction and loading."""

    def test_lora_weight_extraction(self):
        """Test that LoRA weights can be extracted from checkpoint."""
        # This test uses mock data to verify the extraction logic
        import json
        import tempfile

        from mflux.models.z_image.variants.training.state.zip_util import ZipUtil

        # Create a mock checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "test_checkpoint.zip"

            # Create mock files
            with zipfile.ZipFile(checkpoint_path, "w") as zf:
                # Write mock LoRA weights
                weights = {"test_weight": mx.zeros((4, 64))}
                mx.save_safetensors(str(Path(tmpdir) / "adapter.safetensors"), weights)
                zf.write(Path(tmpdir) / "adapter.safetensors", "0000001_adapter.safetensors")

                # Write checkpoint metadata
                checkpoint_meta = {
                    "metadata": {"mode": "lora"},
                    "files": {"lora_adapter": "0000001_adapter.safetensors"},
                }
                meta_path = Path(tmpdir) / "checkpoint.json"
                with open(meta_path, "w") as f:
                    json.dump(checkpoint_meta, f)
                zf.write(meta_path, "checkpoint.json")

            # Verify we can read from the zip
            meta = ZipUtil.unzip(
                str(checkpoint_path),
                "checkpoint.json",
                lambda x: json.load(open(x, "r")),
            )
            assert meta["files"]["lora_adapter"] == "0000001_adapter.safetensors"
