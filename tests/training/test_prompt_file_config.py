from pathlib import Path

import pytest

from mflux.models.common.training.state.training_spec import TrainingSpec


@pytest.mark.fast
def test_prompt_file_is_loaded_and_resolved_relative_to_data_path(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "01.txt").write_text("photo of sks dog\n", encoding="utf-8")
    (data_dir / "01.jpeg").write_bytes(b"")  # image file existence isn't required, but keep it realistic

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "lora_layers": {
            "targets": [
                {
                    "module_path": "transformer_blocks.{block}.attn.to_q",
                    "blocks": {"start": 0, "end": 1},
                    "rank": 4,
                }
            ]
        },
        "data": "data",
    }

    config_path = tmp_path / "train.json"
    spec = TrainingSpec.from_conf(conf, str(config_path), new_folder=False)

    assert spec.data[0].prompt == "photo of sks dog"
    assert spec.data[0].image == data_dir / "01.jpeg"


@pytest.mark.fast
def test_data_images_omitted_discovers_images_and_uses_matching_txt(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Deliberately unsorted creation order; discovery should be deterministic by filename.
    (data_dir / "02.jpeg").write_bytes(b"")
    (data_dir / "02.txt").write_text("two\n", encoding="utf-8")
    (data_dir / "01.jpeg").write_bytes(b"")
    (data_dir / "01.txt").write_text("one\n", encoding="utf-8")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "lora_layers": {
            "targets": [
                {"module_path": "transformer_blocks.{block}.attn.to_q", "blocks": {"start": 0, "end": 1}, "rank": 4}
            ]
        },
        "data": "data",
    }

    config_path = tmp_path / "train.json"
    spec = TrainingSpec.from_conf(conf, str(config_path), new_folder=False)

    assert [e.image.name for e in spec.data] == ["01.jpeg", "02.jpeg"]
    assert [e.prompt for e in spec.data] == ["one", "two"]


@pytest.mark.fast
def test_data_images_omitted_missing_prompt_file_raises(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    (data_dir / "01.jpeg").write_bytes(b"")
    # missing 01.txt

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "lora_layers": {
            "targets": [
                {"module_path": "transformer_blocks.{block}.attn.to_q", "blocks": {"start": 0, "end": 1}, "rank": 4}
            ]
        },
        "data": "data",
    }

    with pytest.raises(ValueError, match="Missing prompt file"):
        TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)


@pytest.mark.fast
def test_data_object_is_rejected(tmp_path: Path):
    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "lora_layers": {
            "targets": [
                {
                    "module_path": "transformer_blocks.{block}.attn.to_q",
                    "blocks": {"start": 0, "end": 1},
                    "rank": 4,
                }
            ]
        },
        "data": {"path": "data"},
    }

    with pytest.raises(ValueError, match="data.*string"):
        TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)


@pytest.mark.fast
def test_timestep_sampling_defaults_and_overrides(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "01.jpeg").write_bytes(b"")
    (data_dir / "01.txt").write_text("a\n", encoding="utf-8")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "lora_layers": {
            "targets": [
                {"module_path": "transformer_blocks.{block}.attn.to_q", "blocks": {"start": 0, "end": 1}, "rank": 4}
            ]
        },
        "data": "data",
    }

    spec = TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)
    assert spec.training_loop.timestep_low == 0
    assert spec.training_loop.timestep_high is None  # defaults to steps at runtime

    conf2 = {
        **conf,
        "training_loop": {"num_epochs": 1, "batch_size": 1, "timestep_low": 3, "timestep_high": 7},
    }
    spec2 = TrainingSpec.from_conf(conf2, str(tmp_path / "train.json"), new_folder=False)
    assert spec2.training_loop.timestep_low == 3
    assert spec2.training_loop.timestep_high == 7

    conf_bad = {
        **conf,
        "training_loop": {"num_epochs": 1, "batch_size": 1, "timestep_low": 10, "timestep_high": 10},
    }
    with pytest.raises(ValueError, match="timestep_low.*timestep_high"):
        TrainingSpec.from_conf(conf_bad, str(tmp_path / "train.json"), new_folder=False)


@pytest.mark.fast
def test_preview_prompt_file_is_loaded_relative_to_config(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "01.jpeg").write_bytes(b"")
    (data_dir / "01.txt").write_text("a\n", encoding="utf-8")
    (data_dir / "preview.txt").write_text("photo of sks dog\n", encoding="utf-8")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "monitoring": {
            "plot_frequency": 1,
            "generate_image_frequency": 20,
        },
        "lora_layers": {
            "targets": [
                {
                    "module_path": "transformer_blocks.{block}.attn.to_q",
                    "blocks": {"start": 0, "end": 1},
                    "rank": 4,
                }
            ]
        },
        "data": "data",
    }

    spec = TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)
    assert spec.monitoring is not None
    assert spec.monitoring.preview_prompts == ["photo of sks dog"]


@pytest.mark.fast
def test_preview_images_included_without_monitoring(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "preview.jpeg").write_bytes(b"")
    (data_dir / "preview.txt").write_text("a\n", encoding="utf-8")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "lora_layers": {
            "targets": [
                {"module_path": "transformer_blocks.{block}.attn.to_q", "blocks": {"start": 0, "end": 1}, "rank": 4}
            ]
        },
        "data": "data",
    }

    spec = TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)
    assert [e.image.name for e in spec.data] == ["preview.jpeg"]


@pytest.mark.fast
def test_mixed_edit_and_txt2img_data_raise(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "01_out.jpeg").write_bytes(b"")
    (data_dir / "01_in.jpeg").write_bytes(b"")
    (data_dir / "01_in.txt").write_text("edit\n", encoding="utf-8")
    (data_dir / "02.jpeg").write_bytes(b"")
    (data_dir / "02.txt").write_text("txt2img\n", encoding="utf-8")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "lora_layers": {
            "targets": [
                {"module_path": "transformer_blocks.{block}.attn.to_q", "blocks": {"start": 0, "end": 1}, "rank": 4}
            ]
        },
        "data": "data",
    }

    with pytest.raises(ValueError, match="mixes edit-style and txt2img"):
        TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)


@pytest.mark.fast
def test_validation_prompt_is_rejected(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "01.jpeg").write_bytes(b"")
    (data_dir / "01.txt").write_text("a\n", encoding="utf-8")
    (data_dir / "preview.txt").write_text("photo of sks dog\n", encoding="utf-8")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "monitoring": {
            "plot_frequency": 1,
            "generate_image_frequency": 20,
            "validation_prompt": "a",
        },
        "lora_layers": {
            "targets": [
                {
                    "module_path": "transformer_blocks.{block}.attn.to_q",
                    "blocks": {"start": 0, "end": 1},
                    "rank": 4,
                }
            ]
        },
        "data": "data",
    }

    with pytest.raises(ValueError, match="validation_prompt"):
        TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)


@pytest.mark.fast
def test_preview_prompt_file_defaults_to_data(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "01.jpeg").write_bytes(b"")
    (data_dir / "01.txt").write_text("a\n", encoding="utf-8")
    (data_dir / "preview.txt").write_text("photo of sks dog\n", encoding="utf-8")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "monitoring": {
            "plot_frequency": 1,
            "generate_image_frequency": 20,
        },
        "lora_layers": {
            "targets": [
                {
                    "module_path": "transformer_blocks.{block}.attn.to_q",
                    "blocks": {"start": 0, "end": 1},
                    "rank": 4,
                }
            ]
        },
        "data": "data",
    }

    spec = TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)
    assert spec.monitoring is not None
    assert spec.monitoring.preview_prompts == ["photo of sks dog"]


@pytest.mark.fast
def test_preview_prompt_files_are_loaded_in_name_order(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "01.jpeg").write_bytes(b"")
    (data_dir / "01.txt").write_text("a\n", encoding="utf-8")
    (data_dir / "preview_2.txt").write_text("two\n", encoding="utf-8")
    (data_dir / "preview_1.txt").write_text("one\n", encoding="utf-8")
    (data_dir / "preview.txt").write_text("zero\n", encoding="utf-8")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "monitoring": {
            "plot_frequency": 1,
            "generate_image_frequency": 20,
        },
        "lora_layers": {
            "targets": [
                {
                    "module_path": "transformer_blocks.{block}.attn.to_q",
                    "blocks": {"start": 0, "end": 1},
                    "rank": 4,
                }
            ]
        },
        "data": "data",
    }

    spec = TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)
    assert spec.monitoring is not None
    assert spec.monitoring.preview_prompts == ["zero", "one", "two"]


@pytest.mark.fast
def test_edit_autodiscovery_monitoring_fallback_uses_first_input_image(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "01_in.jpeg").write_bytes(b"")
    (data_dir / "01_in.txt").write_text("edit one\n", encoding="utf-8")
    (data_dir / "01_out.jpeg").write_bytes(b"")
    (data_dir / "02_in.jpeg").write_bytes(b"")
    (data_dir / "02_in.txt").write_text("edit two\n", encoding="utf-8")
    (data_dir / "02_out.jpeg").write_bytes(b"")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "monitoring": {
            "plot_frequency": 1,
            "generate_image_frequency": 20,
        },
        "lora_layers": {
            "targets": [
                {
                    "module_path": "transformer_blocks.{block}.attn.to_q",
                    "blocks": {"start": 0, "end": 1},
                    "rank": 4,
                }
            ]
        },
        "data": "data",
    }

    spec = TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)
    assert spec.is_edit is True
    assert spec.monitoring is not None
    assert spec.monitoring.preview_prompts == ["edit one"]
    assert spec.monitoring.preview_images is not None
    assert [p.name for p in spec.monitoring.preview_images] == ["01_in.jpeg"]


@pytest.mark.fast
def test_validation_prompt_file_is_rejected(tmp_path: Path):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "01.jpeg").write_bytes(b"")
    (data_dir / "01.txt").write_text("a\n", encoding="utf-8")
    (data_dir / "preview.txt").write_text("photo of sks dog\n", encoding="utf-8")

    conf = {
        "model": "dev",
        "seed": 42,
        "steps": 20,
        "guidance": 3.0,
        "quantize": 4,
        "training_loop": {"num_epochs": 1, "batch_size": 1},
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4},
        "checkpoint": {"output_path": str(tmp_path / "out"), "save_frequency": 10},
        "monitoring": {
            "plot_frequency": 1,
            "generate_image_frequency": 20,
            "validation_prompt_file": "val.txt",
        },
        "lora_layers": {
            "targets": [
                {
                    "module_path": "transformer_blocks.{block}.attn.to_q",
                    "blocks": {"start": 0, "end": 1},
                    "rank": 4,
                }
            ]
        },
        "data": "data",
    }

    with pytest.raises(ValueError, match="validation_prompt_file"):
        TrainingSpec.from_conf(conf, str(tmp_path / "train.json"), new_folder=False)
