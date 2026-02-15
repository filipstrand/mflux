from types import SimpleNamespace

from mflux.models.common.training.trainer import TrainingTrainer


class _DummyOptimizer:
    def __init__(self, state):
        self.optimizer = SimpleNamespace(state=state)
        self.saved_paths = []

    def save(self, path):
        self.saved_paths.append(path)


class TestTrainingTrainer:
    def test_generate_previews_with_optimizer_offload_low_ram(self, monkeypatch):
        dummy_optimizer = _DummyOptimizer(state=["original_state"])
        training_state = SimpleNamespace(optimizer=dummy_optimizer)
        training_spec = SimpleNamespace(low_ram=True)
        adapter = object()

        preview_state_snapshots = []
        clear_cache_calls = []
        gc_calls = []

        def fake_generate_previews(_adapter, _training_spec, _training_state):
            preview_state_snapshots.append(_training_state.optimizer.optimizer.state)

        monkeypatch.setattr(TrainingTrainer, "_generate_previews", fake_generate_previews)
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.clear_cache", lambda: clear_cache_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.gc.collect", lambda: gc_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.load", lambda _path: {"k": "v"})
        monkeypatch.setattr(
            "mflux.models.common.training.trainer.tree_unflatten",
            lambda items: ["restored", items],
        )

        TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)

        assert len(dummy_optimizer.saved_paths) == 1
        assert dummy_optimizer.saved_paths[0].name == "optimizer_offload.safetensors"
        assert preview_state_snapshots == [[]]
        assert dummy_optimizer.optimizer.state == ["restored", [("k", "v")]]
        assert len(clear_cache_calls) == 2
        assert len(gc_calls) == 2

    def test_generate_previews_with_optimizer_offload_non_low_ram(self, monkeypatch):
        dummy_optimizer = _DummyOptimizer(state=["original_state"])
        training_state = SimpleNamespace(optimizer=dummy_optimizer)
        training_spec = SimpleNamespace(low_ram=False)
        adapter = object()

        preview_state_snapshots = []
        clear_cache_calls = []
        gc_calls = []

        def fake_generate_previews(_adapter, _training_spec, _training_state):
            preview_state_snapshots.append(_training_state.optimizer.optimizer.state)

        monkeypatch.setattr(TrainingTrainer, "_generate_previews", fake_generate_previews)
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.clear_cache", lambda: clear_cache_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.gc.collect", lambda: gc_calls.append(1))
        monkeypatch.setattr("mflux.models.common.training.trainer.mx.load", lambda _path: {"k": "v"})
        monkeypatch.setattr(
            "mflux.models.common.training.trainer.tree_unflatten",
            lambda items: ["restored", items],
        )

        TrainingTrainer._generate_previews_with_optimizer_offload(adapter, training_spec, training_state)

        assert len(dummy_optimizer.saved_paths) == 1
        assert dummy_optimizer.saved_paths[0].name == "optimizer_offload.safetensors"
        assert preview_state_snapshots == [[]]
        assert dummy_optimizer.optimizer.state == ["restored", [("k", "v")]]
        assert len(clear_cache_calls) == 2
        assert len(gc_calls) == 2
