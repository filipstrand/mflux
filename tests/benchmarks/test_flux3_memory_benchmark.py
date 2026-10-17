import importlib.util
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock

import pytest

pytestmark = pytest.mark.fast
path = Path(__file__).parents[2] / "scripts/flux3/benchmark_memory.py"
spec = importlib.util.spec_from_file_location("flux3_memory_benchmark", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MemoryBenchmark = module.MemoryBenchmark


class TestMemoryBenchmark:
    @staticmethod
    def backend():
        return Mock(
            get_peak_memory=Mock(return_value=120),
            get_active_memory=Mock(return_value=70),
            get_cache_memory=Mock(return_value=40),
        )

    def test_peak_before_reset_survives_lower_later_peak(self):
        mx = self.backend()
        benchmark = MemoryBenchmark(mx)
        reset = Mock(side_effect=lambda: setattr(mx.get_peak_memory, "return_value", 0))
        benchmark.reset_wrapper(reset)()
        mx.get_peak_memory.return_value = 80
        benchmark.snapshot("finished")
        assert max(benchmark.peaks) == 120
        assert benchmark.events[0]["label"] == "before_peak_reset"
        reset.assert_called_once_with()

    def test_wrapped_static_method_restored_after_exception(self):
        class Component:
            @staticmethod
            def load():
                raise ValueError("failed load")

        original = Component.__dict__["load"]
        benchmark = MemoryBenchmark(self.backend())
        with pytest.raises(ValueError, match="failed load"), ExitStack() as stack:
            benchmark.instrument(stack, Component, "load", "load", static=True)
            Component.load()
        assert Component.__dict__["load"] is original
        assert benchmark.phases == []

    def test_lazy_result_materialised_before_phase_end(self):
        order = []
        result = object()

        class Component:
            @staticmethod
            def encode():
                order.append("encode")
                return result

        mx = self.backend()
        mx.eval.side_effect = lambda value: order.append("eval")
        mx.synchronize.side_effect = lambda: order.append("sync")
        benchmark = MemoryBenchmark(mx)
        with ExitStack() as stack:
            benchmark.instrument(stack, Component, "encode", "encode", static=True, evaluate=True)
            assert Component.encode() is result
        assert order == ["sync", "encode", "eval", "sync", "sync"]
        mx.eval.assert_called_once_with(result)
        assert benchmark.phases[0]["label"] == "encode"

    def test_failed_save_not_counted_as_success(self):
        class Image:
            def save(self):
                raise OSError("disk full")

        benchmark = MemoryBenchmark(self.backend())
        with pytest.raises(OSError, match="disk full"), ExitStack() as stack:
            benchmark.instrument(stack, Image, "save", "save")
            Image().save()
        assert benchmark.saved_images == 0
