import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.compute_precision import ComputePrecision
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_text_encoder.qwen3_text_encoder import Qwen3TextEncoder
from mflux.models.iris.model.iris_transformer.iris_transformer import IrisTransformer
from mflux.models.iris.weights.iris_weight_definition import IrisTurboWeightDefinition, IrisWeightDefinition


class IrisInitializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        turbo: bool,
        quantize: int | None,
        model_path: str | None = None,
        compute_precision: mx.Dtype | None = None,
    ) -> None:
        precision = ComputePrecision(compute_precision)
        if precision.dtype is not None:
            raise NotImplementedError("Iris does not support a reduced compute precision yet.")
        path = model_path if model_path else model_config.model_name
        definition = IrisTurboWeightDefinition if turbo else IrisWeightDefinition
        IrisInitializer._init_config(model, model_config, turbo)
        IrisInitializer._init_tokenizers(model)
        IrisInitializer._init_transformer(model, definition, path, quantize)
        IrisInitializer._init_text_encoder(model, quantize)
        model.compute_precision = precision

    @staticmethod
    def _init_config(model, model_config: ModelConfig, turbo: bool) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.turbo = turbo
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model._noise_override = None  # test hook: golden-parity scripts replace the seeded noise with a fixed tensor

    @staticmethod
    def _init_tokenizers(model) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=IrisWeightDefinition.get_tokenizers(),
            model_path=IrisWeightDefinition.TEXT_ENCODER_REPO,
        )

    @staticmethod
    def _init_transformer(model, definition, path: str, quantize: int | None) -> None:
        component = definition.get_transformer_component()
        weights = WeightLoader.load_single(
            component=component,
            repo_id=path,
            file_pattern=definition.get_download_patterns()[0],
        )
        model.transformer = IrisTransformer()
        model.bits = WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=model.transformer,
            component=component,
            quantize_arg=quantize,
            quantization_predicate=definition.quantization_predicate,
        )

    @staticmethod
    def _init_text_encoder(model, quantize: int | None) -> None:
        component = IrisWeightDefinition.get_text_encoder_component()
        weights = WeightLoader.load_single(
            component=component,
            repo_id=IrisWeightDefinition.TEXT_ENCODER_REPO,
            file_pattern=IrisWeightDefinition.TEXT_ENCODER_PATTERN,
        )
        model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)
        WeightApplier.apply_and_quantize_single(
            weights=weights,
            model=model.text_encoder,
            component=component,
            quantize_arg=quantize,
        )
