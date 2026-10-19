from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights, MetaData
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
        quantize: int | None,
        model_path: str | None = None,
    ) -> None:
        definition = IrisInitializer.weight_definition(model_config)
        saved_path = IrisInitializer.saved_path(model_config, model_path)
        IrisInitializer._init_config(model, model_config)
        weights = IrisInitializer._load_weights(definition, model_config, saved_path)
        IrisInitializer._init_tokenizers(model, definition, saved_path)
        IrisInitializer._init_models(model)
        IrisInitializer._apply_weights(model, weights, quantize, definition)

    @staticmethod
    def saved_path(model_config: ModelConfig, model_path: str | None) -> str | None:
        # The Iris repo id (what a metadata sidecar records as the model) means the upstream checkpoints. Any other
        # path is a saved model, even when it is the model_name of a config that mflux-save built for that path, or
        # when it only arrives as that model_name (Iris(model_config=...) without model_path).
        upstream_repo = model_config.base_model or model_config.model_name
        candidate = model_path or model_config.model_name
        return None if candidate == upstream_repo else candidate

    @staticmethod
    def weight_definition(model_config: ModelConfig):
        return IrisWeightDefinition if model_config.supports_guidance else IrisTurboWeightDefinition

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None

    @staticmethod
    def _load_weights(definition, model_config: ModelConfig, model_path: str | None) -> LoadedWeights:
        if model_path is not None:
            # An mflux-saved model (mflux-save) holds the transformer, the text encoder and the tokenizer.
            return WeightLoader.load(weight_definition=definition, model_path=model_path)
        transformer = WeightLoader.load_single(
            component=definition.source_transformer(),
            repo_id=model_config.model_name,
            file_pattern=definition.SOURCE_TRANSFORMER_PATTERN,
        )
        text_encoder = WeightLoader.load_single(
            component=definition.source_text_encoder(),
            repo_id=IrisWeightDefinition.TEXT_ENCODER_REPO,
            file_pattern=IrisWeightDefinition.TEXT_ENCODER_PATTERN,
        )
        return LoadedWeights(
            components={**transformer.components, **text_encoder.components},
            meta_data=MetaData(),
        )

    @staticmethod
    def _init_tokenizers(model, definition, model_path: str | None) -> None:
        if model_path is not None:
            definitions, path = definition.get_tokenizers(), model_path
        else:
            definitions, path = definition.source_tokenizers(), IrisWeightDefinition.TEXT_ENCODER_REPO
        model.tokenizers = TokenizerLoader.load_all(definitions=definitions, model_path=path)

    @staticmethod
    def _init_models(model) -> None:
        model.transformer = IrisTransformer()
        model.text_encoder = Qwen3TextEncoder(**model.model_config.text_encoder_overrides)

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None, definition) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=definition,
            models={"transformer": model.transformer, "text_encoder": model.text_encoder},
        )
