from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.common_models.qwen3_vl.qwen3_vl_decoder import Qwen3VLDecoder
from mflux.models.common_models.qwen3_vl.qwen3_vl_vision_model import Qwen3VLVisionModel
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_definition import FIBOVLMWeightDefinition


class FiboVLMInitializer:
    @staticmethod
    def init(
        model,
        model_path: str = "briaai/FIBO-vlm",
        quantize: int | None = None,
    ) -> None:
        FiboVLMInitializer._init_config(model, model_path)
        weights = FiboVLMInitializer._load_weights(model_path)
        FiboVLMInitializer._init_tokenizers(model, model_path)
        FiboVLMInitializer._init_models(model)
        FiboVLMInitializer._apply_weights(model, weights, quantize)

    @staticmethod
    def _init_config(model, model_path: str) -> None:
        model.model_path = model_path

    @staticmethod
    def _load_weights(model_path: str) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=FIBOVLMWeightDefinition,
            model_path=model_path,
        )

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=FIBOVLMWeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_models(model) -> None:
        model.decoder = Qwen3VLDecoder(visual=Qwen3VLVisionModel())

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=FIBOVLMWeightDefinition,
            models={
                "decoder": model.decoder,
                "visual": model.decoder.visual,
            },
        )
