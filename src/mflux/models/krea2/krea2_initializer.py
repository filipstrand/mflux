import mlx.core as mx
from mlx import nn

import mflux.models.krea2.model.krea2_scheduler  # noqa: F401 — register er_sde/euler schedulers
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.krea2.model.krea2_text_encoder.text_encoder import Krea2TextEncoder
from mflux.models.krea2.model.krea2_transformer.transformer import Krea2Transformer
from mflux.models.krea2.weights.krea2_lora_mapping import Krea2LoRAMapping
from mflux.models.krea2.weights.krea2_weight_definition import Krea2WeightDefinition
from mflux.models.qwen.model.qwen_vae.qwen_vae import QwenVAE


class Krea2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        path = model_path if model_path else model_config.model_name
        Krea2Initializer._init_config(model, model_config)
        weights = Krea2Initializer._load_weights(path, model_config)
        Krea2Initializer._init_tokenizers(model, path)
        Krea2Initializer._init_models(model, model_config)
        Krea2Initializer._apply_weights(model, weights, quantize)
        Krea2Initializer._apply_lora(model, lora_paths, lora_scales, bake_lora, weights)
        del weights
        mx.eval(model)
        mx.clear_cache()

    # The depth checkpoint's deltas, per block: five attention and three MLP projections.
    _CONTROL_TARGETS = ("attn.wq", "attn.wk", "attn.wv", "attn.wo", "attn.gate", "mlp.gate", "mlp.up", "mlp.down")

    @staticmethod
    def init_controlnet(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        controlnet_path: str | None = None,
        controlnet_strength: float = 1.0,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        # The depth checkpoint holds an input projection twice as wide plus attention/MLP deltas as
        # raw A/B matrices, so it is merged into the dense base weights. Quantization runs after the merge.
        control_source = controlnet_path or model_config.controlnet_model
        if control_source is None:
            # Checked before the base loads, which takes a while.
            raise ValueError("No depth ControlNet checkpoint: pass controlnet_path or a model config that names one.")
        path = model_path if model_path else model_config.model_name
        Krea2Initializer._init_config(model, model_config)
        weights = Krea2Initializer._load_weights(path, model_config)
        if weights.meta_data.quantization_level is not None:
            # The weights load lazily, so this is known before anything heavy runs.
            raise ValueError(
                f"{path} is saved at {weights.meta_data.quantization_level} bits. The Krea 2 depth ControlNet "
                "merges into the unquantized weights: use the original checkpoint and quantize with -q."
            )
        Krea2Initializer._init_tokenizers(model, path)
        Krea2Initializer._init_models(model, model_config)
        Krea2Initializer._apply_weights(model, weights, quantize=None)
        del weights
        Krea2Initializer._apply_control_checkpoint(model, control_source, controlnet_strength)
        model.controlnet_strength = controlnet_strength
        # A baked LoRA is folded into the merged dense weights before -q quantizes them, as mflux-generate-krea2
        # folds it before quantizing. dense_weights stays None: the checkpoint's own tree predates the control
        # merge, so folding from it would drop the control deltas. A LoRA that is not baked stays a live adapter
        # on the quantized layers.
        if bake_lora:
            Krea2Initializer._apply_controlnet_lora(model, lora_paths, lora_scales, bake_lora=True)
            Krea2Initializer._quantize_after_control(model, quantize)
        else:
            Krea2Initializer._quantize_after_control(model, quantize)
            Krea2Initializer._apply_controlnet_lora(model, lora_paths, lora_scales, bake_lora=False)
        mx.eval(model)
        mx.clear_cache()

    @staticmethod
    def _apply_controlnet_lora(
        model, lora_paths: list[str] | None, lora_scales: list[float] | None, bake_lora: bool
    ) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=None,
        )

    @staticmethod
    def _quantize_after_control(model, quantize: int | None) -> None:
        if quantize is None:
            return
        # The same layers the base load quantizes: the definition's predicate, and no text encoder.
        components = {c.name: c for c in Krea2WeightDefinition.get_components()}
        for name, module in (
            ("vae", model.vae),
            ("transformer", model.transformer),
            ("text_encoder", model.text_encoder),
        ):
            if not components[name].skip_quantization:
                nn.quantize(module, bits=quantize, class_predicate=Krea2WeightDefinition.quantization_predicate)
        model.bits = quantize

    @staticmethod
    def _apply_control_checkpoint(model, source: str, strength: float) -> None:
        root = PathResolution.resolve(source, ["*.safetensors"])
        files = [root] if root.is_file() else sorted(root.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(f"Expected one ControlNet .safetensors file in {root}, found {len(files)}.")
        control = mx.load(str(files[0]))
        transformer = model.transformer

        # Every shape is checked before anything changes, so a wrong file fails here and not mid-denoise.
        first_w = control.get("first.weight")
        first_b = control.get("first.bias")
        if first_w is None or first_b is None:
            raise ValueError(f"{files[0]} is not a Krea 2 depth ControlNet: 'first.weight'/'first.bias' are missing.")
        out_features = transformer.first.weight.shape[0]
        in_features = 2 * transformer.channels * transformer.patch**2
        if tuple(first_w.shape) != (out_features, in_features) or tuple(first_b.shape) != (out_features,):
            raise ValueError(
                f"Control 'first' is {tuple(first_w.shape)} + {tuple(first_b.shape)}, "
                f"expected {(out_features, in_features)} + {(out_features,)}."
            )
        # The file must hold exactly one A/B pair per block and target: a partial or foreign file would otherwise
        # load without part of the control.
        expected = {"first.weight", "first.bias"} | {
            f"blocks.{i}.{target}.{half}"
            for i in range(len(transformer.blocks))
            for target in Krea2Initializer._CONTROL_TARGETS
            for half in ("A", "B")
        }
        missing, unknown = sorted(expected - control.keys()), sorted(control.keys() - expected)
        if missing or unknown:
            raise ValueError(
                f"{files[0]} does not match the Krea 2 depth ControlNet: "
                f"{len(missing)} missing keys {missing[:3]}, {len(unknown)} unknown keys {unknown[:3]}."
            )
        # Only the A/B pairs are kept here: each B @ A is a full weight-sized matrix, so they are formed one
        # at a time during the merge below (all 224 at once would hold about 48 GB in float32).
        pairs: dict[tuple[int, str], tuple[mx.array, mx.array]] = {}
        for i, block in enumerate(transformer.blocks):
            for target in Krea2Initializer._CONTROL_TARGETS:
                a = control[f"blocks.{i}.{target}.A"]
                b = control[f"blocks.{i}.{target}.B"]
                sub, attr = target.split(".")
                shape = tuple(getattr(getattr(block, sub), attr).weight.shape)
                if a.shape[0] != b.shape[1] or (b.shape[0], a.shape[1]) != shape:
                    raise ValueError(f"Control delta blocks.{i}.{target} (B{tuple(b.shape)} @ A{tuple(a.shape)}) does not match weight {shape}.")  # fmt: off
                pairs[(i, target)] = (a, b)

        # The reference projection is first(cat([img, ctrl])). Its image columns replace first's weight, so first
        # keeps its shape and a LoRA on it still fits; the depth columns become control_first.
        dtype = transformer.first.weight.dtype
        image_in = transformer.channels * transformer.patch**2
        transformer.first.weight = first_w[:, :image_in].astype(dtype)
        transformer.first.bias = first_b.astype(dtype)
        transformer.control_first = nn.Linear(in_features - image_in, out_features, bias=False)
        transformer.control_first.weight = first_w[:, image_in:].astype(dtype)
        for (i, target), (a, b) in pairs.items():
            sub, attr = target.split(".")
            layer = getattr(getattr(transformer.blocks[i], sub), attr)
            layer.weight = layer.weight + (strength * (b @ a)).astype(layer.weight.dtype)

    @staticmethod
    def _init_tokenizers(model, model_path: str) -> None:
        model.tokenizers = TokenizerLoader.load_all(
            definitions=Krea2WeightDefinition.get_tokenizers(),
            model_path=model_path,
        )

    @staticmethod
    def _init_config(model, model_config: ModelConfig) -> None:
        model.prompt_cache = {}
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.tiling_config = None
        model.lora_paths = None
        model.lora_scales = None

    @staticmethod
    def _load_weights(model_path: str, model_config: ModelConfig) -> LoadedWeights:
        return WeightLoader.load(
            weight_definition=Krea2WeightDefinition,
            model_path=model_path,
            download_patterns=Krea2WeightDefinition.get_download_patterns(model_config.model_name),
        )

    @staticmethod
    def _init_models(model, model_config: ModelConfig) -> None:
        model.vae = QwenVAE()
        model.transformer = Krea2Transformer(**(model_config.transformer_overrides or {}))
        model.text_encoder = Krea2TextEncoder()

    @staticmethod
    def _apply_weights(model, weights: LoadedWeights, quantize: int | None) -> None:
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Krea2WeightDefinition,
            models={
                "vae": model.vae,
                "transformer": model.transformer,
                "text_encoder": model.text_encoder,
            },
        )

    @staticmethod
    def _apply_lora(
        model,
        lora_paths: list[str] | None,
        lora_scales: list[float] | None,
        bake_lora: bool,
        weights: LoadedWeights,
    ) -> None:
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Krea2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )
