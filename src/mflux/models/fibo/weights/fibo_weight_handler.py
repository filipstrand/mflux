from pathlib import Path

import mlx.core as mx
import torch
from mlx.utils import tree_unflatten
from safetensors.torch import load_file as torch_load_file

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.fibo.weights.fibo_weight_mapping import FIBOWeightMapping
from mflux.models.flux.weights.weight_handler import (
    MetaData,
    WeightHandler as FluxWeightHandler,
)
from mflux.utils.download import snapshot_download


class FIBOWeightHandler:
    def __init__(
        self,
        meta_data: MetaData,
        vae: dict | None = None,
        transformer: dict | None = None,
        text_encoder: dict | None = None,
        decoder: dict | None = None,
        visual: dict | None = None,
        config: dict | None = None,
    ):
        self.vae = vae
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.decoder = decoder
        self.visual = visual
        self.config = config
        self.meta_data = meta_data

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> "FIBOWeightHandler":
        root_path: Path | None = None
        if local_path:
            root_path = Path(local_path)
        elif repo_id:
            root_path = FluxWeightHandler.download_or_get_cached_weights(repo_id)

        vae_weights = None
        transformer_weights = None
        text_encoder_weights = None
        quantization_level: int | None = None
        mflux_version: str | None = None

        if root_path is not None:
            (
                vae_weights,
                quantization_level,
                mflux_version,
            ) = FIBOWeightHandler._try_load_saved_component(root_path, "vae")
            (
                transformer_weights,
                quantization_level_t,
                mflux_version_t,
            ) = FIBOWeightHandler._try_load_saved_component(root_path, "transformer")
            (
                text_encoder_weights,
                quantization_level_te,
                mflux_version_te,
            ) = FIBOWeightHandler._try_load_saved_component(root_path, "text_encoder")

            if quantization_level_t is not None:
                quantization_level = quantization_level_t
            if mflux_version_t is not None:
                mflux_version = mflux_version_t

            if quantization_level_te is not None:
                quantization_level = quantization_level_te
            if mflux_version_te is not None:
                mflux_version = mflux_version_te

        if vae_weights is None or transformer_weights is None or text_encoder_weights is None:
            vae_weights = FIBOWeightHandler._load_vae_weights(repo_id, local_path)
            transformer_weights = FIBOWeightHandler._load_transformer_weights(repo_id, local_path)
            text_encoder_weights = FIBOWeightHandler._load_text_encoder_weights(repo_id, local_path)
            quantization_level = None
            mflux_version = None

        return FIBOWeightHandler(
            vae=vae_weights,
            transformer=transformer_weights,
            text_encoder=text_encoder_weights,
            meta_data=MetaData(
                quantization_level=quantization_level,
                scale=None,
                is_lora=False,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def _load_vae_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        root_path = FIBOWeightHandler._get_root_path(repo_id, local_path)
        vae_path = root_path / "vae"
        raw_weights = FIBOWeightHandler._load_safetensors_shards(vae_path)
        mapping = FIBOWeightMapping.get_vae_mapping()
        mapped_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_blocks=4)
        return mapped_weights

    @staticmethod
    def _load_transformer_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        root_path = FIBOWeightHandler._get_root_path(repo_id, local_path)
        transformer_path = root_path / "transformer"
        if transformer_path.exists() and list(transformer_path.glob("*.safetensors")):
            raw_weights = FIBOWeightHandler._load_safetensors_shards(transformer_path)
        else:
            raw_weights = FIBOWeightHandler._load_safetensors_shards(root_path)
        mapping = FIBOWeightMapping.get_transformer_mapping()
        mapped_weights = WeightMapper.apply_mapping(
            raw_weights,
            mapping,
            num_blocks=38,
            num_layers=46,
        )
        return mapped_weights

    @staticmethod
    def _load_text_encoder_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        root_path = FIBOWeightHandler._get_root_path(repo_id, local_path)
        text_encoder_path = root_path / "text_encoder"
        raw_weights = FIBOWeightHandler._load_safetensors_shards(text_encoder_path)
        mapping = FIBOWeightMapping.get_text_encoder_mapping()
        mapped_weights = WeightMapper.apply_mapping(
            raw_weights,
            mapping,
            num_blocks=36,
        )
        return mapped_weights

    @staticmethod
    def _get_root_path(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> Path:
        if local_path:
            return Path(local_path)
        return Path(FluxWeightHandler.download_or_get_cached_weights(repo_id or "briaai/FIBO"))

    @staticmethod
    def _load_safetensors_shards(path: Path) -> dict[str, mx.array]:
        shard_files = sorted(f for f in path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            torch_weights = torch_load_file(str(shard))
            for key, tensor in torch_weights.items():
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                all_weights[key] = mx.array(tensor.numpy())

        return all_weights

    @staticmethod
    def _try_load_saved_component(
        root_path: Path,
        component_name: str,
    ) -> tuple[dict | None, int | None, str | None]:
        component_path = root_path / component_name
        if not component_path.exists():
            return None, None, None

        shard_files = sorted(f for f in component_path.glob("*.safetensors") if not f.name.startswith("._"))
        if not shard_files:
            return None, None, None

        all_weights: dict[str, mx.array] = {}
        quantization_level: int | None = None
        mflux_version: str | None = None

        for idx, shard in enumerate(shard_files):
            data = mx.load(str(shard), return_metadata=True)
            weights_dict = data[0]
            all_weights.update(dict(weights_dict.items()))

            if idx == 0 and len(data) > 1:
                quantization_level = data[1].get("quantization_level")
                mflux_version = data[1].get("mflux_version")

        if quantization_level is None and mflux_version is None:
            return None, None, None

        unflattened = tree_unflatten(list(all_weights.items()))
        return unflattened, quantization_level, mflux_version

    @staticmethod
    def _load_vlm_safetensors_shards(root_path: Path) -> dict[str, mx.array]:
        shard_files = sorted([f for f in root_path.glob("*.safetensors") if not f.name.startswith("._")])
        if not shard_files:
            raise FileNotFoundError(f"No safetensors files found in {root_path}")

        all_weights: dict[str, mx.array] = {}
        for shard in shard_files:
            try:
                data = mx.load(str(shard), return_metadata=True)
                all_weights.update(dict(data[0].items()))
            except Exception:  # noqa: BLE001, PERF203
                torch_weights = torch_load_file(str(shard))
                for name, tensor in torch_weights.items():
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float16)
                    all_weights[name] = mx.array(tensor.numpy())
        return all_weights

    @staticmethod
    def load_vlm_decoder_weights(
        repo_id: str = "briaai/FIBO-vlm",
        local_path: str | None = None,
    ) -> dict:
        if local_path:
            root_path = Path(local_path)
        else:
            root_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=["*.safetensors", "*.json"],
                )
            )

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(root_path) if local_path else repo_id)
        num_layers = config.text_config.num_hidden_layers

        raw_weights = FIBOWeightHandler._load_vlm_safetensors_shards(root_path)

        decoder_weights = {
            k: v for k, v in raw_weights.items() if k.startswith("model.language_model") or k.startswith("lm_head")
        }

        mapping = FIBOWeightMapping.get_vlm_decoder_mapping(num_layers=num_layers)
        mapped_weights = WeightMapper.apply_mapping(decoder_weights, mapping, num_blocks=num_layers)

        return mapped_weights

    @staticmethod
    def load_vlm_visual_weights(
        repo_id: str = "briaai/FIBO-vlm",
        local_path: str | None = None,
    ) -> dict | None:
        if local_path:
            root_path = Path(local_path)
        else:
            root_path = Path(
                snapshot_download(
                    repo_id=repo_id,
                    allow_patterns=["*.safetensors", "*.json"],
                )
            )

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(str(root_path) if local_path else repo_id)
        depth = config.vision_config.depth

        raw_weights = FIBOWeightHandler._load_vlm_safetensors_shards(root_path)

        visual_weights = {k: v for k, v in raw_weights.items() if k.startswith("model.visual")}

        if not visual_weights:
            return None

        mapping = FIBOWeightMapping.get_vlm_visual_mapping(depth=depth)
        mapped_weights = WeightMapper.apply_mapping(visual_weights, mapping, num_blocks=depth)

        return mapped_weights if mapped_weights else None

    @staticmethod
    def load_vlm_regular_weights(
        repo_id: str = "briaai/FIBO-vlm",
        local_path: str | None = None,
    ) -> "FIBOWeightHandler":
        from transformers import Qwen3VLForConditionalGeneration

        if local_path:
            model = Qwen3VLForConditionalGeneration.from_pretrained(local_path, torch_dtype=torch.bfloat16)
        else:
            model = Qwen3VLForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.bfloat16)

        text_config = model.config.text_config
        rope_params = getattr(text_config, "rope_parameters", None) or {}

        mrope_section = rope_params.get("mrope_section")
        if mrope_section is None:
            mrope_section = [24, 20, 20]

        rope_theta = rope_params.get("rope_theta")
        if rope_theta is None:
            rope_theta = getattr(text_config, "rope_theta", 5000000.0)

        rope_type = rope_params.get("rope_type", "default")
        attention_scaling = 1.0

        config = {
            "vocab_size": text_config.vocab_size,
            "hidden_size": text_config.hidden_size,
            "num_hidden_layers": text_config.num_hidden_layers,
            "num_attention_heads": text_config.num_attention_heads,
            "num_key_value_heads": getattr(text_config, "num_key_value_heads", text_config.num_attention_heads),
            "intermediate_size": text_config.intermediate_size,
            "max_position_embeddings": text_config.max_position_embeddings,
            "rope_theta": rope_theta,
            "rms_norm_eps": getattr(text_config, "rms_norm_eps", 1e-6),
            "head_dim": getattr(text_config, "head_dim", 128),
            "attention_bias": getattr(text_config, "attention_bias", False),
            "attention_dropout": getattr(text_config, "attention_dropout", 0.0),
            "rope_type": rope_type,
            "mrope_section": mrope_section,
            "attention_scaling": attention_scaling,
            "image_token_id": model.config.image_token_id,
        }

        if hasattr(model.config, "vision_config") and model.config.vision_config is not None:
            vision_config = model.config.vision_config
            config["vision_config"] = {
                "patch_size": vision_config.patch_size,
                "temporal_patch_size": vision_config.temporal_patch_size,
                "in_channels": vision_config.in_channels,
                "hidden_size": vision_config.hidden_size,
                "num_heads": vision_config.num_heads,
                "intermediate_size": vision_config.intermediate_size,
                "depth": vision_config.depth,
                "spatial_merge_size": vision_config.spatial_merge_size,
                "num_position_embeddings": vision_config.num_position_embeddings,
                "out_hidden_size": vision_config.out_hidden_size,
                "deepstack_visual_indexes": list(vision_config.deepstack_visual_indexes),
                "hidden_act": vision_config.hidden_act,
            }

        state_dict = model.state_dict()
        num_layers = text_config.num_hidden_layers

        raw_weights: dict[str, mx.array] = {}
        for key, tensor in state_dict.items():
            if not key.startswith("model.language_model") and not key.startswith("lm_head"):
                continue

            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float16)

            raw_weights[key] = mx.array(tensor.detach().cpu().numpy())

        mapping = FIBOWeightMapping.get_vlm_decoder_mapping(num_layers=num_layers)
        decoder_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_blocks=num_layers)
        visual_weights = FIBOWeightHandler.load_vlm_visual_weights(repo_id, local_path)

        handler = FIBOWeightHandler(
            decoder=decoder_weights,
            visual=visual_weights,
            config=config,
            meta_data=MetaData(
                quantization_level=None,
                scale=None,
                is_lora=False,
                mflux_version=None,
            ),
        )
        return handler
