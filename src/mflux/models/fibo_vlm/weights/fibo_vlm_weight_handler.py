from pathlib import Path

import mlx.core as mx
import torch
from safetensors.torch import load_file as torch_load_file

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.fibo.weights import FIBOWeightHandler
from mflux.models.fibo_vlm.weights.fibo_vlm_weight_mapping import FIBOVLMWeightMapping
from mflux.models.flux.weights.weight_handler import (
    MetaData,
)
from mflux.utils.download import snapshot_download


class FIBOVLMWeightHandler:
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

        mapping = FIBOVLMWeightMapping.get_vlm_decoder_mapping(num_layers=num_layers)
        decoder_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_blocks=num_layers)
        visual_weights = FIBOVLMWeightHandler.load_vlm_visual_weights(repo_id, local_path)

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

        raw_weights = FIBOVLMWeightHandler._load_vlm_safetensors_shards(root_path)

        decoder_weights = {
            k: v for k, v in raw_weights.items() if k.startswith("model.language_model") or k.startswith("lm_head")
        }

        mapping = FIBOVLMWeightMapping.get_vlm_decoder_mapping(num_layers=num_layers)
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

        raw_weights = FIBOVLMWeightHandler._load_vlm_safetensors_shards(root_path)

        visual_weights = {k: v for k, v in raw_weights.items() if k.startswith("model.visual")}

        if not visual_weights:
            return None

        mapping = FIBOVLMWeightMapping.get_vlm_visual_mapping(depth=depth)
        mapped_weights = WeightMapper.apply_mapping(visual_weights, mapping, num_blocks=depth)

        return mapped_weights if mapped_weights else None
