import logging

import mlx.core as mx
import torch
from mlx.utils import tree_unflatten
from safetensors import safe_open

logger = logging.getLogger(__name__)


# This script is based on `convert_flux_lora.py` from `kohya-ss/sd-scripts`.
# For more info, see: https://github.com/kohya-ss/sd-scripts/blob/sd3/networks/convert_flux_lora.py


class LoRAConverter:
    @staticmethod
    def load_weights(lora_path: str) -> dict:
        state_dict = LoRAConverter._load_pytorch_weights(lora_path)
        state_dict = LoRAConverter._convert_weights_to_diffusers(state_dict)
        state_dict = LoRAConverter._convert_to_mlx(state_dict)
        state_dict = list(state_dict.items())
        state_dict = tree_unflatten(state_dict)
        return state_dict

    @staticmethod
    def _load_pytorch_weights(lora_path: str) -> dict:
        state_dict = {}
        with safe_open(lora_path, framework="pt") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        return state_dict

    @staticmethod
    def _convert_weights_to_diffusers(source: dict) -> dict:
        target = {}
        for i in range(19):
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_double_blocks_{i}_img_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_out.0",
            )
            LoRAConverter._convert_to_diffusers_cat(
                source,
                target,
                f"lora_unet_double_blocks_{i}_img_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.to_q",
                    f"transformer.transformer_blocks.{i}.attn.to_k",
                    f"transformer.transformer_blocks.{i}.attn.to_v",
                ],
            )
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_double_blocks_{i}_img_mlp_0",
                f"transformer.transformer_blocks.{i}.ff.net.0.proj",
            )
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_double_blocks_{i}_img_mlp_2",
                f"transformer.transformer_blocks.{i}.ff.net.2",
            )
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_double_blocks_{i}_img_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1.linear",
            )
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_double_blocks_{i}_txt_attn_proj",
                f"transformer.transformer_blocks.{i}.attn.to_add_out",
            )
            LoRAConverter._convert_to_diffusers_cat(
                source,
                target,
                f"lora_unet_double_blocks_{i}_txt_attn_qkv",
                [
                    f"transformer.transformer_blocks.{i}.attn.add_q_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_k_proj",
                    f"transformer.transformer_blocks.{i}.attn.add_v_proj",
                ],
            )
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_double_blocks_{i}_txt_mlp_0",
                f"transformer.transformer_blocks.{i}.ff_context.net.0.proj",
            )
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_double_blocks_{i}_txt_mlp_2",
                f"transformer.transformer_blocks.{i}.ff_context.net.2",
            )
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_double_blocks_{i}_txt_mod_lin",
                f"transformer.transformer_blocks.{i}.norm1_context.linear",
            )

        for i in range(38):
            LoRAConverter._convert_to_diffusers_cat(
                source,
                target,
                f"lora_unet_single_blocks_{i}_linear1",
                [
                    f"transformer.single_transformer_blocks.{i}.attn.to_q",
                    f"transformer.single_transformer_blocks.{i}.attn.to_k",
                    f"transformer.single_transformer_blocks.{i}.attn.to_v",
                    f"transformer.single_transformer_blocks.{i}.proj_mlp",
                ],
                dims=[3072, 3072, 3072, 12288],
            )
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_single_blocks_{i}_linear2",
                f"transformer.single_transformer_blocks.{i}.proj_out",
            )
            LoRAConverter._convert_to_diffusers(
                source,
                target,
                f"lora_unet_single_blocks_{i}_modulation_lin",
                f"transformer.single_transformer_blocks.{i}.norm.linear",
            )

        if len(source) > 0:
            logger.warning(f"Unsupported keys for diffusers: {source.keys()}")
        return target

    @staticmethod
    def _convert_to_diffusers(source: dict, target: dict, source_key: str, target_key: str):
        if source_key + ".lora_down.weight" not in source:
            return
        down_weight = source.pop(source_key + ".lora_down.weight")

        # scale weight by alpha and dim
        rank = down_weight.shape[0]
        alpha = source.pop(source_key + ".alpha").item()  # alpha is scalar
        scale = alpha / rank  # LoRA is scaled by 'alpha / rank' in forward pass, so we need to scale it back here

        # calculate scale_down and scale_up to keep the same value. if scale is 4, scale_down is 2 and scale_up is 2
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        target[target_key + ".lora_A.weight"] = down_weight * scale_down
        target[target_key + ".lora_B.weight"] = source.pop(source_key + ".lora_up.weight") * scale_up

    @staticmethod
    def _convert_to_diffusers_cat(
        source: dict,
        target: dict,
        source_key: str,
        target_keys: list[str],
        dims=None,
    ):
        if source_key + ".lora_down.weight" not in source:
            return
        down_weight = source.pop(source_key + ".lora_down.weight")
        up_weight = source.pop(source_key + ".lora_up.weight")
        source_lora_rank = down_weight.shape[0]

        # scale weight by alpha and dim
        alpha = source.pop(source_key + ".alpha")
        scale = alpha / source_lora_rank

        # calculate scale_down and scale_up
        scale_down = scale
        scale_up = 1.0
        while scale_down * 2 < scale_up:
            scale_down *= 2
            scale_up /= 2

        down_weight = down_weight * scale_down
        up_weight = up_weight * scale_up

        # calculate dims if not provided
        num_splits = len(target_keys)
        if dims is None:
            dims = [up_weight.shape[0] // num_splits] * num_splits
        else:
            assert sum(dims) == up_weight.shape[0]

        # check up-weight is sparse or not
        is_sparse = False
        if source_lora_rank % num_splits == 0:
            diffusers_rank = source_lora_rank // num_splits
            is_sparse = True
            i = 0
            for j in range(len(dims)):
                for k in range(len(dims)):
                    if j == k:
                        continue
                    is_sparse = is_sparse and torch.all(
                        up_weight[
                            i : i + dims[j],
                            k * diffusers_rank : (k + 1) * diffusers_rank,
                        ]
                        == 0
                    )
                i += dims[j]
            if is_sparse:
                logger.info(f"weight is sparse: {source_key}")

        # make diffusers weight
        diffusers_down_keys = [k + ".lora_A.weight" for k in target_keys]
        diffusers_up_keys = [k + ".lora_B.weight" for k in target_keys]
        if not is_sparse:
            # down_weight is copied to each split
            target.update({k: down_weight for k in diffusers_down_keys})

            # up_weight is split to each split
            target.update({k: v for k, v in zip(diffusers_up_keys, torch.split(up_weight, dims, dim=0))})
        else:
            # down_weight is chunked to each split
            target.update(
                {
                    k: v
                    for k, v in zip(
                        diffusers_down_keys,
                        torch.chunk(down_weight, num_splits, dim=0),
                    )
                }
            )

            # up_weight is sparse: only non-zero values are copied to each split
            i = 0
            for j in range(len(dims)):
                target[diffusers_up_keys[j]] = up_weight[
                    i : i + dims[j],
                    j * diffusers_rank : (j + 1) * diffusers_rank,
                ].contiguous()
                i += dims[j]

    @staticmethod
    def _convert_to_mlx(torch_dict: dict):
        mlx_dict = {}
        for key, value in torch_dict.items():
            if isinstance(value, torch.Tensor):
                mlx_dict[key] = mx.array(value.detach().cpu())
            else:
                mlx_dict[key] = value
        return mlx_dict
