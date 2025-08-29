import json
from pathlib import Path

import mlx.core as mx
from safetensors.mlx import load_file as mlx_load_file

from mflux.models.flux.weights.weight_handler import MetaData, WeightHandler


class QwenWeightHandler:
    def __init__(
        self,
        meta_data: MetaData,
        qwen_text_encoder: dict | None = None,
        transformer: dict | None = None,
        vae: dict | None = None,
    ):
        self.qwen_text_encoder = qwen_text_encoder
        self.transformer = transformer
        self.vae = vae
        self.meta_data = meta_data

    def num_transformer_blocks(self) -> int:
        return len(self.transformer["transformer_blocks"]) if self.transformer and "transformer_blocks" in self.transformer else 0

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> "QwenWeightHandler":
        # Load the weights from disk, huggingface cache, or download from huggingface
        root_path = Path(local_path) if local_path else WeightHandler.download_or_get_cached_weights(repo_id)

        # Load the weights
        transformer, quantization_level, mflux_version = QwenWeightHandler.load_transformer(root_path=root_path)
        qwen_text_encoder, _, _ = QwenWeightHandler._load_qwen_text_encoder(root_path=root_path)
        vae, _, _ = QwenWeightHandler._load_vae(root_path=root_path)

        return QwenWeightHandler(
            qwen_text_encoder=qwen_text_encoder,
            transformer=transformer,
            vae=vae,
            meta_data=MetaData(
                quantization_level=quantization_level,
                scale=None,
                is_lora=False,
                mflux_version=mflux_version,
            ),
        )

    @staticmethod
    def load_transformer(root_path: Path) -> tuple[dict, int | None, str | None]:
        flat = QwenWeightHandler._load_safetensors_shards(root_path / "transformer", loading_mode="multi_glob")
        mapped_weights = QwenWeightHandler._manual_transformer_mapping(flat)
        return mapped_weights, None, None

    @staticmethod
    def _load_qwen_text_encoder(root_path: Path) -> tuple[dict, int | None, str | None]:
        all_weights = QwenWeightHandler._load_safetensors_shards(root_path / "text_encoder", loading_mode="multi_json")
        mapped_weights = QwenWeightHandler._manual_text_encoder_mapping(all_weights)
        return mapped_weights, None, None

    @staticmethod
    def _load_vae(root_path: Path) -> tuple[dict, int | None, str | None]:
        weights = QwenWeightHandler._load_safetensors_shards(root_path / "vae", loading_mode="single")
        mapped_weights = QwenWeightHandler._manual_flux_style_mapping(weights)
        return mapped_weights, None, None

    @staticmethod
    def _manual_flux_style_mapping(diffusers_weights: dict) -> dict:
        weights = {}

        # 1. Simple direct mappings (like Flux does)
        weights["decoder"] = {}

        # conv_in: decoder.conv_in.weight -> decoder.conv_in.conv3d.weight
        weights["decoder"]["conv_in"] = {"conv3d": {
            "weight": diffusers_weights["decoder.conv_in.weight"],
            "bias": diffusers_weights["decoder.conv_in.bias"]
        }}

        # conv_out: decoder.conv_out.weight -> decoder.conv_out.conv3d.weight
        weights["decoder"]["conv_out"] = {"conv3d": {
            "weight": diffusers_weights["decoder.conv_out.weight"],
            "bias": diffusers_weights["decoder.conv_out.bias"]
        }}

        # norm_out: decoder.norm_out.gamma -> decoder.norm_out.weight
        weights["decoder"]["norm_out"] = {
            "weight": diffusers_weights["decoder.norm_out.gamma"]
        }

        # post_quant_conv: post_quant_conv.weight -> post_quant_conv.conv3d.weight
        weights["post_quant_conv"] = {"conv3d": {
            "weight": diffusers_weights["post_quant_conv.weight"],
            "bias": diffusers_weights["post_quant_conv.bias"]
        }}

        # 2. Mid block (manual structure building)
        weights["decoder"]["mid_block"] = {}

        # mid_block.resnets (list of 2 resnets)
        weights["decoder"]["mid_block"]["resnets"] = [{}, {}]
        for i in range(2):
            resnet = weights["decoder"]["mid_block"]["resnets"][i]
            # conv1.weight -> conv1.conv3d.weight
            resnet["conv1"] = {"conv3d": {
                "weight": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv1.weight"],
                "bias": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv1.bias"]
            }}
            # conv2.weight -> conv2.conv3d.weight
            resnet["conv2"] = {"conv3d": {
                "weight": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv2.weight"],
                "bias": diffusers_weights[f"decoder.mid_block.resnets.{i}.conv2.bias"]
            }}
            # norm1.gamma -> norm1.weight
            resnet["norm1"] = {"weight": diffusers_weights[f"decoder.mid_block.resnets.{i}.norm1.gamma"]}
            resnet["norm2"] = {"weight": diffusers_weights[f"decoder.mid_block.resnets.{i}.norm2.gamma"]}

        # mid_block.attentions (list of 1 attention)
        weights["decoder"]["mid_block"]["attentions"] = [{}]
        attn = weights["decoder"]["mid_block"]["attentions"][0]
        attn["norm"] = {"weight": diffusers_weights["decoder.mid_block.attentions.0.norm.gamma"]}
        # Note: to_qkv and proj are Conv2d, not Conv3d - no conv3d wrapper
        attn["to_qkv"] = {
            "weight": diffusers_weights["decoder.mid_block.attentions.0.to_qkv.weight"],
            "bias": diffusers_weights["decoder.mid_block.attentions.0.to_qkv.bias"]
        }
        attn["proj"] = {
            "weight": diffusers_weights["decoder.mid_block.attentions.0.proj.weight"],
            "bias": diffusers_weights["decoder.mid_block.attentions.0.proj.bias"]
        }

        # 3. Up blocks (manual structure building)
        for block_idx in range(4):
            up_block_key = f"up_block{block_idx}"
            weights["decoder"][up_block_key] = {}

            # resnets (list of 3 resnets)
            weights["decoder"][up_block_key]["resnets"] = [{}, {}, {}]
            for res_idx in range(3):
                resnet = weights["decoder"][up_block_key]["resnets"][res_idx]

                # conv1.weight -> conv1.conv3d.weight
                resnet["conv1"] = {"conv3d": {
                    "weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv1.weight"],
                    "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv1.bias"]
                }}
                # conv2.weight -> conv2.conv3d.weight
                resnet["conv2"] = {"conv3d": {
                    "weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv2.weight"],
                    "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv2.bias"]
                }}
                # norm1.gamma -> norm1.weight
                resnet["norm1"] = {"weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.norm1.gamma"]}
                resnet["norm2"] = {"weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.norm2.gamma"]}

                # Handle optional conv_shortcut -> skip_conv (only exists for some resnets)
                shortcut_key = f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv_shortcut.weight"
                if shortcut_key in diffusers_weights:
                    resnet["skip_conv"] = {"conv3d": {
                        "weight": diffusers_weights[shortcut_key],
                        "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.resnets.{res_idx}.conv_shortcut.bias"]
                    }}

            # upsamplers (only for blocks 0, 1, 2)
            if block_idx <= 2:
                weights["decoder"][up_block_key]["upsamplers"] = [{}]
                upsampler = weights["decoder"][up_block_key]["upsamplers"][0]

                # resample.1.weight -> resample_conv.weight (Conv2d, no conv3d wrapper)
                upsampler["resample_conv"] = {
                    "weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.resample.1.weight"],
                    "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.resample.1.bias"]
                }

                # time_conv (only for blocks 0, 1)
                if block_idx <= 1:
                    upsampler["time_conv"] = {"conv3d": {
                        "weight": diffusers_weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.time_conv.weight"],
                        "bias": diffusers_weights[f"decoder.up_blocks.{block_idx}.upsamplers.0.time_conv.bias"]
                    }}

        return weights

    @staticmethod
    def _manual_transformer_mapping(diffusers_weights: dict) -> dict:
        weights = {}

        # 1. Top-level mappings (exact MLX parameter names)
        weights["img_in"] = {
            "weight": diffusers_weights["img_in.weight"],
            "bias": diffusers_weights["img_in.bias"]
        }
        weights["txt_norm"] = {"weight": diffusers_weights["txt_norm.weight"]}
        weights["txt_in"] = {
            "weight": diffusers_weights["txt_in.weight"],
            "bias": diffusers_weights["txt_in.bias"]
        }

        # 2. Time text embedder (exact MLX structure)
        weights["time_text_embed"] = {"timestep_embedder": {
            "linear_1": {
                "weight": diffusers_weights["time_text_embed.timestep_embedder.linear_1.weight"],
                "bias": diffusers_weights["time_text_embed.timestep_embedder.linear_1.bias"]
            },
            "linear_2": {
                "weight": diffusers_weights["time_text_embed.timestep_embedder.linear_2.weight"],
                "bias": diffusers_weights["time_text_embed.timestep_embedder.linear_2.bias"]
            }
        }}

        # 3. Output head (exact MLX structure)
        weights["norm_out"] = {
            "linear": {
                "weight": diffusers_weights["norm_out.linear.weight"],
                "bias": diffusers_weights["norm_out.linear.bias"]
            }
        }
        weights["proj_out"] = {
            "weight": diffusers_weights["proj_out.weight"],
            "bias": diffusers_weights["proj_out.bias"]
        }

        # 4. Transformer blocks (exact MLX parameter names - no applier needed!)
        transformer_blocks = []
        for block_idx in range(60):  # 60 blocks based on debug output
            block = {}

            # Modulation layers (exact MLX parameter names)
            block["img_mod_linear"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.img_mod.1.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.img_mod.1.bias"]
            }
            block["txt_mod_linear"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mod.1.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mod.1.bias"]
            }

            # Attention layers (exact MLX parameter names - no nesting under "attn")
            block["to_q"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_q.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_q.bias"]
            }
            block["to_k"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_k.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_k.bias"]
            }
            block["to_v"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_v.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_v.bias"]
            }
            block["add_q_proj"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_q_proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_q_proj.bias"]
            }
            block["add_k_proj"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_k_proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_k_proj.bias"]
            }
            block["add_v_proj"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_v_proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.add_v_proj.bias"]
            }

            # Attention norms (exact MLX parameter names)
            block["norm_q"] = {"weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.norm_q.weight"]}
            block["norm_k"] = {"weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.norm_k.weight"]}
            block["norm_added_q"] = {"weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.norm_added_q.weight"]}
            block["norm_added_k"] = {"weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.norm_added_k.weight"]}

            # Attention outputs (exact MLX parameter names)
            block["attn_to_out"] = [{
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_out.0.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_out.0.bias"]
            }]
            block["to_add_out"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_add_out.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.attn.to_add_out.bias"]
            }

            # MLP layers (exact MLX parameter names)
            block["img_mlp_in"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.img_mlp.net.0.proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.img_mlp.net.0.proj.bias"]
            }
            block["img_mlp_out"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.img_mlp.net.2.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.img_mlp.net.2.bias"]
            }
            block["txt_mlp_in"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mlp.net.0.proj.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mlp.net.0.proj.bias"]
            }
            block["txt_mlp_out"] = {
                "weight": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mlp.net.2.weight"],
                "bias": diffusers_weights[f"transformer_blocks.{block_idx}.txt_mlp.net.2.bias"]
            }

            transformer_blocks.append(block)

        weights["transformer_blocks"] = transformer_blocks
        return weights

    @staticmethod
    def _load_safetensors_shards(path: Path, loading_mode: str = "multi_glob") -> dict[str, mx.array]:
        all_weights = {}

        if loading_mode == "single":
            # VAE style: Single file loading
            safetensors_files = list(path.glob("*.safetensors"))
            if not safetensors_files:
                raise FileNotFoundError(f"No safetensors files found in {path}")

            weights_file = safetensors_files[0]
            data = mx.load(str(weights_file), return_metadata=True)
            all_weights = dict(data[0].items())

        elif loading_mode == "multi_json":
            # Text encoder style: Use JSON index to map params to files
            index_path = path / "model.safetensors.index.json"
            with open(index_path) as f:
                index = json.load(f)

            # Group weights by file
            files_to_load = {}
            for param_name, file_name in index["weight_map"].items():
                if file_name not in files_to_load:
                    files_to_load[file_name] = []
                files_to_load[file_name].append(param_name)

            # Load weights from each file
            for file_name, param_names in files_to_load.items():
                file_path = path / file_name

                # Load the safetensor file with fallback to torch conversion
                try:
                    file_weights = mlx_load_file(str(file_path))
                except Exception:
                    # If MLX can't load directly, try with torch and convert
                    import torch
                    from safetensors.torch import load_file as torch_load_file

                    torch_weights = torch_load_file(str(file_path))
                    file_weights = {}
                    for name, tensor in torch_weights.items():
                        # Convert to float32 if bfloat16, then to MLX
                        if tensor.dtype == torch.bfloat16:
                            tensor = tensor.to(torch.float32)
                        file_weights[name] = mx.array(tensor.numpy())

                # Add requested parameters to combined weights
                for param_name in param_names:
                    if param_name in file_weights:
                        all_weights[param_name] = file_weights[param_name]
        else:  # "multi_glob"
            # Transformer style: Directly glob all safetensors files
            shard_files = sorted([f for f in path.glob("*.safetensors") if not f.name.startswith("._")])
            if not shard_files:
                raise FileNotFoundError(f"No safetensors found in {path}")

            for shard in shard_files:
                data, metadata = mx.load(str(shard), return_metadata=True)
                all_weights.update(dict(data.items()))

        return all_weights

    @staticmethod
    def _manual_text_encoder_mapping(hf_weights: dict[str, mx.array]) -> dict:
        weights = {}
        converted_count = 0
        skipped_count = 0

        # Skip LM head and vision encoder weights - we only need the text encoder
        filtered_weights = {}
        for hf_name, weight in hf_weights.items():
            if hf_name.startswith("lm_head") or hf_name.startswith("visual."):
                skipped_count += 1
                continue
            filtered_weights[hf_name] = weight

        # 1. Top-level embeddings (exact MLX parameter names)
        weights["encoder"] = {}
        weights["encoder"]["embed_tokens"] = {
            "weight": filtered_weights["model.embed_tokens.weight"]
        }
        converted_count += 1

        # 2. Final norm (exact MLX parameter names)
        weights["encoder"]["norm"] = {
            "weight": filtered_weights["model.norm.weight"]
        }
        converted_count += 1

        # 3. Encoder layers (exact MLX parameter names - 28 layers)
        layers = []
        for layer_idx in range(28):  # 28 text encoder layers
            layer = {}

            # Layer norms
            layer["input_layernorm"] = {
                "weight": filtered_weights[f"model.layers.{layer_idx}.input_layernorm.weight"]
            }
            layer["post_attention_layernorm"] = {
                "weight": filtered_weights[f"model.layers.{layer_idx}.post_attention_layernorm.weight"]
            }
            converted_count += 2

            # Self attention (exact MLX parameter names)
            layer["self_attn"] = {
                "q_proj": {
                    "weight": filtered_weights[f"model.layers.{layer_idx}.self_attn.q_proj.weight"],
                    "bias": filtered_weights[f"model.layers.{layer_idx}.self_attn.q_proj.bias"]
                },
                "k_proj": {
                    "weight": filtered_weights[f"model.layers.{layer_idx}.self_attn.k_proj.weight"],
                    "bias": filtered_weights[f"model.layers.{layer_idx}.self_attn.k_proj.bias"]
                },
                "v_proj": {
                    "weight": filtered_weights[f"model.layers.{layer_idx}.self_attn.v_proj.weight"],
                    "bias": filtered_weights[f"model.layers.{layer_idx}.self_attn.v_proj.bias"]
                },
                "o_proj": {
                    "weight": filtered_weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight"]
                    # Note: o_proj has no bias in MLX structure
                }
            }
            converted_count += 7

            # MLP (exact MLX parameter names)
            layer["mlp"] = {
                "gate_proj": {
                    "weight": filtered_weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight"]
                    # Note: gate_proj has no bias in MLX structure
                },
                "up_proj": {
                    "weight": filtered_weights[f"model.layers.{layer_idx}.mlp.up_proj.weight"]
                    # Note: up_proj has no bias in MLX structure
                },
                "down_proj": {
                    "weight": filtered_weights[f"model.layers.{layer_idx}.mlp.down_proj.weight"]
                    # Note: down_proj has no bias in MLX structure
                }
            }
            converted_count += 3

            layers.append(layer)

        weights["encoder"]["layers"] = layers
        return weights
