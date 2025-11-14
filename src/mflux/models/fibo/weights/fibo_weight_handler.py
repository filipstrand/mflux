import mlx.core as mx
import torch
from diffusers import BriaFiboPipeline

from mflux.models.common.weights.mapping.weight_mapper import WeightMapper
from mflux.models.fibo.weights.fibo_weight_mapping import FIBOWeightMapping
from mflux.models.flux.weights.weight_handler import MetaData


class FIBOWeightHandler:
    def __init__(
        self,
        meta_data: MetaData,
        vae: dict | None = None,
    ):
        self.vae = vae
        self.meta_data = meta_data

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> "FIBOWeightHandler":
        # Load weights from PyTorch model
        vae_weights = FIBOWeightHandler._load_vae_weights(repo_id, local_path)

        return FIBOWeightHandler(
            vae=vae_weights,
            meta_data=MetaData(
                quantization_level=None,
                scale=None,
                is_lora=False,
                mflux_version=None,
            ),
        )

    @staticmethod
    def _load_vae_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        # Load PyTorch model
        if local_path:
            pipe = BriaFiboPipeline.from_pretrained(local_path, torch_dtype=torch.bfloat16)
        else:
            pipe = BriaFiboPipeline.from_pretrained(repo_id or "briaai/FIBO", torch_dtype=torch.bfloat16)

        # Extract decoder weights
        decoder_state_dict = pipe.vae.decoder.state_dict()
        post_quant_state_dict = pipe.vae.post_quant_conv.state_dict()

        # Convert PyTorch tensors to MLX arrays
        raw_weights = {}
        for key, tensor in decoder_state_dict.items():
            # Convert bfloat16 to float32, then to MLX
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            raw_weights[f"decoder.{key}"] = mx.array(tensor.detach().cpu().numpy())

        for key, tensor in post_quant_state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            raw_weights[f"post_quant_conv.{key}"] = mx.array(tensor.detach().cpu().numpy())

        # Apply mapping
        # FIBO has 4 up_blocks, each with 3 resnets
        mapping = FIBOWeightMapping.get_vae_mapping()
        mapped_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_blocks=4)

        return mapped_weights
