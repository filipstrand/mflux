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
        transformer: dict | None = None,
    ):
        self.vae = vae
        self.transformer = transformer
        self.meta_data = meta_data

    @staticmethod
    def load_regular_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> "FIBOWeightHandler":
        # Load weights from PyTorch model
        vae_weights = FIBOWeightHandler._load_vae_weights(repo_id, local_path)
        transformer_weights = FIBOWeightHandler._load_transformer_weights(repo_id, local_path)

        return FIBOWeightHandler(
            vae=vae_weights,
            transformer=transformer_weights,
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
        pipe = FIBOWeightHandler._load_pipeline(repo_id, local_path)

        # Extract encoder / decoder / quant / post-quant weights
        encoder_state_dict = pipe.vae.encoder.state_dict()
        decoder_state_dict = pipe.vae.decoder.state_dict()
        quant_state_dict = pipe.vae.quant_conv.state_dict()
        post_quant_state_dict = pipe.vae.post_quant_conv.state_dict()

        # Convert PyTorch tensors to MLX arrays
        raw_weights = {}
        for key, tensor in encoder_state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            raw_weights[f"encoder.{key}"] = mx.array(tensor.detach().cpu().numpy())

        for key, tensor in decoder_state_dict.items():
            # Convert bfloat16 to float32, then to MLX
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            raw_weights[f"decoder.{key}"] = mx.array(tensor.detach().cpu().numpy())

        for key, tensor in quant_state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            raw_weights[f"quant_conv.{key}"] = mx.array(tensor.detach().cpu().numpy())

        for key, tensor in post_quant_state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            raw_weights[f"post_quant_conv.{key}"] = mx.array(tensor.detach().cpu().numpy())

        # Apply mapping
        # FIBO has 4 up_blocks, each with 3 resnets
        mapping = FIBOWeightMapping.get_vae_mapping()
        mapped_weights = WeightMapper.apply_mapping(raw_weights, mapping, num_blocks=4)

        return mapped_weights

    @staticmethod
    def _load_transformer_weights(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> dict:
        """
        Load the PyTorch FIBO transformer weights and convert them to MLX arrays.

        For the transformer we mirror the diffusers module structure in MLX, so we can
        keep the state_dict keys as-is (no additional mapping required).
        """
        pipe = FIBOWeightHandler._load_pipeline(repo_id, local_path)

        transformer_state_dict = pipe.transformer.state_dict()

        raw_weights: dict[str, mx.array] = {}
        for key, tensor in transformer_state_dict.items():
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.to(torch.float32)
            raw_weights[key] = mx.array(tensor.detach().cpu().numpy())

        # Apply declarative transformer mapping.
        # FIBO has 19 joint + 38 single blocks = 57 caption_projection layers.
        mapping = FIBOWeightMapping.get_transformer_mapping()
        mapped_weights = WeightMapper.apply_mapping(
            raw_weights,
            mapping,
            num_blocks=38,  # cover both transformer_blocks (0-18) and single_transformer_blocks (0-37)
            num_layers=57,  # caption_projection layers
        )

        return mapped_weights

    @staticmethod
    def _load_pipeline(
        repo_id: str | None = None,
        local_path: str | None = None,
    ) -> BriaFiboPipeline:
        if local_path:
            return BriaFiboPipeline.from_pretrained(local_path, torch_dtype=torch.bfloat16)
        return BriaFiboPipeline.from_pretrained(repo_id or "briaai/FIBO", torch_dtype=torch.bfloat16)
