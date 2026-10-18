import dataclasses

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

SMALL_DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
# The repo root holds three checkpoints. Only this one has diffusers keys for the whole VAE
# (the unchanged encoder, the small decoder and its own post_quant_conv). small_decoder.safetensors
# uses the BFL key names, so the loader must not read every *.safetensors in the folder.
SMALL_DECODER_FILE = "diffusion_pytorch_model.safetensors"
SMALL_DECODER_BLOCK_OUT_CHANNELS = (96, 192, 384, 384)


class Flux2SmallDecoder:
    @staticmethod
    def component() -> ComponentDefinition:
        vae = next(c for c in Flux2KleinWeightDefinition.get_components() if c.name == "vae")
        return dataclasses.replace(vae, hf_subdir="", weight_files=[SMALL_DECODER_FILE])

    @staticmethod
    def load_vae(bits: int | None = None) -> Flux2VAE:
        # The whole VAE is replaced, as in diffusers (`vae=AutoencoderKLFlux2.from_pretrained(...)`):
        # post_quant_conv was trained with the small decoder and differs from the one in klein.
        component = Flux2SmallDecoder.component()
        weights = WeightLoader.load_single(component, SMALL_DECODER_REPO, file_pattern=SMALL_DECODER_FILE)
        vae = Flux2VAE(decoder_block_out_channels=SMALL_DECODER_BLOCK_OUT_CHANNELS)
        Flux2SmallDecoder.check_weights(vae, weights.components.get(component.name) or {})
        WeightApplier.apply_and_quantize_single(
            weights,
            vae,
            component,
            bits,
            quantization_predicate=Flux2KleinWeightDefinition.quantization_predicate,
        )
        mx.eval(vae.parameters())
        return vae

    @staticmethod
    def check_weights(vae: Flux2VAE, weights: dict) -> None:
        # Module.update(strict=False) accepts missing keys and wrong shapes without an error.
        expected = {path: tuple(value.shape) for path, value in tree_flatten(vae.parameters())}
        supplied = {path: tuple(value.shape) for path, value in tree_flatten(weights)}
        missing = sorted(set(expected) - set(supplied))
        unexpected = sorted(set(supplied) - set(expected))
        wrong_shape = sorted(p for p in set(expected) & set(supplied) if expected[p] != supplied[p])
        if missing or unexpected or wrong_shape:
            raise ValueError(
                f"The {SMALL_DECODER_REPO} checkpoint does not fit the small FLUX.2 VAE. "
                f"Missing: {missing[:5]} ({len(missing)}). Unexpected: {unexpected[:5]} ({len(unexpected)}). "
                f"Wrong shape: {[(p, supplied[p], expected[p]) for p in wrong_shape[:5]]} ({len(wrong_shape)})."
            )
