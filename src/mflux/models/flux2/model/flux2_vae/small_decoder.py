import dataclasses

import mlx.core as mx
from mlx.utils import tree_flatten

from mflux.models.common.weights.loading.loaded_weights import LoadedWeights
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.flux2.weights.flux2_weight_definition import Flux2KleinWeightDefinition

SMALL_DECODER_REPO = "black-forest-labs/FLUX.2-small-decoder"
# The repo root holds three checkpoints. Only this one has diffusers keys for the whole VAE
# (the unchanged encoder, the small decoder and its own post_quant_conv; mflux applies the last two).
# small_decoder.safetensors uses the BFL key names, so the loader must not read every *.safetensors in the folder.
SMALL_DECODER_FILE = "diffusion_pytorch_model.safetensors"
SMALL_DECODER_BLOCK_OUT_CHANNELS = (96, 192, 384, 384)


class Flux2SmallDecoderLoader:
    # The parts of the checkpoint that replace the model's own: the small decoder and the post_quant_conv it was
    # trained with (it differs from the one in klein).
    DECODING_PARTS = ("decoder", "post_quant_conv")

    @staticmethod
    def component() -> ComponentDefinition:
        vae = next(c for c in Flux2KleinWeightDefinition.get_components() if c.name == "vae")
        return dataclasses.replace(vae, hf_subdir="", weight_files=[SMALL_DECODER_FILE])

    @staticmethod
    def with_small_decoder(vae: Flux2VAE, bits: int | None = None) -> Flux2VAE:
        # Only decoding changes. Edit references keep being encoded, and latents normalized, by the model's own
        # encoder, quant_conv and bn (those of a saved or quantized model too). The checkpoint's encoder is never
        # applied: its keys are dropped first, and the new VAE's encoder is replaced before anything is evaluated.
        component = Flux2SmallDecoderLoader.component()
        weights = WeightLoader.load_single(component, SMALL_DECODER_REPO, file_pattern=SMALL_DECODER_FILE)
        supplied = weights.components.get(component.name) or {}
        decoding = {part: supplied[part] for part in Flux2SmallDecoderLoader.DECODING_PARTS if part in supplied}
        weights = LoadedWeights(components={component.name: decoding}, meta_data=weights.meta_data)

        small_vae = Flux2VAE(decoder_block_out_channels=SMALL_DECODER_BLOCK_OUT_CHANNELS)
        Flux2SmallDecoderLoader.check_weights(Flux2SmallDecoderLoader._decoding_parameters(small_vae), decoding)
        WeightApplier.apply_and_quantize_single(
            weights,
            small_vae,
            component,
            bits,
            quantization_predicate=Flux2KleinWeightDefinition.quantization_predicate,
        )
        small_vae.encoder, small_vae.quant_conv, small_vae.bn = vae.encoder, vae.quant_conv, vae.bn
        mx.eval(small_vae.parameters())
        return small_vae

    @staticmethod
    def _decoding_parameters(vae: Flux2VAE) -> dict:
        parameters = vae.parameters()
        return {part: parameters[part] for part in Flux2SmallDecoderLoader.DECODING_PARTS}

    @staticmethod
    def check_weights(parameters: dict, weights: dict) -> None:
        # Module.update(strict=False) accepts missing keys and wrong shapes without an error.
        expected = {path: tuple(value.shape) for path, value in tree_flatten(parameters)}
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
