import warnings

from mflux.models.common.vae.block_decoder import VAEBlockDecoder


class VAEDecodeOptions:
    # One policy for the CLIs of models whose VAE uses VAEBlockDecoder: --vae-full-precision selects the full-image
    # decode, the image metadata records it, and --vae-tiling / --vae-tile-size do not change the decode.
    TILING_FLAGS = ("--vae-tiling", "--vae-tile-size")
    DECODE_TILING_REASON = "This model does not decode in tiles. It decodes in row strips with full-image GroupNorm statistics, or on the full image with --vae-full-precision."  # fmt: skip
    ENCODE_TILING_REASON = "--vae-tiling and --vae-tile-size change only the encode of an input image. The decode does not use tiles. It uses row strips with full-image GroupNorm statistics, or the full image with --vae-full-precision."  # fmt: skip

    @staticmethod
    def ignored_options() -> dict[str, str]:
        # For a CLI that has no input image: the tiling flags change nothing.
        return {flag: VAEDecodeOptions.DECODE_TILING_REASON for flag in VAEDecodeOptions.TILING_FLAGS}

    @staticmethod
    def conditional_options() -> dict[str, dict[str, str]]:
        # For a CLI with an input image: the tiling flags still tile its encode.
        entry = {"condition": "the encode of an input image", "reason": VAEDecodeOptions.ENCODE_TILING_REASON}
        return {flag: dict(entry) for flag in VAEDecodeOptions.TILING_FLAGS}

    @staticmethod
    def warn_encode_only_tiling() -> None:
        from mflux.cli.parser.parsers import CommandLineParser

        if CommandLineParser._option_was_provided(*VAEDecodeOptions.TILING_FLAGS):
            warnings.warn(VAEDecodeOptions.ENCODE_TILING_REASON, stacklevel=2)

    @staticmethod
    def apply(args, model) -> None:
        if getattr(args, "vae_full_precision", False):
            model.vae.decode_mode = VAEBlockDecoder.FULL_PRECISION

    @staticmethod
    def generation_parameters(model) -> dict:
        vae = getattr(model, "vae", None)
        if getattr(vae, "decode_mode", VAEBlockDecoder.LEGACY) == VAEBlockDecoder.FULL_PRECISION:
            return {"vae_full_precision": True}
        return {}
