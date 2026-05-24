from argparse import ArgumentParser

from mflux.models.common.config import ModelConfig

FLUX2_MODEL_CHOICES = "flux2-klein-4b, flux2-klein-9b, flux2-klein-base-4b, flux2-klein-base-9b"


def ensure_flux2_model(parser: ArgumentParser, model_config: ModelConfig) -> None:
    if not is_flux2_model(model_config):
        parser.error(f"This command only supports FLUX.2 Klein models. Use one of: {FLUX2_MODEL_CHOICES}.")


def is_flux2_model(model_config: ModelConfig) -> bool:
    return any(_is_flux2_identifier(identifier) for identifier in _model_identifiers(model_config))


def is_flux2_base_model(model_config: ModelConfig) -> bool:
    return any(_is_flux2_base_identifier(identifier) for identifier in _model_identifiers(model_config))


def _model_identifiers(model_config: ModelConfig) -> tuple[str | None, str | None]:
    return model_config.model_name, model_config.base_model


def _is_flux2_identifier(identifier: str | None) -> bool:
    return bool(identifier and "flux.2" in identifier.lower())


def _is_flux2_base_identifier(identifier: str | None) -> bool:
    return bool(identifier and "flux.2-klein-base" in identifier.lower())
