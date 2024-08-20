from enum import Enum


class ModelConfig(Enum):
    FLUX1_DEV = ("black-forest-labs/FLUX.1-dev", "dev", 512)
    FLUX1_SCHNELL = ("black-forest-labs/FLUX.1-schnell", "schnell", 256)

    def __init__(self, model_name: str, alias: str, max_sequence_length: int):
        self.alias = alias
        self.model_name = model_name
        self.max_sequence_length = max_sequence_length

    @staticmethod
    def from_repo(model_name: str) -> "ModelConfig":
        try:
            for model in ModelConfig:
                if model.model_name == model_name:
                    return model
        except KeyError:
            raise ValueError(f"'{model_name}' is not a valid model")

    @staticmethod
    def from_alias(alias: str) -> "ModelConfig":
        try:
            for model in ModelConfig:
                if model.alias == alias:
                    return model
        except KeyError:
            raise ValueError(f"'{alias}' is not a valid model")
