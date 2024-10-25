from enum import Enum


class ModelConfig(Enum):
    # ==== black-forest-labs official models =====
    FLUX1_DEV = ("black-forest-labs/FLUX.1-dev", "dev", 1000, 512)
    FLUX1_SCHNELL = ("black-forest-labs/FLUX.1-schnell", "schnell", 1000, 256)
    # ==== third party compatible models - best effort compatibility =====
    # https://huggingface.co/Freepik/flux.1-lite-8B-alpha distilled from dev
    FLUX1_FREEPIK_LITE_8B_ALPHA = ("Freepik/flux.1-lite-8B-alpha", "freepik-lite-8b-alpha", 1000, 512)

    def __init__(
        self,
        model_name: str,
        alias: str,
        num_train_steps: int,
        max_sequence_length: int,
    ):
        self.alias = alias
        self.model_name = model_name
        self.num_train_steps = num_train_steps
        self.max_sequence_length = max_sequence_length

    @staticmethod
    def from_alias(alias: str) -> "ModelConfig":
        try:
            for model in ModelConfig:
                if model.alias == alias:
                    return model
        except KeyError:
            raise ValueError(f"'{alias}' is not a valid model")
