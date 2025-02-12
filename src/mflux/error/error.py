class ModelConfigError(ValueError):
    """User error in model config."""


class InvalidBaseModel(ModelConfigError):
    """Invalid base model, cannot infer model properties."""
