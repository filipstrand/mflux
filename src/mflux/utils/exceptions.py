class MFluxException(Exception):
    pass


class ImageSavingException(MFluxException):
    pass


class MetadataEmbedException(MFluxException):
    pass


class MFluxUserException(MFluxException):
    pass


class PromptFileReadError(MFluxUserException):
    pass


class StopImageGenerationException(MFluxUserException):
    pass


class StopTrainingException(MFluxUserException):
    pass


class CommandExecutionError(MFluxException):
    def __init__(self, cmd: list[str], return_code: int, stdout: str | None, stderr: str | None):
        self.cmd = cmd
        self.return_code = return_code
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        super().__init__(
            f"Command '{' '.join(cmd)}' exited with code {return_code}. See .stdout / .stderr for details."
        )


class ReferenceVsOutputImageError(AssertionError):
    pass


class ModelConfigError(ValueError):
    pass


class InvalidBaseModel(ModelConfigError):
    pass
