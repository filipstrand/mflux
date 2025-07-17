class MFluxException(Exception):
    """base class for all custom exceptions in mflux package."""


class ImageSavingException(MFluxException):
    """error occurred while attempting to save image to storage."""


class MetadataEmbedException(MFluxException):
    """error occurred while attempting to embed metadata in image"""


class MFluxUserException(MFluxException):
    """an exception raised by user behavior or intention."""


class PromptFileReadError(MFluxUserException):
    """Exception raised for errors in reading the prompt file."""


class StopImageGenerationException(MFluxUserException):
    """user has requested to stop a image generation in progress."""


class StopTrainingException(MFluxUserException):
    """user has requested to stop the training process in progress."""


class CommandExecutionError(MFluxException):
    """Raised when a subprocess command invoked by mflux tooling fails."""

    def __init__(self, cmd: list[str], return_code: int, stdout: str | None, stderr: str | None):
        self.cmd = cmd
        self.return_code = return_code
        self.stdout = stdout or ""
        self.stderr = stderr or ""
        super().__init__(
            f"Command '{' '.join(cmd)}' exited with code {return_code}. See .stdout / .stderr for details."
        )
