class MFluxException(Exception):
    """base class for all custom exceptions in mflux package."""


class ImageSavingException(MFluxException):
    """error ocurred while attempting to save image to storage."""


class MetadataEmbedException(MFluxException):
    """error ocurred while attempting to embed metadata in image"""


class MFluxUserException(MFluxException):
    """an exception raised by user behavior or intention."""


class StopImageGenerationException(MFluxUserException):
    """user has requested to stop a image generation in progress."""
