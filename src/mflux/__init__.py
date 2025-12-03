import os

# Set TOKENIZERS_PARALLELISM to avoid fork warning
# This must be set before any tokenizers are imported/used
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Export main classes for convenient import
from mflux.models.flux.variants.txt2img.flux import Flux1

__all__ = ["Flux1"]
