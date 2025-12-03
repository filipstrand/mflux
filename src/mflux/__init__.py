import os

# Set TOKENIZERS_PARALLELISM to avoid fork warning
# This must be set before any tokenizers are imported/used
if "TOKENIZERS_PARALLELISM" not in os.environ:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
