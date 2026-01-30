"""Z-Image model constants.

Centralized constants for validation limits and model configuration.
These are shared across multiple Z-Image modules to ensure consistency.
"""

# Batch/generation limits
MAX_BATCH_SIZE = 64  # Maximum seeds per batch or sequential generation
MAX_PROMPT_LENGTH = 10000  # Maximum characters per prompt

# Image dimension limits (in pixels)
MAX_DIMENSION = 4096  # Maximum height/width
MIN_DIMENSION = 64  # Minimum height/width
