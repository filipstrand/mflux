"""Download utilities for mflux weights with cache-first behavior."""

from huggingface_hub import snapshot_download as hf_snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError


def snapshot_download(repo_id: str, **kwargs) -> str:
    """Download repo files with cache-first behavior.

    This wrapper function attempts to use cached files first before downloading.
    If local_files_only is explicitly set to True, it will not fall back to downloading.
    Otherwise, it will try local cache first, then download if cache is not found.

    Args:
        repo_id: A user or an organization name and a repo name separated by a `/`.
        **kwargs: Additional arguments passed to huggingface_hub.snapshot_download

    Returns:
        str: Path to the downloaded/cached repository

    Docs: https://huggingface.co/docs/huggingface_hub/v0.33.1/en/package_reference/file_download#huggingface_hub.snapshot_download
    """  # fmt: off
    # Extract relevant kwargs for our logic
    force_download = kwargs.get("force_download", False)
    local_files_only = kwargs.get("local_files_only", False)

    # If force_download or local_files_only is explicitly set, use original behavior
    if force_download or local_files_only:
        return hf_snapshot_download(repo_id=repo_id, **kwargs)

    # Try to use cached files first
    try:
        # Override local_files_only to True for cache check
        cache_kwargs = kwargs.copy()
        cache_kwargs["local_files_only"] = True
        return hf_snapshot_download(repo_id=repo_id, **cache_kwargs)
    except LocalEntryNotFoundError:
        # Cache doesn't exist, download from Hugging Face
        download_kwargs = kwargs.copy()
        download_kwargs["local_files_only"] = False
        return hf_snapshot_download(repo_id=repo_id, **download_kwargs)
