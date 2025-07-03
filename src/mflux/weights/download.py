"""Download utilities for mflux weights with cache-first behavior."""

from huggingface_hub import snapshot_download as hf_snapshot_download
from huggingface_hub.utils import LocalEntryNotFoundError


def snapshot_download(repo_id: str, fallback_repo_id="black-forest-labs/FLUX.1-dev", **kwargs) -> str:
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
        cache_path = hf_snapshot_download(repo_id=repo_id, **cache_kwargs)
        return cache_path
    except LocalEntryNotFoundError:
        # Cache doesn't exist, download from Hugging Face
        # optimistically assuming provider has the target files
        download_kwargs = kwargs.copy()
        download_kwargs["local_files_only"] = False
        try:
            cache_path = hf_snapshot_download(repo_id=repo_id, **cache_kwargs)
            print(f"{repo_id} cache found for {kwargs}")
            return cache_path
        except LocalEntryNotFoundError:
            # most likely: third party derivative model re-uses official resources
            # TODO: does dev/schnell official resources differ? if so need a repo_id -> fallback map
            #       similar to model/base_model setup
            cache_path = snapshot_download(fallback_repo_id, **kwargs)
            print(f"{repo_id} serving as fallback repo_id for resources {kwargs.get('allow_patterns', '')}")
            return cache_path
