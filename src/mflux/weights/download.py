"""Download utilities for mflux weights with cache-first behavior."""

from huggingface_hub import snapshot_download as hf_snapshot_download


def snapshot_download(repo_id: str, **kwargs) -> str:
    """Download repo files with cache-first behavior.

    This wrapper function is a wrapper for upstream hf_snapshot_download

    2025-07-08 HOT FIX: just pass the args through for now, see #235 discussion
    """
    return hf_snapshot_download(repo_id, **kwargs)
