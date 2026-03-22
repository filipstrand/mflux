from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub.utils import LocalEntryNotFoundError

from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


class TestTokenizerResolution:
    @pytest.mark.fast
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_full_cache_uses_primary_without_redownload(self, mock_download, tmp_path):
        cached_root = tmp_path / "cached"
        _touch(cached_root / "tokenizer" / "tokenizer.json")
        mock_download.return_value = str(cached_root)

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=["tokenizer/**", "text_encoder/**"],
        )

        assert result == cached_root / "tokenizer"
        assert mock_download.call_count == 1
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True

    @pytest.mark.fast
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_partial_cache_redownloads_missing_primary_tokenizer(self, mock_download, tmp_path):
        partial_root = tmp_path / "partial"
        full_root = tmp_path / "full"
        _touch(partial_root / "text_encoder" / "model.safetensors")
        _touch(full_root / "tokenizer" / "tokenizer.json")
        mock_download.side_effect = [str(partial_root), str(full_root)]

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=["tokenizer/**", "text_encoder/**"],
        )

        assert result == full_root / "tokenizer"
        assert mock_download.call_count == 2
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True
        assert "local_files_only" not in mock_download.call_args_list[1].kwargs

    @pytest.mark.fast
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_no_cache_downloads_tokenizer(self, mock_download, tmp_path):
        downloaded_root = tmp_path / "downloaded"
        _touch(downloaded_root / "tokenizer" / "tokenizer.json")
        mock_download.side_effect = [LocalEntryNotFoundError("no cache"), str(downloaded_root)]

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=None,
            download_patterns=["tokenizer/**"],
        )

        assert result == downloaded_root / "tokenizer"
        assert mock_download.call_count == 2
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True
        assert "local_files_only" not in mock_download.call_args_list[1].kwargs

    @pytest.mark.fast
    def test_fallback_with_tokenizer_files_is_used_for_local_path(self, tmp_path):
        model_root = tmp_path / "model"
        model_root.mkdir()
        _touch(model_root / "text_encoder" / "tokenizer.json")

        result = TokenizerLoader._resolve_path(
            model_path=str(model_root),
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=None,
        )

        assert result == model_root / "text_encoder"

    @pytest.mark.fast
    def test_fallback_without_tokenizer_files_raises_clear_error(self, tmp_path):
        model_root = tmp_path / "model"
        model_root.mkdir()
        _touch(model_root / "text_encoder" / "model.safetensors")

        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path=str(model_root),
                hf_subdir="tokenizer",
                fallback_subdirs=["text_encoder", "."],
                download_patterns=None,
            )

        assert "No usable tokenizer files" in str(exc_info.value)
        assert "text_encoder" in str(exc_info.value)

    @pytest.mark.fast
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_offline_partial_cache_raises_targeted_error(self, mock_download, tmp_path):
        partial_root = tmp_path / "partial"
        _touch(partial_root / "text_encoder" / "model.safetensors")
        mock_download.side_effect = [str(partial_root), OSError("offline")]

        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path="org/model",
                hf_subdir="tokenizer",
                fallback_subdirs=["text_encoder", "."],
                download_patterns=["tokenizer/**", "text_encoder/**"],
            )

        assert "Incomplete Hugging Face tokenizer cache" in str(exc_info.value)
        assert "Re-run with network access" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, OSError)

    @pytest.mark.fast
    def test_sentencepiece_tokenizer_counts_as_usable_primary(self, tmp_path):
        model_root = tmp_path / "model"
        model_root.mkdir()
        _touch(model_root / "tokenizer_2" / "spiece.model")

        result = TokenizerLoader._resolve_path(
            model_path=str(model_root),
            hf_subdir="tokenizer_2",
            fallback_subdirs=None,
            download_patterns=None,
        )

        assert result == model_root / "tokenizer_2"
