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
    @patch.object(TokenizerLoader, "_is_tokenizer_loadable")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_full_cache_uses_primary_without_redownload(self, mock_download, mock_is_loadable, tmp_path):
        cached_root = tmp_path / "cached"
        (cached_root / "tokenizer").mkdir(parents=True)
        mock_download.return_value = str(cached_root)
        mock_is_loadable.side_effect = lambda path, tokenizer_class: path == cached_root / "tokenizer"

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=["tokenizer/**", "text_encoder/**"],
            tokenizer_class="AutoTokenizer",
        )

        assert result == cached_root / "tokenizer"
        assert mock_download.call_count == 1
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_is_tokenizer_loadable")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_partial_cache_redownloads_missing_primary_tokenizer(
        self,
        mock_download,
        mock_is_loadable,
        tmp_path,
    ):
        partial_root = tmp_path / "partial"
        full_root = tmp_path / "full"
        (partial_root / "text_encoder").mkdir(parents=True)
        (full_root / "tokenizer").mkdir(parents=True)
        mock_download.side_effect = [str(partial_root), str(full_root)]
        mock_is_loadable.side_effect = lambda path, tokenizer_class: path == full_root / "tokenizer"

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=["tokenizer/**", "text_encoder/**"],
            tokenizer_class="AutoTokenizer",
        )

        assert result == full_root / "tokenizer"
        assert mock_download.call_count == 2
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True
        assert "local_files_only" not in mock_download.call_args_list[1].kwargs

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_is_tokenizer_loadable")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_no_cache_downloads_tokenizer(self, mock_download, mock_is_loadable, tmp_path):
        downloaded_root = tmp_path / "downloaded"
        (downloaded_root / "tokenizer").mkdir(parents=True)
        mock_download.side_effect = [LocalEntryNotFoundError("no cache"), str(downloaded_root)]
        mock_is_loadable.side_effect = lambda path, tokenizer_class: path == downloaded_root / "tokenizer"

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=None,
            download_patterns=["tokenizer/**"],
            tokenizer_class="AutoTokenizer",
        )

        assert result == downloaded_root / "tokenizer"
        assert mock_download.call_count == 2
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True
        assert "local_files_only" not in mock_download.call_args_list[1].kwargs

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_is_tokenizer_loadable")
    def test_fallback_with_tokenizer_files_is_used_for_local_path(self, mock_is_loadable, tmp_path):
        model_root = tmp_path / "model"
        model_root.mkdir()
        (model_root / "text_encoder").mkdir(parents=True)
        mock_is_loadable.side_effect = lambda path, tokenizer_class: path == model_root / "text_encoder"

        result = TokenizerLoader._resolve_path(
            model_path=str(model_root),
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=None,
            tokenizer_class="AutoTokenizer",
        )

        assert result == model_root / "text_encoder"

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_is_tokenizer_loadable", return_value=False)
    def test_fallback_without_tokenizer_files_raises_clear_error(self, _mock_is_loadable, tmp_path):
        model_root = tmp_path / "model"
        model_root.mkdir()
        _touch(model_root / "text_encoder" / "model.safetensors")

        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path=str(model_root),
                hf_subdir="tokenizer",
                fallback_subdirs=["text_encoder", "."],
                download_patterns=None,
                tokenizer_class="AutoTokenizer",
            )

        assert "No usable tokenizer files" in str(exc_info.value)
        assert "text_encoder" in str(exc_info.value)

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_is_tokenizer_loadable")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_offline_partial_cache_raises_targeted_error(self, mock_download, mock_is_loadable, tmp_path):
        partial_root = tmp_path / "partial"
        (partial_root / "text_encoder").mkdir(parents=True)
        mock_download.side_effect = [str(partial_root), OSError("offline")]
        mock_is_loadable.side_effect = lambda path, tokenizer_class: path == partial_root / "text_encoder"

        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path="org/model",
                hf_subdir="tokenizer",
                fallback_subdirs=["text_encoder", "."],
                download_patterns=["tokenizer/**", "text_encoder/**"],
                tokenizer_class="AutoTokenizer",
            )

        assert "Incomplete Hugging Face tokenizer cache" in str(exc_info.value)
        assert "Re-run with network access" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, OSError)

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_load_raw_tokenizer", return_value=object())
    def test_tokenizer_model_counts_as_usable_primary(self, mock_load_raw, tmp_path):
        model_root = tmp_path / "model"
        model_root.mkdir()
        _touch(model_root / "tokenizer" / "tokenizer.model")

        assert TokenizerLoader._is_tokenizer_loadable(model_root / "tokenizer", "AutoTokenizer")
        mock_load_raw.assert_called_once_with(
            tokenizer_path=model_root / "tokenizer",
            tokenizer_class="AutoTokenizer",
        )

    @pytest.mark.fast
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_first_download_failure_is_not_reported_as_incomplete_cache(self, mock_download):
        mock_download.side_effect = [LocalEntryNotFoundError("no cache"), RuntimeError("repo missing")]

        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path="org/model",
                hf_subdir="tokenizer",
                fallback_subdirs=None,
                download_patterns=["tokenizer/**"],
                tokenizer_class="AutoTokenizer",
            )

        assert "Failed to download tokenizer files" in str(exc_info.value)
        assert "Incomplete Hugging Face tokenizer cache" not in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
