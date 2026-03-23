from pathlib import Path
from unittest.mock import patch

import pytest
from huggingface_hub.utils import LocalEntryNotFoundError
from transformers import BertTokenizer

from mflux.models.common.tokenizer.tokenizer_loader import TokenizerLoader


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _write_real_tokenizer(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    vocab_file = path / "vocab.txt"
    vocab_file.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "hello", "world"]), encoding="utf-8")
    BertTokenizer(vocab_file=str(vocab_file)).save_pretrained(path)


def _write_minimal_qwen2_tokenizer_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "vocab.json").write_text('{"<|endoftext|>": 0, "hello": 1}', encoding="utf-8")
    (path / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}", encoding="utf-8")


class TestTokenizerResolution:
    @pytest.mark.fast
    def test_real_primary_tokenizer_layout_resolves(self, tmp_path):
        model_root = tmp_path / "model"
        tokenizer_path = model_root / "tokenizer"
        _write_real_tokenizer(tokenizer_path)

        result = TokenizerLoader._resolve_path(
            model_path=str(model_root),
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=None,
            tokenizer_class="AutoTokenizer",
        )

        assert result == tokenizer_path

    @pytest.mark.fast
    def test_real_fallback_tokenizer_layout_resolves(self, tmp_path):
        model_root = tmp_path / "model"
        fallback_path = model_root / "text_encoder"
        _write_real_tokenizer(fallback_path)

        result = TokenizerLoader._resolve_path(
            model_path=str(model_root),
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=None,
            tokenizer_class="AutoTokenizer",
        )

        assert result == fallback_path

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_load_qwen2_tokenizer_workaround", return_value=object())
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_hf_root_qwen2_tokenizer_layout_resolves(self, mock_download, _mock_qwen2_load, tmp_path):
        cached_root = tmp_path / "cached"
        _write_minimal_qwen2_tokenizer_files(cached_root)
        mock_download.return_value = str(cached_root)

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir=".",
            fallback_subdirs=["tokenizer", "text_encoder"],
            download_patterns=["vocab.json", "merges.txt", "tokenizer.json", "tokenizer_config.json"],
            tokenizer_class="Qwen2Tokenizer",
        )

        assert result == cached_root
        assert mock_download.call_count == 1
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True

    @pytest.mark.fast
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_hf_refresh_can_use_real_fallback_tokenizer_layout(self, mock_download, tmp_path):
        partial_root = tmp_path / "partial"
        refreshed_root = tmp_path / "refreshed"
        partial_root.mkdir()
        _write_real_tokenizer(refreshed_root / "text_encoder")
        mock_download.side_effect = [str(partial_root), str(refreshed_root)]

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=["tokenizer/**", "text_encoder/**"],
            tokenizer_class="AutoTokenizer",
        )

        assert result == refreshed_root / "text_encoder"
        assert mock_download.call_count == 2
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_probe_tokenizer_path")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_full_cache_uses_primary_without_redownload(self, mock_download, mock_probe_path, tmp_path):
        cached_root = tmp_path / "cached"
        (cached_root / "tokenizer").mkdir(parents=True)
        mock_download.return_value = str(cached_root)
        mock_probe_path.side_effect = lambda path, tokenizer_class: (path == cached_root / "tokenizer", None)

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
    @patch.object(TokenizerLoader, "_probe_tokenizer_path")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_partial_cache_redownloads_missing_primary_tokenizer(
        self,
        mock_download,
        mock_probe_path,
        tmp_path,
    ):
        partial_root = tmp_path / "partial"
        full_root = tmp_path / "full"
        (partial_root / "text_encoder").mkdir(parents=True)
        (full_root / "tokenizer").mkdir(parents=True)
        mock_download.side_effect = [str(partial_root), str(full_root)]
        mock_probe_path.side_effect = lambda path, tokenizer_class: (path == full_root / "tokenizer", None)

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
    @patch.object(TokenizerLoader, "_probe_tokenizer_path")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_no_cache_downloads_tokenizer(self, mock_download, mock_probe_path, tmp_path):
        downloaded_root = tmp_path / "downloaded"
        (downloaded_root / "tokenizer").mkdir(parents=True)
        mock_download.side_effect = [LocalEntryNotFoundError("no cache"), str(downloaded_root)]
        mock_probe_path.side_effect = lambda path, tokenizer_class: (path == downloaded_root / "tokenizer", None)

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
    @patch.object(TokenizerLoader, "_load_raw_tokenizer")
    def test_existing_primary_dir_without_tokenizer_artifacts_uses_fallback(self, mock_load_raw, tmp_path):
        model_root = tmp_path / "model"
        model_root.mkdir()
        fallback_path = model_root / "tokenizer"
        _touch(fallback_path / "tokenizer.json")

        def _mock_load(*, tokenizer_path, tokenizer_class, chat_template=None):
            if tokenizer_path == fallback_path:
                return object()
            raise RuntimeError("not a tokenizer layout")

        mock_load_raw.side_effect = _mock_load

        result = TokenizerLoader._resolve_path(
            model_path=str(model_root),
            hf_subdir=".",
            fallback_subdirs=["tokenizer"],
            download_patterns=None,
            tokenizer_class="AutoTokenizer",
        )

        assert result == fallback_path

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
    @patch.object(TokenizerLoader, "_load_raw_tokenizer", side_effect=RuntimeError("bad fallback tokenizer"))
    def test_unloadable_fallback_tokenizer_is_reported_as_missing_layout(self, mock_load_raw, tmp_path):
        model_root = tmp_path / "model"
        model_root.mkdir()
        _touch(model_root / "text_encoder" / "tokenizer.json")

        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path=str(model_root),
                hf_subdir="tokenizer",
                fallback_subdirs=["text_encoder"],
                download_patterns=None,
                tokenizer_class="AutoTokenizer",
            )

        assert "No usable tokenizer files" in str(exc_info.value)
        assert "text_encoder" in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        mock_load_raw.assert_called_once_with(
            tokenizer_path=model_root / "text_encoder",
            tokenizer_class="AutoTokenizer",
        )

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_is_tokenizer_loadable")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_offline_partial_cache_raises_targeted_error(self, mock_download, mock_is_loadable, tmp_path):
        partial_root = tmp_path / "partial"
        (partial_root / "text_encoder").mkdir(parents=True)
        mock_download.side_effect = [str(partial_root), OSError("offline")]
        mock_is_loadable.return_value = False

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
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_refresh_without_any_usable_tokenizer_raises_clear_hf_error(self, mock_download, tmp_path):
        partial_root = tmp_path / "partial"
        refreshed_root = tmp_path / "refreshed"
        partial_root.mkdir()
        refreshed_root.mkdir()
        mock_download.side_effect = [str(partial_root), str(refreshed_root)]

        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path="org/model",
                hf_subdir="tokenizer",
                fallback_subdirs=["text_encoder", "."],
                download_patterns=["tokenizer/**", "text_encoder/**"],
                tokenizer_class="AutoTokenizer",
            )

        assert "No usable tokenizer files were found for Hugging Face repo" in str(exc_info.value)
        assert "Incomplete Hugging Face tokenizer cache" not in str(exc_info.value)
        assert exc_info.value.__cause__ is None
        assert mock_download.call_count == 2
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True
        assert "local_files_only" not in mock_download.call_args_list[1].kwargs

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_is_tokenizer_loadable")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_offline_cached_fallback_layout_still_uses_fallback(self, mock_download, mock_is_loadable, tmp_path):
        cached_root = tmp_path / "cached"
        (cached_root / "text_encoder").mkdir(parents=True)
        mock_download.return_value = str(cached_root)
        mock_is_loadable.side_effect = lambda path, tokenizer_class: path == cached_root / "text_encoder"

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=["tokenizer/**", "text_encoder/**"],
            tokenizer_class="AutoTokenizer",
        )

        assert result == cached_root / "text_encoder"
        assert mock_download.call_count == 1
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True

    @pytest.mark.fast
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_offline_partial_primary_artifacts_still_use_cached_fallback(self, mock_download, tmp_path):
        cached_root = tmp_path / "cached"
        _touch(cached_root / "tokenizer" / "tokenizer_config.json")
        _write_real_tokenizer(cached_root / "text_encoder")
        mock_download.side_effect = [str(cached_root), OSError("offline")]

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=["text_encoder", "."],
            download_patterns=["tokenizer/**", "text_encoder/**"],
            tokenizer_class="AutoTokenizer",
        )

        assert result == cached_root / "text_encoder"
        assert mock_download.call_count == 1
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_load_raw_tokenizer", side_effect=RuntimeError("bad tokenizer"))
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_offline_cached_primary_load_error_raises_targeted_error(self, mock_download, _mock_load_raw, tmp_path):
        cached_root = tmp_path / "cached"
        _touch(cached_root / "tokenizer" / "tokenizer.json")
        mock_download.side_effect = [str(cached_root), OSError("offline")]

        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path="org/model",
                hf_subdir="tokenizer",
                fallback_subdirs=None,
                download_patterns=["tokenizer/**"],
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

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_load_raw_tokenizer", side_effect=RuntimeError("bad tokenizer"))
    def test_existing_primary_load_error_is_preserved(self, _mock_load_raw, tmp_path):
        model_root = tmp_path / "model"
        _touch(model_root / "tokenizer" / "tokenizer.json")

        with pytest.raises(RuntimeError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path=str(model_root),
                hf_subdir="tokenizer",
                fallback_subdirs=None,
                download_patterns=None,
                tokenizer_class="AutoTokenizer",
            )

        assert "failed to load them" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_load_raw_tokenizer")
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_cached_hf_primary_load_error_redownloads_once_and_recovers(self, mock_download, mock_load_raw, tmp_path):
        cached_root = tmp_path / "cached"
        fresh_root = tmp_path / "fresh"
        _touch(cached_root / "tokenizer" / "tokenizer.json")
        _touch(fresh_root / "tokenizer" / "tokenizer.json")
        mock_download.side_effect = [str(cached_root), str(fresh_root)]

        def _mock_load(*, tokenizer_path, tokenizer_class, chat_template=None):
            if tokenizer_path == cached_root / "tokenizer":
                raise RuntimeError("bad cached tokenizer")
            return object()

        mock_load_raw.side_effect = _mock_load

        result = TokenizerLoader._resolve_path(
            model_path="org/model",
            hf_subdir="tokenizer",
            fallback_subdirs=None,
            download_patterns=["tokenizer/**"],
            tokenizer_class="AutoTokenizer",
        )

        assert result == fresh_root / "tokenizer"
        assert mock_download.call_count == 2
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True
        assert "local_files_only" not in mock_download.call_args_list[1].kwargs

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_load_raw_tokenizer", side_effect=RuntimeError("bad tokenizer"))
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_cached_hf_primary_load_error_is_preserved_after_refresh(self, mock_download, _mock_load_raw, tmp_path):
        cached_root = tmp_path / "cached"
        fresh_root = tmp_path / "fresh"
        _touch(cached_root / "tokenizer" / "tokenizer.json")
        _touch(fresh_root / "tokenizer" / "tokenizer.json")
        mock_download.side_effect = [str(cached_root), str(fresh_root)]

        with pytest.raises(RuntimeError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path="org/model",
                hf_subdir="tokenizer",
                fallback_subdirs=None,
                download_patterns=["tokenizer/**"],
                tokenizer_class="AutoTokenizer",
            )

        assert "failed to load them" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        assert mock_download.call_count == 2
        assert mock_download.call_args_list[0].kwargs["local_files_only"] is True
        assert "local_files_only" not in mock_download.call_args_list[1].kwargs

    @pytest.mark.fast
    def test_missing_tilde_path_raises_model_not_found(self):
        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path="~/definitely-not-a-real-tokenizer-path",
                hf_subdir="tokenizer",
                fallback_subdirs=None,
                download_patterns=None,
                tokenizer_class="AutoTokenizer",
            )

        assert "Model not found" in str(exc_info.value)

    @pytest.mark.fast
    @patch("mflux.models.common.tokenizer.tokenizer_loader.snapshot_download")
    def test_missing_relative_path_is_not_treated_as_hf_repo(self, mock_download):
        with pytest.raises(FileNotFoundError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path="./org",
                hf_subdir="tokenizer",
                fallback_subdirs=None,
                download_patterns=None,
                tokenizer_class="AutoTokenizer",
            )

        assert "Model not found" in str(exc_info.value)
        mock_download.assert_not_called()

    @pytest.mark.fast
    @patch.object(TokenizerLoader, "_is_tokenizer_loadable", return_value=True)
    @patch.object(TokenizerLoader, "_probe_tokenizer_path", return_value=(False, RuntimeError("bad tokenizer")))
    def test_broken_primary_does_not_fall_back(self, mock_probe_path, mock_is_loadable, tmp_path):
        model_root = tmp_path / "model"
        _touch(model_root / "tokenizer" / "tokenizer.json")
        (model_root / "text_encoder").mkdir(parents=True)

        with pytest.raises(RuntimeError) as exc_info:
            TokenizerLoader._resolve_path(
                model_path=str(model_root),
                hf_subdir="tokenizer",
                fallback_subdirs=["text_encoder"],
                download_patterns=None,
                tokenizer_class="AutoTokenizer",
            )

        assert "failed to load them" in str(exc_info.value)
        assert isinstance(exc_info.value.__cause__, RuntimeError)
        mock_probe_path.assert_called_once_with(model_root / "tokenizer", "AutoTokenizer")
        mock_is_loadable.assert_not_called()
