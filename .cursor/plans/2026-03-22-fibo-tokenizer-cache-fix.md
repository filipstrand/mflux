## Goal
- Fix the FIBO tokenizer load failure when a Hugging Face repo is only partially cached locally.

## Root Cause
- `fd07e82` fixed a real offline usability problem: cached Hugging Face tokenizers should load without requiring a network download.
- `TokenizerLoader._resolve_path()` first tries `snapshot_download(..., local_files_only=True)`.
- If the cache already contains some matching files under `text_encoder/**` but not `tokenizer/**`, Hugging Face returns the cached snapshot path instead of raising.
- The current resolver then sees that `tokenizer/` is missing, falls back to `text_encoder/`, and `AutoTokenizer.from_pretrained()` fails with the sentencepiece/tiktoken-style error.
- So the regression is not "offline loading is wrong"; it is that the resolver currently treats a partial cache as a complete tokenizer cache.
- The first implementation pass fixed the partial-cache redownload path, but it still used a filename heuristic for "usable tokenizer" and would reject valid layouts such as `tokenizer.model`.
- That pass also rewrote fresh Hugging Face download failures (for example missing/private repos or first-download network failures) into the same "incomplete cache" error, which is misleading when no local cache existed yet.

## Planned Changes
- Keep the local-first behavior from `fd07e82` so fully cached Hugging Face tokenizers still work offline.
- Update `src/mflux/models/common/tokenizer/tokenizer_loader.py` so `snapshot_download(..., local_files_only=True)` is treated as a candidate snapshot root, not proof that the requested tokenizer is complete.
- Replace the filename-based tokenizer heuristic with an actual tokenizer loadability probe that uses the configured tokenizer class, so valid layouts such as `tokenizer.model` still resolve correctly.
- If the primary tokenizer is unusable for an HF repo, fetch the tokenizer files with a normal download and retry the usability check.
- Preserve fallback behavior for genuine alternate layouts when the primary location is missing or clearly not a tokenizer layout. If the primary location contains tokenizer artifacts but still fails to load, surface that real load error instead of silently falling back.
- Do not use fallback subdirs as a rescue path for incomplete HF tokenizer caches when those fallback paths are not themselves loadable.
- Distinguish between "cached snapshot was incomplete" and "initial Hugging Face download failed" so user-facing errors match the actual failure mode.
- Preserve real primary-tokenizer load failures as load errors instead of rewriting them as "missing tokenizer files".
- Add regression tests covering the partial-cache case, valid `tokenizer.model` layouts, and fresh-download failures.

## Resolution Algorithm
```text
candidate_root = resolve_hf_or_local_root(...)
primary_path = candidate_root / hf_subdir

if primary_path is loadable:
    use primary_path
elif primary_path exists and has tokenizer artifacts:
    fail with the real tokenizer load error
elif model_path is an HF repo:
    fetch tokenizer files with normal download
    candidate_root = refreshed snapshot root
    primary_path = candidate_root / hf_subdir

    if primary_path is loadable:
        use primary_path
    elif primary_path exists and has tokenizer artifacts:
        fail with the real tokenizer load error
    elif a fallback path is loadable:
        use fallback path
    elif refresh failed after an unusable cached snapshot:
        fail with a clear incomplete-tokenizer cache error
    else:
        fail with a clear Hugging Face tokenizer resolution error
elif a fallback path is loadable:
    use fallback path
else:
    fail with a clear incomplete-tokenizer cache error
```

## Files
- `src/mflux/models/common/tokenizer/tokenizer_loader.py`
- A tokenizer-focused test file covering partial-cache resolution behavior

## Verification
- Run the new/updated fast regression test.
- Run lints on touched files and fix any issues introduced by the change.
- Confirm the new tests cover the partial-cache retry path, a valid offline fallback layout, the "primary exists but is the wrong layout" fallback path, and the non-cache Hugging Face download failure path.

## Research Links
- Hugging Face `snapshot_download` docs: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download
- Hugging Face `hf_hub_download` docs: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.hf_hub_download
- Hugging Face download guide: https://huggingface.co/docs/huggingface_hub/main/en/guides/download
- `snapshot_download` source (shows that returning a cached snapshot folder does not prove all requested files are present): https://raw.githubusercontent.com/huggingface/huggingface_hub/main/src/huggingface_hub/_snapshot_download.py
- Related Hugging Face cache/local-state discussion: https://github.com/huggingface/huggingface_hub/issues/2607
