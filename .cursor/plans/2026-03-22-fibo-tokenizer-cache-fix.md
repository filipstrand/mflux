## Goal
- Fix the FIBO tokenizer load failure when a Hugging Face repo is only partially cached locally.

## Root Cause
- `fd07e82` fixed a real offline usability problem: cached Hugging Face tokenizers should load without requiring a network download.
- `TokenizerLoader._resolve_path()` first tries `snapshot_download(..., local_files_only=True)`.
- If the cache already contains some matching files under `text_encoder/**` but not `tokenizer/**`, Hugging Face returns the cached snapshot path instead of raising.
- The current resolver then sees that `tokenizer/` is missing, falls back to `text_encoder/`, and `AutoTokenizer.from_pretrained()` fails with the sentencepiece/tiktoken-style error.
- So the regression is not "offline loading is wrong"; it is that the resolver currently treats a partial cache as a complete tokenizer cache.

## Planned Changes
- Keep the local-first behavior from `fd07e82` so fully cached Hugging Face tokenizers still work offline.
- Update `src/mflux/models/common/tokenizer/tokenizer_loader.py` so `snapshot_download(..., local_files_only=True)` is treated as a candidate snapshot root, not proof that the requested tokenizer is complete.
- Add a tokenizer completeness check for the primary tokenizer path so offline loading succeeds only when the expected tokenizer artifacts are actually present.
- If the primary tokenizer is incomplete for an HF repo, fetch the tokenizer files with a normal download and retry the completeness check.
- Preserve fallback behavior only for genuine alternate layouts: if the primary tokenizer is still absent after retrying, and a fallback tokenizer path is complete, use the fallback.
- Do not use fallback subdirs as a rescue path for incomplete HF tokenizer caches; fail with a clear error if neither the primary nor fallback path is complete.
- Add a regression test covering the partial-cache case.

## Resolution Algorithm
```text
candidate_root = resolve_hf_or_local_root(...)
primary_path = candidate_root / hf_subdir

if primary_path is complete:
    use primary_path
elif model_path is an HF repo:
    fetch tokenizer files with normal download
    candidate_root = refreshed snapshot root
    primary_path = candidate_root / hf_subdir

    if primary_path is complete:
        use primary_path
    elif primary_path is absent and a fallback path is complete:
        use fallback path
    else:
        fail with a clear incomplete-tokenizer error
elif a fallback path is complete:
    use fallback path
else:
    fail with a clear incomplete-tokenizer error
```

## Files
- `src/mflux/models/common/tokenizer/tokenizer_loader.py`
- A tokenizer-focused test file covering partial-cache resolution behavior

## Verification
- Run the new/updated fast regression test.
- Run lints on touched files and fix any issues introduced by the change.

## Research Links
- Hugging Face `snapshot_download` docs: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download
- Hugging Face `hf_hub_download` docs: https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.hf_hub_download
- Hugging Face download guide: https://huggingface.co/docs/huggingface_hub/main/en/guides/download
- `snapshot_download` source (shows that returning a cached snapshot folder does not prove all requested files are present): https://raw.githubusercontent.com/huggingface/huggingface_hub/main/src/huggingface_hub/_snapshot_download.py
- Related Hugging Face cache/local-state discussion: https://github.com/huggingface/huggingface_hub/issues/2607
