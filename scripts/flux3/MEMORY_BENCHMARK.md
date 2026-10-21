# FLUX 3 memory measurements

Use `benchmark_memory.py` to compare the existing default and `--low-ram` CLI modes in fresh processes. The wrapper records phase timings and memory without changing the production loading policy.

## Reproduce

Use a saved local checkpoint and a unique report and image path for each run:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /usr/bin/time -l uv run python scripts/flux3/benchmark_memory.py \
  --report /tmp/flux3-lowram-1.json -- \
  --model /path/to/saved-flux3-klein-q4 --base-model flux3-image-klein \
  --low-ram --width 512 --height 512 --steps 16 --seed 42 \
  --prompt 'A small red ceramic robot standing beside a green fern on a wooden desk, soft morning window light, detailed handmade textures, product photograph.' \
  --output /tmp/flux3-lowram-1.png --make-conf
```

Run in this order: low-RAM, default, default, low-RAM. Remove `--low-ram` for default runs. Use a fresh process and unique filenames each time. Keep checkpoint, precision, prompt, dimensions, steps and seed fixed. The local checkpoint must include the tokenizer files. For a remote model, remove the offline variables and download it before measuring.

The current CLI releases the text encoder after encoding in both modes. For a single seed, default mode retains the transformer for decoding. Low-RAM mode releases it and sets a 1 GB allocator cache limit. Low-RAM also enables applicable VAE tiling defaults. This comparison measures those existing behaviours together.

## What the report measures

- `construct` includes model construction. MLX loads lazily, so later encoding and denoising also include weight reads and materialisation.
- `encode` and `decode` materialise their results. All phase boundaries synchronise Metal. This instrumentation changes timing; compare runs with the same wrapper.
- `before_loop` and `after_loop` measure the existing memory callbacks. The interval from `before_loop_end` to `after_loop_start` estimates denoising, including loop setup and callbacks.
- `mlx_lifetime_peak_bytes` retains the maximum peak across reset segments, including the peak before low-RAM resets the counter. It measures MLX allocation, not total system memory.
- `process_peak_rss_bytes` is the separate macOS lifetime RSS counter, in bytes. Do not add it to MLX allocation or treat it as a replacement for the Metal memory metric.
- `python_wall_s` excludes interpreter and uv startup. `/usr/bin/time -l` supplies external process wall time.
- System swap and memory pressure include every process. They do not isolate the benchmark. Filesystem cache is not flushed; repeated processes are not proven cold-cache runs.
- `versions` comes from installed distribution metadata. `source_path` identifies the imported generation CLI, and `source_sha` records its checkout revision (null for an installed package outside Git). `benchmark_source_sha` separately records the wrapper's checkout revision. The installed distribution can be older than the imported source selected with `PYTHONPATH`.

Reports include local paths and prompts. Review them before sharing. The wrapper restores its instrumentation on exit, writes a failure report for CLI errors, and refuses to replace an existing report. Success requires `saved_images` to equal `expected_images`, the number of seeds resolved by the CLI (including automatic seeds and metadata). An interrupted run fails even if earlier images were saved.

## Follow-up scope

Start with these baselines before proposing additional staged loading. Test BF16 and larger images separately, under recorded memory pressure. A loading change should preserve outputs for fixed inputs, reduce the relevant memory peak, and disclose extra I/O or latency. Repeated prompts and seeds, changed prompts, Base guidance and reference-image editing need their own validation before a general residency policy is proposed.

## M1 Max 32 GB results, 20 October 2026

Production source and wrapper: this branch at `a295c51ae502c7078cfab556110c53176a22890b` (main at `ff74500` plus this PR), MFLUX 0.23.0, macOS 14.6, MLX 0.32.0. Each report records both revisions as `source_sha` and `benchmark_source_sha`.

The checkpoint is a 4-bit conversion of [`black-forest-labs/flux-3-image-klein`](https://huggingface.co/black-forest-labs/flux-3-image-klein) at revision `1755314e2216b2a2c5cc0dd899d4e49ee3404d2c`. All five BF16 weight files matched the published hashes first. It was saved from the same revision:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run mflux-save --model /path/to/flux-3-image-klein \
  --base-model flux3-image-klein --quantize 4 --path /path/to/saved-flux3-klein-q4
```

A second conversion produced byte-identical files, so the hashes in [`run.json`](benchmarks/m1-max-32gb-q4-20261020/run.json) can be checked by repeating it.

The [four reports](benchmarks/m1-max-32gb-q4-20261020/) are unedited script output except for local paths, which are replaced with `<checkout>`, `<checkpoint>` and `<output>`. `run.json` adds what the script does not record: the conversion, the checkpoint hashes, the `/usr/bin/time` external wall times and the image pixel hashes.

| Run order | External wall, s | Python wall, s | Denoise interval, s | Peak MLX, GB | Peak RSS, GB |
| --- | ---: | ---: | ---: | ---: | ---: |
| lowram-1 | 52.63 | 51.89 | 42.44 | 3.997 | 3.352 |
| default-1 | 49.45 | 48.05 | 39.81 | 6.710 | 3.447 |
| default-2 | 67.35 | 65.87 | 53.94 | 6.710 | 3.450 |
| lowram-2 | 88.57 | 87.72 | 77.15 | 3.997 | 3.365 |

GB in the table means decimal gigabytes. All four 512×512 images have identical RGB pixels. The low-RAM runs use about 40% less peak MLX allocation than the default runs for this case.

Other applications were active. System swap grew from 2.0 GB to 4.4 GB across the sequence, and memory pressure reached the critical level after `default-1`; each report records swap and pressure before and after. These timings are observations under shared load, not a speed ranking. The q4 case demonstrates that the existing low-RAM path reduces allocation. It does not establish a need for another loading policy.

The recorder tests pass, including interrupted multi-seed runs and imports from a separate checkout or an installed package. Three initial counterexamples (losing a pre-reset peak, skipping lazy evaluation, and leaking instrumentation) each failed a relevant test. Restoring the two reporting bugs also fails the new regression tests; the restored recorder passes again.
