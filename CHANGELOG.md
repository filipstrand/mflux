# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.4] - 2026-01-20

### ✨ Improvements

- **Flux2 LoRA aliasing**: Add key aliases for `base_model` prefixes to improve LoRA resolution across configs.

### 📝 Documentation

- **Agent guidance**: Clarify skill references for Cursor agents.

---

## [0.15.3] - 2026-01-19

### 🐛 Bug Fixes

- **Flux2 Klein local path**: Fix errors when using a local FLUX.2-klein-9B path in `mflux-save` and `mflux-generate-flux2`.

---

## [0.15.2] - 2026-01-19

### 🐛 Bug Fixes

- **Flux2 edit (low-ram)**: Normalize tiled VAE latents to 4D before patchifying to avoid shape errors.

---

## [0.15.1] - 2026-01-18

### 🐛 Bug Fixes

- **PyPI metadata**: Removed invalid architecture classifier that blocked uploads (`Architecture :: AArch64`).

---

## [0.15.0] - 2026-01-18

### 🎨 New Model Support

- **Flux2 Klein (4B/9B)**: Full MLX port of Flux2 Klein (including multi-image editing support).
- **New command**: `mflux-generate-flux2` for Flux2 Klein image generation.
- **New command**: `mflux-generate-flux2-edit` for Flux2 Klein image editing.

### 🔧 Improvements

- **Qwen3-VL shared module**: Extracted `qwen3_vl` into `models/common_models/` for reuse across model families (Flux2 and Fibo etc).
- **Experimental CUDA support**: Added initial CUDA backend support as an experimental feature.
- **Test Infrastructure**: Image tests are pinned to MLX v0.30.3.

### 📝 Documentation

- **README reorganization**: Reorganized the main README for better structure and readability.

---

## [0.14.2] - 2026-01-13

### 📊 Improved Metadata Handling

- **Enhanced IPTC & XMP Support**: Significant improvements to metadata reading and writing, ensuring better compatibility with professional image editing tools.
- **Robust Metadata Extraction**: Refined logic for extracting generation parameters from previously generated images.
- **New Metadata Tests**: Added comprehensive test suite for IPTC metadata building and original image info utilities.

### 🤖 DX & Maintenance

- **Cursor AI Workflows**: Introduced standardized Cursor commands and agent rules in `.cursor/` for improved development consistency and automation.
- **SeedVR2 & ControlNet Tweaks**: Minor refinements to SeedVR2 and ControlNet model implementations.
- **Documentation Updates**: Updated README and added AGENTS.md for better contributor onboarding.

---

## [0.14.1] - 2026-01-01

### 🔧 SeedVR2 Improvements

- **Enhanced Color Correction**: Implemented precise LAB histogram matching with wavelet reconstruction for superior color consistency between input and upscaled images.
- **Configurable Softness**: Added a new `--softness` parameter (0.0 to 1.0) to control input pre-downsampling, allowing for smoother upscaling results when desired.
- **RoPE Alignment**: Fixed RoPE dimension mismatch (increased to 128) to perfectly match the reference 3B transformer architecture.

### 🤖 DX & Maintenance

- **Updated `.cursorrules`**: Added standard procedure for test output preservation and release management.
- **Updated Test Infrastructure**: Updated SeedVR2 reference images and fixed dimension-related test failures.

---

## [0.14.0] - 2025-12-31

### 🎨 New Model Support

- **SeedVR2 Diffusion Upscaler**: Added support for [SeedVR2](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler), a powerful diffusion-based image upscaler.
- **New command**: `mflux-upscale-seedvr2` for high-quality image upscaling.
- **Tiling support**: Tiling is enabled by default for SeedVR2 to support high-resolution upscaling on standard memory configurations.

### 🔧 Improvements

- **Global VAE Tiling Support**: Introduced a unified VAE tiling system (`VAETiler`) that supports both tiled encoding and decoding.
- **Low-RAM Mode Enhancements**: Enabling `--low-ram` now automatically activates VAE tiling across all model families (Flux, Qwen, FIBO, Z-Image), significantly reducing memory pressure for high-resolution generation on Apple Silicon.
- **Robust Offline Cache Handling**: Improved logic for detecting complete cached models on HuggingFace Hub, handling symlinks and missing files more reliably to prevent runtime errors during offline use.
- **Selective Weight Loading**: Support for loading specific weight files, enabling more flexible model configurations and better resource sharing between related models.
- **CLI UX Improvements**:
  - Multi-image generation (multiple seeds or input images) now automatically appends suffixes (`_seed_{seed}` or `_{image_name}`) to output filenames to prevent accidental overwrites.
  - Better model configuration resolution with a priority-based system for resolving ambiguous model names.
- **Enhanced Shell Completions**: Significant updates to shell completion generation to support new commands and properly handle positional arguments and subparsers.
- **Qwen Test Hardening**: Updated Qwen image generation and edit tests to use 8-bit quantization for more robust and faster testing.
- **Test Infrastructure**: Added automatic MLX version pinning (v0.29.2) in `make test-fast` to ensure consistent test environments across different development setups.

### 📝 Documentation

- Added information about pre-quantized models available on HuggingFace for easier access.

---

## [0.13.3] - 2025-12-06

### 🐛 Bug Fixes

- **LoRA save bloat prevention**: Bake and strip LoRA wrappers before sharding to avoid exploding shard counts/sizes when saving quantized models with multiple/mismatched LoRAs (see [issue #217 comment](https://github.com/filipstrand/mflux/issues/217#issuecomment-3615321206)).
- **Regression test hardening**: LoRA model-saving tests now include size guardrails (5% tolerance) while using the bundled local LoRA fixtures to catch shard bloat regressions early.

---

## [0.13.2] - 2025-12-05

### ✨ Improvements

- **Better error messages for multi-file LoRA repos**: When a HuggingFace LoRA repo contains multiple `.safetensors` files, the error message now displays copy-paste ready options instead of a raw list
- **Z-Image LoRA format support**: Added support for Kohya and ComfyUI LoRA naming conventions, enabling compatibility with more community LoRAs.

---

## [0.13.1] - 2025-12-03

### 🐛 Bug Fixes

- **FIBO VLM chat template not loaded**: Fixed issue where the FIBO VLM tokenizer's chat template was not being loaded with `transformers` v5, causing `apply_chat_template()` to fail. The tokenizer loader now properly extracts and sets the chat template from the tokenizer config.

---

## [0.13.0] - 2025-12-03

# MFLUX v.0.13.0 Release Notes

### 🎨 New Model Support

- **Z-Image Turbo Support**: Added support for [Z-Image Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), a fast distilled Z-Image variant optimized for speed
- **New command**: `mflux-generate-z-image-turbo` for rapid image generation (with LoRA support, img2img, and quantization)

### ✨ New Features

- **FIBO VLM Quantization Support**: The FIBO VLM commands (`mflux-fibo-inspire`, `mflux-fibo-refine`) now support quantization via the `-q` flag (3, 4, 5, 6, or 8-bit)

- **Unified `--model` argument**: The `--model` flag now accepts local paths, HuggingFace repos, or predefined model names
  - Local paths: `--model /Users/me/models/fibo-4bit` or `--model ~/my-model`
  - HuggingFace repos: `--model briaai/Fibo-mlx-4bit`
  - Predefined names: `--model dev`, `--model schnell`, `--model fibo`
  - This mirrors how LoRA paths work for a consistent UX

- **Scale Factor Dimensions for Img2Img**: Generalized the scale factor feature (e.g., `2x`, `0.5x`, `auto`) from upscaling to all img2img commands
  - Specify output dimensions relative to input image: `--width 2x --height 2x`
  - Use `auto` to match input image dimensions: `--width auto --height auto`
  - Mix scale factors with absolute values: `--width 2x --height 512`
  - Supported in: `mflux-generate`, `mflux-generate-z-image-turbo`, `mflux-generate-fibo`, `mflux-generate-kontext`, `mflux-generate-qwen`
- **DimensionResolver utility**: New `DimensionResolver.resolve()` for consistent dimension handling across commands

### 🔧 Architecture Improvements

- **Unified Resolution System**: New `resolution/` module for consistent parameter resolution across all models
  - `PathResolution`: Resolves model paths from local paths, HuggingFace repos, or predefined names
  - `LoRAResolution`: Handles LoRA path resolution from all supported formats
  - `ConfigResolution`: Centralizes configuration resolution logic  
  - `QuantizationResolution`: Determines quantization from saved models or CLI args
- **Unified Weight Loading System**: Complete rewrite of weight handling with declarative mappings
  - New `WeightLoader` with single `load(model_path)` interface
  - `WeightDefinition` classes define model structure per model family
  - `WeightMapping` declarative mappings replace imperative weight handlers
  - Removed all per-model `weight_handler_*.py` files in favor of unified system
- **Unified Tokenizer System**: New common tokenizer module
  - `TokenizerLoader.load_all()` with unified `model_path` interface
  - Removed model-specific tokenizer handlers (`clip_tokenizer.py`, `t5_tokenizer.py`, etc.)
- **Unified LoRA API**: Simplified LoRA loading to a single `lora_paths` parameter
  - All LoRA formats now resolved through `LoRALibrary.resolve_paths()`:
    - Local paths: `/path/to/lora.safetensors`
    - Registry names: `my-lora` (from `LORA_LIBRARY_PATH`)
    - HuggingFace repos: `author/model`
    - **New**: HuggingFace collections: `repo_id:filename.safetensors`
  - Simplified model initialization: just pass `lora_paths` and everything resolves automatically
- **Unified Latent Creator Interface**: Standardized `unpack_latents(latents, height, width)` signature across all model families
  - `FluxLatentCreator`, `ZImageLatentCreator`, `FiboLatentCreator`, and `QwenLatentCreator` now share the same interface
  - Moved `FIBO._unpack_latents` to `FiboLatentCreator.unpack_latents` for consistency
- **StepwiseHandler Refactor**: Fixed `StepwiseHandler` to work with all model types by accepting a `latent_creator` parameter
  - Previously hardcoded to `FluxLatentCreator`, now model-agnostic
  - Each command passes its appropriate latent creator to `CallbackManager.register_callbacks()`
- **CLI Reorganization**: Moved CLI entry points to model-specific directories (e.g., `mflux/models/flux/cli/`)

### 🔄 Breaking Changes

- **Simplified `generate_image()` API** (programmatic users only):
  - Removed `Config` class - parameters are now passed directly to `generate_image()`
  - Removed `RuntimeConfig` class - internal complexity eliminated
  - Added `Flux1` export to main `mflux` module for cleaner imports
- **LoRA API simplified** (programmatic users only):
  - Removed `lora_names` and `lora_repo_id` parameters from all model classes (`Flux1`, `QwenImage`, `QwenImageEdit`, etc.)
  - Removed `--lora-name` and `--lora-repo-id` CLI arguments
  - Removed `LoRAHuggingFaceDownloader` class

### 🔄 Breaking Changes (CLI)

- **`--path` flag removed**: The deprecated `--path` flag for loading models has been removed. Use `--model` instead for local paths, HuggingFace repos, or predefined model names.

### 📦 Dependency Updates

- **Updated `huggingface-hub`** from `>=0.24.5,<1.0` to `>=1.1.6,<2.0`
  - v1.1.6 includes fix for incomplete file listing in `snapshot_download` which could cause cache corruption
  - Removed explicit `accelerate` and `filelock` dependencies (pulled in as transitive dependencies)
- **Updated `transformers`** from `>=4.57,<5.0` to `>=5.0.0rc0,<6.0`
  - Required for `huggingface-hub` 1.x compatibility
  - Added workaround for `Qwen2Tokenizer` bug in transformers 5.0.0rc0 where vocab/merges files are not loaded correctly via `from_pretrained()`

### 🐛 Bug Fixes

- **Qwen empty negative prompt crash**: Fixed crash when running Qwen models without a `--negative-prompt` argument. Empty prompts now use a space as fallback to ensure valid tokenization.

- **`--model` flag not working**: Fixed bug where the `--model` argument wasn't being used for loading models from HuggingFace or local paths. All CLI commands now correctly use `--model` for model path resolution.
- **Model Saving Index File**: Fixed issue where locally saved models (via `mflux-save`) would fail to load when uploaded to HuggingFace, due to missing `model.safetensors.index.json`. The model saver now generates this index file alongside the safetensor shards, ensuring compatibility with both mflux and standard HuggingFace loading paths. (see [#285](https://github.com/filipstrand/mflux/issues/285))

### 🧪 Test Infrastructure

- **Test markers**: Added `fast` and `slow` pytest markers to categorize tests
  - Fast tests: Unit tests that don't generate images (parsers, schedulers, resolution, utilities)
  - Slow tests: Integration tests that generate actual images and compare to references
- **New Makefile targets**:
  - `make test-fast` - Run fast tests only (quick feedback during development)
  - `make test-slow` - Run slow tests only (image generation tests)
  - `make test` - Run all tests (unchanged)
- Run specific test categories: `pytest -m fast` or `pytest -m slow`
- **GitHub Actions CI**: Fast tests now run automatically on PRs and pushes to main

### 🔧 Internal Changes

- Simplified `WeightLoader.load()` to take a single `model_path` parameter instead of separate `repo_id` and `local_path`
- Simplified `TokenizerLoader.load_all()` with the same unified `model_path` interface
- Renamed `local_path` parameter to `model_path` in all model constructors for clarity
- Removed `quantization_util.py` - quantization now handled through `QuantizationResolution`
- Removed `lora_huggingface_downloader.py` - downloading integrated into `LoRAResolution`
- Added comprehensive test coverage for resolution modules

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: Z-Image Turbo support, architecture improvements, core development

---

## [0.12.1] - 2025-11-27

### 🐛 Bug Fixes

- **FIBO VLM Tokenizer Download**: Fixed an issue where the FIBO VLM tokenizer files would not download automatically when the model weights were cached but tokenizer files were missing. The initializer now properly checks for tokenizer file existence and downloads them if needed.

---

## [0.12.0] - 2025-11-27

# MFLUX v.0.12.0 Release Notes

### 🎨 New Model Support

- **Bria FIBO Support**: Added support for [FIBO](https://huggingface.co/briaai/FIBO), the first open-source JSON-native text-to-image model from [Bria.ai](https://bria.ai)
- **Three operation modes**: Generate (text-to-image with VLM expansion), Refine (structured prompt editing), and Inspire (image-to-prompt extraction)
- **New commands**:
  - `mflux-generate-fibo` - Generate images from text prompts with VLM-guided JSON expansion
  - `mflux-refine-fibo` - Refine images using structured JSON prompts for targeted attribute editing
  - `mflux-inspire-fibo` - Extract structured prompts from reference images for style transfer and remixing
- **VLM-guided JSON prompting**: Automatically expands short text prompts into 1,000+ word structured schemas using a fine-tuned Qwen3-VL model

### 🔧 Restructure and 🔄 Breaking Changes

- **Common module reorganization**: Moved shared functionality to `models/common/` for better code reuse
  - Unified latent creators across model families
  - Centralized scheduler implementations
  - Common quantization utilities
  - Shared model saving functionality

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: FIBO model implementation, architecture, core development

---

## [0.11.1] - 2025-11-13

# MFLUX v.0.11.1 Release Notes

### 🎨 New Model Support

- **Qwen Image Edit Support**: Added support for the Qwen Image Edit model, enabling natural language image editing capabilities
- **New command**: `mflux-generate-qwen-edit` for image editing with text instructions
- **Multiple image support**: Edit images using multiple reference images via `--image-paths` parameter
- **Model**: Uses `Qwen/Qwen-Image-Edit-2509` for high-quality image editing
- **Quantization support**: Full support for quantized models (8-bit recommended for optimal quality)

### 🔧 Improvements

- **Dedicated Qwen Image command**: Added `mflux-generate-qwen` as a dedicated command for Qwen Image model generation. The `mflux-generate` command now only supports Flux models.
- **Image comparison utility refactoring**: Refactored `image_compare.py` into a cleaner class-based structure with static methods
- **Error handling**: Moved `ReferenceVsOutputImageError` to the main exceptions module for better organization

### 🔄 Breaking Changes

⚠️ **Qwen Image Command Change**: The Qwen Image model now requires using the dedicated `mflux-generate-qwen` command instead of `mflux-generate --model qwen`. This provides better separation between Flux and Qwen model families and improves command clarity.

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: Qwen Image Edit model implementation, code refactoring

---

## [0.11.0] - 2025-10-14

# MFLUX v.0.11.0 Release Notes

### 🎨 New Model Support

- **Qwen Image Support**: Added support for the Qwen Image text-to-image model, enabling a new generation of visual content creation
- **New command**: `mflux-generate` now supports Qwen models for image generation
- **Qwen-specific features**: Full LoRA support with Qwen naming conventions, img2img support, and optimized weight handling
- **Qwen-Image-mflux-6bit Model**: Added [filipstrand/Qwen-Image-mflux-6bit](https://huggingface.co/filipstrand/Qwen-Image-mflux-6bit) quantized model to HF

### 🏗️ Major Architecture Improvements

- **Package Restructure**: Complete reorganization of the codebase to support multiple model architectures
  - Moved from flat structure to organized `models/` hierarchy (`models/flux/`, `models/qwen/`, `models/depth_pro/`)
  - Better separation of concerns with dedicated model, variant, tokenizer, and weight handler modules
  - Improved maintainability and extensibility for future model additions
- **Namespace Package**: Converted mflux to a namespace package (in preparation for mflux.mcp extension)
- **Common Module**: Extracted shared functionality into `models/common/` for better code reuse
  - Unified LoRA handling across different model types
  - Shared attention utilities
  - Common download and weight management utilities

### 📊 Metadata Enhancements

- **XMP/IPTC Metadata Support**: Added comprehensive metadata support for professional workflows
  - Write XMP and IPTC metadata to generated images
  - Industry-standard metadata formats for better compatibility with professional image tools
  - Enhanced metadata reading and writing capabilities
- **New `mflux-info` command**: Display detailed metadata information from generated images
  - View generation parameters, model information, and settings
  - Extract metadata from any mflux-generated image
  - Professional-grade metadata inspection

### 🔧 Scheduler System

- **Scheduler Interface**: Introduced a new scheduler abstraction for better extensibility
  - Clean interface for implementing custom sampling schedulers
  - Foundation for future scheduler additions (Euler, DPM++, etc.)
  - Current implementation: Linear scheduler (existing behavior preserved)
- **Scheduler Selection**: Added `--scheduler` command-line argument for choosing schedulers

### 🐛 Bug Fixes

- **Non-Quantized Model Loading**: Fixed critical bug where locally saved non-quantized models failed to load properly
- **Model Weight Handling**: Improved weight loading reliability for edge cases

### 🔧 Developer Experience

- **MLX 0.29.2 Support**: Updated MLX dependency to support the latest version (mlx>=0.27.0,<0.30.0)
- **Python 3.13 Support**: Unblocked sentencepiece and torch dependencies for Python 3.13
  - Updated dependency specifications for better Python 3.13 compatibility
  - Ensured smooth experience on latest Python versions
- **Test Improvements**: Enhanced image comparison logic to allow similar images that are "close enough"
  - More robust test suite that accommodates minor numerical differences
  - Reduced false positives in image generation tests
- **CI Updates**: Removed Claude CI agent (replacement coming soon)

### 🔄 Breaking Changes

⚠️ **Import Path Changes**: Due to the package restructure, some internal import paths have changed. If you're using mflux as a library and importing internal modules directly, you may need to update your imports:
- Flux modules moved from `mflux.flux.*` to `mflux.models.flux.*`
- Common utilities moved to `mflux.models.common.*`
- CLI tools remain unchanged and fully backward compatible

### 👩‍💻 Contributors

- **Filip Strand (@filipstrand)**: Qwen model support, package restructure, core development
- **Alessandro Rizzo (@azrahello)**: XMP/IPTC metadata support, info command implementation
- **Anthony Wu (@anthonywu)**: Scheduler interface, namespace package conversion, Python 3.13 improvements, bug fixes

---

## [0.10.0] - 2025-08-04

# MFLUX v.0.10.0 Release Notes

### 🎨 Model Improvements

- **FLUX.1 Krea [dev] Support!**
- **FLUX.1-Krea-dev-mflux-4bit Model**: Added [filipstrand/FLUX.1-Krea-dev-mflux-4bit](https://huggingface.co/filipstrand/FLUX.1-Krea-dev-mflux-4bit) quantized model to HF
- **FLUX.1-Kontext-dev-mflux-4bit Model**: Added [akx/FLUX.1-Kontext-dev-mflux-4bit](https://huggingface.co/akx/FLUX.1-Kontext-dev-mflux-4bit) quantized model to HF, contributed by @akx

### ✨ New Features

- **5-bit Quantization Support**: Added support for 5-bit quantization as a new option alongside existing 3, 4, 6, and 8-bit quantization levels

### 🔧 Improvements

- **Enhanced Default Inference Steps**: Increased default inference steps for dev models from 14 to 25 for improved image quality
- **Multiple Model Aliases Support**: Improved model configuration system to properly support multiple aliases per model, making model selection more flexible and robust

### 🐛 Bug Fixes

- **LoRA Resume Training**: Fixed critical bug where adapters created after training interruption would fail to load for generation with `AttributeError: 'list' object has no attribute 'weight'`. The issue occurred because the resume loading logic wasn't properly handling layers that are legitimately lists in the transformer architecture (like `attn.to_out`). (see [#224](https://github.com/filipstrand/mflux/issues/224))

### 🔧 Technical Requirements

- **MLX Compatibility**: This release assumes MLX 0.27.0 and upwards for optimal performance and compatibility
- **MLX Compatibility for test**: Fix MLX version to 0.27.1 for image generation tests
- **Non-strict Weight Updates**: Explicitly added non-strict mode (`strict=False`) for weight updates to maintain compatibility with later MLX versions that enforce stricter weight validation by default

### 👩‍💻 Developer Experience

- **Streamlined Release Process**: Removed TestPyPi publishing step from release workflow for simplified deployment

### 🙏 Contributors

- **[@filipstrand](https://github.com/filipstrand)** - FLUX.1 Krea [dev] model support, 5-bit quantization, enhanced defaults, and various improvements
- **[@akx](https://github.com/akx)** - Added 4-bit quantized Kontext model to HF

---

## [0.9.6] - 2025-07-20

# MFLUX v.0.9.6 Release Notes

### 🔧 Technical Details

- Cap the upper MLX dependency to a known working version (0.26.1) to avoid compatibility issues with newer MLX releases that enforce stricter weight validation (see [#238](https://github.com/filipstrand/mflux/pull/238))

## [0.9.5] - 2025-07-17

# MFLUX v.0.9.5 Release Notes

### 🐛 Bug Fixes

- **Fixed faulty imports**: Corrected import issues in the mflux module to ensure proper package initialization and functionality

## [0.9.4] - 2025-07-17

# MFLUX v.0.9.4 Release Notes

### 🛠️ Dependency Updates

- Expanded MLX dependency range from `mlx>=0.22.0,<=0.26.1` to `mlx>=0.22.0,<0.27.0` to support newer MLX versions

### 🔧 Developer Experience

- Refactor the release script into a reusable Python module for better maintainability

## [0.9.3] - 2025-07-08

# MFLUX v.0.9.3 Release Notes

### 😖 Revert "Offline Resilience" change

On a "cold start" where user has not previously downloaded the requested model, the workflow does not successfully request the download of all the expected files, blocking the image generation workflow for first time users. The feature will be re-evaluated carefully after this hot fix.

## [0.9.2] - 2025-07-08

# MFLUX v.0.9.2 Release Notes

### 🏗️ Build System Improvements

- **Updated build backend**: Migrated from setuptools to modern `uv build` backend for faster and more reliable package builds
- **Enhanced artifact exclusion**: Optimized distribution packages by excluding documentation assets (~27MB) and example images (~5MB) from published packages
- **New `make build` command**: Added development build command for testing distribution packages and validating sizes

### 🗃️ Offline Resilience

- **Local-first behavior**: Implemented cache-first downloading to improve resilience when HuggingFace Hub or network connectivity is unavailable
- **Graceful fallback**: System automatically uses cached model files when available, falling back to downloads only when necessary
- **Improved reliability**: Enhanced model loading reliability in environments with unstable internet connections

### 🔧 Developer Experience

- **Release script improvements**: Enhanced release automation with better error handling and duplicate version detection
- **Build system fixes**: Fixed minor typos in Makefile that could cause build issues

## Contributors

- **Anthony Wu (@anthonywu)**: Build system modernization, offline resilience implementation
- **Filip Strand (@filipstrand)**: Release automation improvements, build fixes

---

## [0.9.1] - 2025-07-04

# MFLUX v.0.9.1 Release Notes

### 🛠️ Dependency Fixes

- Restricted MLX dependency upper bound to **0.26.1** (`mlx>=0.22.0,<=0.26.1`) to prevent incompatibility issues with MLX 0.26.2.

### 🎨 Inpaint Mask Tool Improvements

- Enhanced interactive inpaint masking tool with additional shape options (ellipse, rectangle, and free-hand drawing).
- Added eraser mode for precise mask corrections.
- Implemented undo/redo history for non-destructive editing when crafting masks.

### 👩‍💻 Developer Experience

- Introduced initial `mypy` static-type checking configuration and performed a first round of type-hint clean-up across the codebase.
- Upgraded *pre-commit* hooks and addressed newly surfaced lint warnings for a cleaner commit experience.

## Contributors

- **Filip Strand (@filipstrand)**
- **Anthony Wu (@anthonywu)**

---

## [0.9.0] - 2025-06-28

# MFLUX v.0.9.0 Release Notes

## Major New Features

### 📸 FLUX.1 Kontext

- **Added FLUX.1 Kontext support**: Official Black Forest Labs model for character consistency, local editing, and style reference
- **New command**: `mflux-generate-kontext` for image-guided generation with text instructions
- **Advanced image editing capabilities**: Sequential editing, style transfer, character consistency, and local modifications
- **Comprehensive documentation**: Detailed prompting guide with tips, templates, and best practices
- **Automatic model handling**: Uses `dev-kontext` model configuration with optimized defaults

### 🖼️ Scale Factor Support for Image Upscaling

- **Enhanced upscaling dimensions**: Added support for scale factors (e.g., `2x`, `1.5x`) in addition to absolute pixel values
- **Mixed dimension types**: Ability to combine scale factors and absolute values (e.g., `--height 2x --width 1024`)
- **Auto dimension handling**: Use `auto` to preserve original image dimensions
- **Safety warnings**: Automatic warnings when requested dimensions exceed recommended limits
- **Pixel-perfect scaling**: Scale factors automatically align to 16-pixel boundaries for optimal results

### ⌨️ Shell Completions

- **ZSH completion support**: Full tab completion for all mflux CLI commands and arguments
- **Smart completions**: Context-aware completions for model names, quantization levels, LoRA styles, and file paths
- **Easy installation**: Simple `mflux-completions` command for automatic setup
- **Dynamic generation**: Completions stay in sync with code changes and new commands
- **Comprehensive coverage**: Supports all 15+ mflux commands with proper argument validation

### 🗂️ Cache Management Improvements

- **Platform-native caching**: Uses `platformdirs` for macOS-idiomatic cache locations (`~/Library/Caches/mflux/`)
- **Automatic migration**: Seamless migration from legacy `~/.cache/mflux` to new platform-appropriate locations
- **Environment variable support**: `MFLUX_CACHE_DIR` for custom cache locations
- **Improved organization**: Separate cache directories for different types of data (models, LoRAs, etc.)
- **Backward compatibility**: Automatic symlink creation for legacy path compatibility

## Breaking Changes

### 🔧 Python API Class Naming Standardization

- **Class rename**: `FluxInContextFill` is now `Flux1InContextFill` to follow consistent naming convention
- **Class rename**: `FluxConceptFromImage` is now `Flux1ConceptFromImage` to follow consistent naming convention
- **Breaking change for library users**: If you import these classes directly in Python code, you may need to update your imports
- **CLI tools unaffected**: All command-line tools (`mflux-generate-*`) continue to work without changes

## Contributors

Contributors:
- **Anthony Wu (@anthonywu)**: Scale factor support, shell completions, cache refactor
- **Filip Strand (@filipstrand)**: Kontext support, class naming standardization, core development

## [0.8.0] - 2025-06-14

# MFLUX v.0.8.0 Release Notes

## Experimental AI Features

### 👗 CatVTON (Virtual Try-On)
- **[EXPERIMENTAL]** Added virtual try-on capabilities using in-context learning via `mflux-generate-in-context-catvton`
- Support for person image, person mask, and garment image inputs for comprehensive virtual clothing try-on
- Automatic prompting for virtual try-on scenarios with optimized default prompts
- Side-by-side generation showing garment product shot alongside styled result
- AI-powered virtual clothing fitting with realistic lighting and fabric properties

### ✏️ IC-Edit (In-Context Editing)
- **[EXPERIMENTAL]** Added natural language image editing capabilities via `mflux-generate-in-context-edit`
- Natural language image editing using simple text instructions like "make the hair black" or "add sunglasses"
- Automatic diptych template formatting for optimal editing results
- Optimal resolution auto-sizing for 512px width (the resolution IC-Edit was trained on)
- Specialized LoRA automatically downloaded and applied for enhanced editing capabilities

## Enhanced Generation Control

### 🔎 Image Upscaling
- **Built-in upscaling capabilities**: Enhanced image quality and resolution enhancement for generated images
- Seamless integration with existing generation workflow
- Professional-grade upscaling for production-ready outputs

## Interpretability research

### 🧠 Concept Attention
- **Enhanced image generation control**: Fine-grained control over image generation focus areas using attention-based concepts
- Improved composition and subject handling for more precise artistic direction
- Advanced attention mechanisms for better understanding of prompt concepts

## Workflow & Performance Improvements

### 🪫 Battery Saver
- **Power management**: Automatic power optimization during extended generation sessions
- Configurable power-saving modes specifically designed for laptop users
- Smart resource management for long-running batch operations

### 📝 Prompt File Support
- **File-based prompt input**: Batch operations via `--prompt-file` for large-scale generation projects
- Dynamic prompt updates for large batch generation workflows
- Support for external prompt management and automation systems

### 🔄 Redux Function Balancing
- **Enhanced Redux capabilities**: Improved control over image-to-image transformation strength
- Better quality variations with adjustable parameters for more predictable results
- Refined Redux algorithm for more natural image variations

### 📥 Stdin Prompt Support
- **LLM Integration Ready**: Added support for providing prompts via stdin using `--prompt -`
- Enables seamless integration with LLMs and other text generation tools
- Supports both single-line and multi-line prompts through stdin
- Perfect for automation workflows and dynamic prompt generation
- Example usage: `echo "A beautiful landscape" | mflux-generate --prompt -`

## Developer Experience

### 🔧 LORA_LIBRARY_PATH Improvements
- **Unix-style resource discovery**: Enhanced LoRA library path handling for better organization
- Improved path handling for LoRA weight discovery across multiple directories
- Better cross-platform compatibility for LoRA management

### 🧪 Testing & Documentation
- New command-line arguments for both experimental features with comprehensive help
- Comprehensive argument parser tests for new functionality
- Updated documentation with experimental feature warnings and usage guidelines
- Added note about upcoming FLUX.1 Kontext model from Black Forest Labs

## Architecture Improvements

### 📚 Documentation Structure
- Refactored "In-Context LoRA" section to "In-Context Generation" with clear subcategories
- Enhanced documentation structure for better organization and user navigation
- Improved categorization of experimental vs stable features

### 🔄 Code Architecture Changes
- **Class rename**: `Flux1InContextLora` is now `Flux1InContextDev` to better reflect the dev model variant
- **Module reorganization**: Moved from `mflux.community.in_context_lora.flux_in_context_lora` to `mflux.community.in_context.flux_in_context_dev`
- **Breaking change for library users**: If you import the class directly, update your imports accordingly


### ⚡ Performance Optimizations
- Updated MLX dependency to latest version for improved performance and stability
- Removed PyTorch dependency for DepthPro model, significantly reducing installation requirements
- Streamlined dependencies for faster installation and reduced disk usage

## Experimental Notice

⚠️ **Important**: CatVTON and IC-Edit features are experimental and may be removed or significantly changed in future updates. These features represent cutting-edge AI capabilities that are still under active development.

## Contributors

Special thanks to the following contributors for their exceptional work since v0.7.1:
- **Anthony Wu (@anthonywu)**: Battery Saver implementation, Prompt File Support, Stdin Prompt Support, LORA_LIBRARY_PATH improvements
- **Alessandro (@azrahello)**: Redux Function Balancing enhancements
- **Filip Strand (@filipstrand)**: Core development, experimental features integration, infrastructure improvements

## [0.7.1] - 2025-05-06

# MFLUX v.0.7.1 Release Notes

## New Features

### 🎭 Multi-LoRA Support
- **Multiple LoRA Loading**: Added support for loading multiple LoRA adapters simultaneously when using the in-context feature
- Enhanced creative flexibility by combining multiple artistic styles in a single generation
- Reference: [GitHub Issue #178](https://github.com/filipstrand/mflux/issues/178)

## [0.7.0] - 2025-04-25
# MFLUX v.0.7.0 Release Notes

## Major New Features

### 🖌️ FLUX.1 Tools | Fill

- Added support for the FLUX.1-Fill model for inpainting and outpainting
- Introduced `mflux-generate-fill` command-line tool for selective image editing
- Implemented interactive mask creation tool to easily mark areas for regeneration
- Added outpainting capabilities with customizable canvas expansion
- Includes helper tools for creating outpaint image canvases and masks

### 🔍 FLUX.1 Tools | Depth

- Added support for the FLUX.1-Depth model for depth-conditioned image generation
- Implemented Apple's ML Depth Pro model in MLX for state-of-the-art depth map extraction
- Added `mflux-generate-depth` and `mflux-save-depth` command-line tools
- Added ability to use either auto-generated depth maps or custom depth maps

### 🔄 FLUX.1 Tools | Redux

- Added Redux tool as a new image variation technique
- Implemented a different approach compared to standard image-to-image generation
- Uses image embedding joined with T5 text encodings for more natural variations
- Added Redux-specific weight handlers and initialization

## New Models

### 🔎 Apple ML Depth Pro

- Added native MLX implementation of Apple's ML Depth Pro model for both separate use, and as a part of the Depth tool functionality

### 🖼️ Google SigLIP Vision Transformer

- Added SigLIP vision model for the Redux functionality

## Architecture Improvements

### 💾 Weight Management Improvements

- Added support for saving MFLUX version information in model metadata

### 🧠 Memory Optimization

- Additional improvements to the `--low-ram` option
- Better memory management for image generation models

## Contributors

- @anthonywu 
- @ssakar 
- @akx 

## [0.6.2] - 2025-03-13

# MFLUX v.0.6.2 Release Notes

## Bug Fixes

### 💾 Model Saving Fix
- **Fixed local model saving**: Resolved bug preventing users from saving models locally with `mflux-save`
- Restored full functionality for local model storage and management

## [0.6.1] - 2025-03-11

# MFLUX v.0.6.1 Release Notes

## Bug Fixes

### 🛑 Image Generation Interruption
- **Fixed interruption flow**: Properly handles interruptions during image generation, ensuring graceful stops even when no callbacks are registered
- **Keyboard interrupt handling**: Ensures image generation can be stopped via Ctrl+C in all diffusion model variants (standard Flux, ControlNet, and In-Context LoRA)
- Relocated `StopImageGenerationException` from stepwise handler to main generation functions for more robust interruption system

## Test Stability Improvements

### 🧪 Test Reliability
- **Fixed sporadic test failures**: Resolved intermittent failures in auto-seeds test case when using random seed count of 1
- Improved test consistency and reliability

## Code Quality Improvements

### 🔧 Code Standards
- **Formatting and linting fixes**: Fixed various formatting issues that were missed in the v0.6.0 release
- Enhanced code consistency and maintainability

## [0.6.0] - 2025-03-05
# MFLUX v.0.6.0 Release Notes

## Major New Features

### 🌐 Third-Party HuggingFace Model Support
- Comprehensive ModelConfig refactor to support compatible HuggingFace dev/schnell models
- Added ability to use models like `Freepik/flux.1-lite-8B-alpha` and `shuttleai/shuttle-3-diffusion`
- New `--base-model` parameter to specify which base architecture (dev or schnell) a third-party model is derived from
- Maintains backward compatibility while opening up the ecosystem to community-created models

### 🎭 In-Context LoRA
- Added support for In-Context LoRA, a powerful technique that allows you to generate images in a specific style based on a reference image without requiring model fine-tuning
- Introduced a new command-line tool: `mflux-generate-in-context`
- Includes 10 pre-defined styles from the Hugging Face ali-vilab/In-Context-LoRA repository
- Detailed documentation on how to use this feature effectively with prompting tips and best practices

### 🔌 Automatic LoRA Downloads
- Added ability to automatically download LoRAs from Hugging Face when specified by repository ID
- Simplifies workflow by eliminating the need to manually download LoRA files before use

### 🧠 Memory Optimizations
- Added `--low-ram` option to reduce GPU memory usage by constraining the MLX cache size and releasing text encoders and transformer components after use
- Implemented memory saver for ControlNet to reduce RAM requirements
- General memory usage optimizations throughout the codebase

### 🗜️ Enhanced Quantization Options
- Added support for 3-bit and 6-bit quantization (requires mlx > v0.21.0)
- Expanded quantization options now include 3, 4, 6, and 8-bit precision

## ⚠️Breaking changes

Previously saved quantized models will not work for v.0.6.0 and later.  See #149 for more details.

## Interface Improvements

### 🔧 Modified Parameters

- The previous `--init-image-path` parameter is now `--image-path` 
- The previous `--init-image-strength` parameter is now `--image-strength` 

### 🖼️ Image Generation Enhancements
- Added `--auto-seeds` option to generate multiple images with random seeds in a single command
- Added option to override previously saved test images
- Added `--controlnet-save-canny` option to save the Canny edge detection reference image used by ControlNet
- Improved handling of edge cases for img2img generation

### 🔄 Callback System
- Implemented a general callback mechanism for more flexible image generation pipelines
- Added support for before-loop callbacks to accept latents
- Enhanced StepwiseHandler to include initial latent

## Architecture Improvements

### 🏗️ Code Refactoring
- Removed 'init' prefix for a more general interface
- Removed `ConfigControlnet` - the `controlnet_strength` attribute is now on `Config`
- Simplified quantization by removing unnecessary class predicates 
- Refactored model configuration system
- Refactored transformer blocks for better maintainability
- Unified attention mechanism in single and joint attention blocks
- Added support for variable numbers of transformer blocks
- Optimized with fast SDPA (Scaled Dot-Product Attention)
- Added PromptCache for small optimization when generating with repeated prompts

### 🧰 Developer Tools
- Added Batch Image Renamer tool as an isolated uv run script
- Added descriptive comments for attention computations

## Compatibility Updates
- Updated to support the latest mlx version
- Fixed compatibility issues with HuggingFace dev/schnell models

## Bug Fixes
- Fixed handling of edge cases for img2img generation
- Various small fixes and improvements throughout the codebase


## Contributors

- @anthonywu
- @ssakar
- @azrahello
- @DanaCase

## [0.5.1] - 2024-12-23

# MFLUX v.0.5.1 Release Notes

## Bug Fixes

### 🔧 LoRA Loading Fix
- **Quantized model LoRA compatibility**: Fixed critical bug where locally saved quantized models failed to set LoRA weights
- Users can now successfully combine local quantized models with external LoRA adapters
- Improved reliability for advanced workflows combining quantization and LoRA fine-tuning

## [0.5.0] - 2024-12-22

# MFLUX v.0.5.0 Release Notes

## Major New Features

### 🎛️ DreamBooth Fine-tuning
- **DreamBooth support**: Introduced V1 of fine-tuning support in MFLUX
- Enables custom model training for personalized image generation
- Full fine-tuning pipeline with training configuration options

## Architecture Improvements

### 🔧 Weight Management Overhaul
- **Rewritten LoRA handling**: Completely rewritten LoRA weight handling system
- Improved performance and reliability for LoRA operations
- Better support for complex LoRA workflows

## Developer Experience

### 🧪 Testing & Quality
- **Enhanced test coverage**: Added comprehensive tests for new and existing features
- Multi-LoRA testing support
- Local model saving test coverage

### 📊 New Dependencies
- **Matplotlib integration**: Added matplotlib for visualizing training loss during fine-tuning
- **TOML support**: Added TOML library for better handling of MFLUX version metadata
- Enhanced configuration management

## [0.4.1] - 2024-10-29

# MFLUX v.0.4.1 Release Notes

## Bug Fixes

### 🐛 Image Generation Fixes
- **Img2img resolution fix**: Fixed img2img functionality for non-square image resolutions
- Improved compatibility with various aspect ratios

## [0.4.0] - 2024-10-28

# MFLUX v.0.4.0 Release Notes

## Major New Features

### 🖼️ Image-to-Image Generation
- **Img2Img Support**: Introduced the ability to generate images based on an initial reference image
- Transform existing images using AI-powered generation techniques
- Control the strength of transformation to balance between original image preservation and creative generation
- Perfect for iterating on designs and creating variations of existing artwork

### 📊 Metadata-Driven Generation
- **Image Generation from Metadata**: Added support to generate images directly from provided metadata files
- Streamlined workflow for recreating images with specific parameters
- Enhanced reproducibility for professional and research workflows
- Automated parameter loading from previously generated images

### 🔍 Real-time Generation Monitoring
- **Progressive Step Output**: Optionally output each step of the image generation process for real-time monitoring
- Visual feedback during generation for better understanding of the AI process
- Debug and fine-tune generation parameters by observing intermediate steps
- Educational tool for understanding diffusion model progression

## Developer Experience Improvements

### 🛠️ Enhanced Command-Line Interface
- **Improved argument handling**: Enhanced parsing and validation for command-line arguments
- Better error messages and user guidance for parameter configuration
- More intuitive command structure for complex generation workflows

### 🧪 Testing & Quality Assurance
- **Automated Testing**: Added comprehensive automatic tests for image generation and command-line argument handling
- Improved reliability and stability for all generation modes
- Continuous integration testing for better code quality

### 🔧 Development Workflow
- **Pre-Commit Hooks**: Integrated pre-commit hooks with `ruff`, `isort`, and typo checks for better code consistency
- Enhanced developer experience with automated code quality checks
- Streamlined contribution process for open source development

## [0.3.0] - 2024-09-24

# MFLUX v.0.3.0 Release Notes

## Major New Features

### 🕹️ ControlNet Support
- **ControlNet Canny support**: Added Canny edge detection ControlNet functionality for precise image control
- Enhanced control over image generation with edge-guided conditioning

## Model Export Improvements

### 📦 Advanced Model Export
- **Quantized model export with LoRA**: Added ability to export quantized models with LoRA weights baked in
- Streamlined deployment for fine-tuned models

## Developer Experience

### 🛠️ Development Tools
- **Enhanced development workflow**: Improved developer experience with uv, ruff, makefile, pre-commit hooks
- Better code quality tools and automated checks
- Streamlined contribution process

## Legal & Licensing

### ⚖️ Open Source License
- **Official MIT license**: Established clear open source licensing for the project
- Legal clarity for users and contributors

## [0.2.1] - 2024-09-14

# MFLUX v.0.2.1 Release Notes

## Improvements

### 🔧 LoRA Enhancements
- **Enhanced LoRA support**: Improved compatibility and performance for LoRA weight loading
- Better integration with existing workflows
- Refined handling of LoRA adapters

## [0.2.0] - 2024-09-07

# MFLUX v.0.2.0 Release Notes

## Major Milestone

### 🚀 Official PyPI Release
- **First official PyPI release**: `pip install mflux` - making MFLUX easily installable for everyone
- Big thanks to @deto for letting us have the "mflux" name on PyPI!

## New Features

### 🎨 Core Image Generation
- **Command-line tools**: Introduced dedicated commands for better user experience
  - `mflux-generate` for generating images
  - `mflux-save` for saving quantized models to disk
- **🗜️ Quantization support**: Added support for quantized models with 4-bit and 8-bit precision
- **LoRA weights**: Added support for loading trained LoRA (Low-Rank Adaptation) weights
- **Automatic metadata**: Images now automatically save metadata when generated

## Developer Experience

### 📦 Distribution
- Official packaging and distribution through PyPI
- Simplified installation process for end users
- Professional project structure and naming
