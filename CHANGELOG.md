# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **[EXPERIMENTAL]** CatVTON (Virtual Try-On) support via `mflux-generate-in-context-catvton`
  - Virtual try-on capabilities using in-context learning
  - Support for person image, person mask, and garment image inputs
  - Automatic prompting for virtual try-on scenarios
  - Side-by-side generation showing garment and styled result
- **[EXPERIMENTAL]** IC-Edit (In-Context Editing) support via `mflux-generate-in-context-edit`
  - Natural language image editing using text instructions
  - Automatic diptych template formatting
  - Optimal resolution auto-sizing for 512px width
  - Specialized LoRA automatically downloaded and applied
- New command-line arguments for both experimental features
- Comprehensive argument parser tests for new functionality
- Updated documentation with experimental feature warnings
- Added note about upcoming FLUX.1 Kontext model from Black Forest Labs

### Changed
- Refactored "In-Context LoRA" section to "In-Context Generation" with subcategories
- Enhanced documentation structure for better organization

### Experimental
- CatVTON and IC-Edit features may be removed or significantly changed in future updates

## [0.7.1] - 2025-05-06
### Added
- Added support for loading multiple LoRAs when using the in-context feature

## [0.7.0] - 2025-04-25
### Added
- **🖌️ FLUX.1 Tools | Fill**: Support for FLUX.1-Fill model for inpainting and outpainting
  - `mflux-generate-fill` command-line tool for selective image editing
  - Interactive mask creation tool to easily mark areas for regeneration
  - Outpainting capabilities with customizable canvas expansion
  - Helper tools for creating outpaint image canvases and masks
- **🔍 FLUX.1 Tools | Depth**: Support for FLUX.1-Depth model for depth-conditioned image generation
  - Apple's ML Depth Pro model in MLX for state-of-the-art depth map extraction
  - `mflux-generate-depth` and `mflux-save-depth` command-line tools
  - Ability to use either auto-generated depth maps or custom depth maps
- **🔄 FLUX.1 Tools | Redux**: Redux tool as a new image variation technique
  - Different approach compared to standard image-to-image generation
  - Uses image embedding joined with T5 text encodings for more natural variations
  - Redux-specific weight handlers and initialization
- **🔎 Apple ML Depth Pro**: Native MLX implementation for both separate use and Depth tool functionality
- **🖼️ Google SigLIP Vision Transformer**: SigLIP vision model for Redux functionality

### Changed
- Additional improvements to the `--low-ram` option
- Better memory management for image generation models
- Support for saving MFLUX version information in model metadata

## [0.6.2] - 2025-03-13
### Fixed
- Fixed bug preventing users from saving models locally with `mflux-save`

## [0.6.1] - 2025-03-11
### Fixed
- **Fixed image generation interruption flow**: Properly handles interruptions during image generation, ensuring graceful stops even when no callbacks are registered
- **Fixed keyboard interrupt handling**: Ensures image generation can be stopped via Ctrl+C in all diffusion model variants
- **Fixed sporadic test failures**: Resolved auto-seeds test case intermittent failures
- **Code formatting and linting**: Fixed various formatting issues

## [0.6.0] - 2025-03-05
### Added
- **🌐 Third-Party HuggingFace Model Support**: Comprehensive ModelConfig refactor to support compatible HuggingFace dev/schnell models
  - Ability to use models like `Freepik/flux.1-lite-8B-alpha` and `shuttleai/shuttle-3-diffusion`
  - New `--base-model` parameter to specify base architecture (dev or schnell)
- **🎭 In-Context LoRA**: Support for In-Context LoRA technique
  - New command-line tool: `mflux-generate-in-context`
  - 10 pre-defined styles from Hugging Face ali-vilab/In-Context-LoRA repository
- **🔌 Automatic LoRA Downloads**: Ability to automatically download LoRAs from Hugging Face
- **🧠 Memory Optimizations**: `--low-ram` option to reduce GPU memory usage
- **🔢 Enhanced Quantization Options**: Support for 3-bit and 6-bit quantization
- **🔧 Interface Improvements**: `--auto-seeds` option to generate multiple images with random seeds

### Changed
- **⚠️ BREAKING**: Previously saved quantized models will not work for v.0.6.0 and later
- **⚠️ BREAKING**: `--init-image-path` parameter is now `--image-path`
- **⚠️ BREAKING**: `--init-image-strength` parameter is now `--image-strength`
- Removed `ConfigControlnet` - `controlnet_strength` attribute is now on `Config`
- Refactored model configuration system and transformer blocks
- Added `--controlnet-save-canny` option to save Canny edge detection reference image

## [0.5.1] - 2024-12-23
### Fixed
- Fixed bug which caused locally saved quantized models to fail to set LoRA weights

## [0.5.0] - 2024-12-22
### Added
- **DreamBooth fine-tuning support**: V1 of fine-tuning support in MFLUX
- Matplotlib as dependency for visualizing training loss
- TOML library as dependency for better handling of MFLUX version metadata

### Changed
- Completely rewritten LoRA weight handling
- Better test coverage including multi-LoRA and local model saving tests

## [0.4.1] - 2024-10-29
### Fixed
- Fixed img2img for non-square image resolutions

## [0.4.0] - 2024-10-28
### Added
- **Img2Img Support**: Generate images based on initial reference image
- **Image Generation from Metadata**: Generate images directly from metadata files
- **Progressive Step Output**: Optionally output each step of image generation process
- Enhanced command-line argument handling
- Automated testing for image generation and command-line arguments
- Pre-commit hooks with `ruff`, `isort`, and typo checks

## [0.3.0] - 2024-09-24
### Added
- **ControlNet Canny support**
- Enhanced dev experience with uv, ruff, makefile, pre-commit
- Ability to export quantized model with LoRA weights baked in
- Official MIT license

## [0.2.1] - 2024-09-14
### Changed
- Better LoRA support

## [0.2.0] - 2024-09-07
### Added
- **Official PyPI release**: `pip install mflux` (Big thanks to @deto for the name!)
- New commands:
  - `mflux-generate` for generating an image
  - `mflux-save` for saving a quantized model to disk
- Support for quantized models (4 bit and 8 bit)
- Support for loading trained LoRA weights
- Automatically saves metadata when saving an image 