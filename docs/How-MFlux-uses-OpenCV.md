# How MFlux uses OpenCV

## Summary

MFlux uses OpenCV (`opencv-python`, imported as `cv2`) only to prepare ControlNet control images. OpenCV does no model work. All model math runs in MLX.

All `import cv2` lines are lazy. They are inside the function that needs them, with the comment `# only here: OpenCV is optional`. Thus MFlux starts and runs without OpenCV until a code path needs one of these functions.

Test result (OpenCV blocked with `sys.modules["cv2"] = None`): all 46 CLI entry points import with no error.

## Where MFlux calls OpenCV

| File | OpenCV calls | Purpose |
|---|---|---|
| `models/flux/variants/controlnet/controlnet_util.py` | `cv2.Canny` | Canny edge map from the input image |
| `models/z_image/variants/controlnet/controlnet_util.py` | `cv2.Canny` | `canny` control |
| same file | `cv2.createLineSegmentDetector`, `cv2.line` | `mlsd` control (straight lines, no neural model) |
| `models/openpose/openpose.py` | `cvtColor`, `resize`, `copyMakeBorder`, `GaussianBlur`, `circle`, `ellipse`, `fillConvexPoly`, `addWeighted` | Pre- and post-processing for the MLX OpenPose model, and drawing of the skeleton |

Paths are relative to `src/mflux/`.

## What needs OpenCV

| Feature | Why |
|---|---|
| `mflux-generate-controlnet` with a Canny ControlNet (model name has `Canny`, for example `InstantX/FLUX.1-dev-Controlnet-Canny`) | Canny edges from `--controlnet-image-path` |
| `mflux-generate-z-image-controlnet` with `canny` | `cv2.Canny` |
| `mflux-generate-z-image-controlnet` with `mlsd` | LineSegmentDetector. It needs `opencv-python>=4.8`. |
| `mflux-generate-z-image-controlnet` with `pose` | OpenPose image steps. This control also needs torch for the weights. |

Without OpenCV, these features stop with `ImportError` when they prepare the control image.

## What runs without OpenCV

All other commands. This includes:

- Z-Image ControlNet `depth` and `hed` controls
- Krea 2 depth ControlNet
- Qwen 2.1 ControlNet (you supply a preprocessed control image)
- `mflux-upscale-controlnet`
- all text-to-image and edit commands

## macOS and Linux/CUDA

`pyproject.toml` lists `opencv-python>=4.10.0,<6.0` for all platforms. The installed version in this repo is 4.13.

On macOS, the wheel installs and operates with no extra steps.

On Linux, `opencv-python` is the GUI build. It loads system libraries such as `libGL.so.1`. A minimal server or container image often does not have them, and then `import cv2` fails. Two fixes:

1. Install the system library, for example `apt install libgl1 libglib2.0-0`.
2. Replace `opencv-python` with `opencv-python-headless`. MFlux uses no OpenCV GUI functions, so the headless build is sufficient.

OpenCV runs on the CPU. MFlux does not use OpenCV CUDA.

## How to install without OpenCV

OpenCV is a hard dependency, so a normal install pulls it. Two options:

1. Install as usual, then remove OpenCV: `pip uninstall opencv-python`.
2. Install mflux with `--no-deps`, then install the other dependencies from `pyproject.toml` yourself.

After this, the features in "What needs OpenCV" fail, and all other features operate as usual.
