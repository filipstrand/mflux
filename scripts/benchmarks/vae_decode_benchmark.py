"""Measure the VAE decode of an mflux model without running the transformer for each test.

capture: runs the model once for each prompt and saves the latent that reaches VAEUtil.decode, plus the VAE
         weights. It does not decode.
decode:  loads only the saved VAE and decodes each saved latent in each mode. It reports the peak MLX memory,
         the median time and the PSNR against the legacy decode. With --profile it also reports the peak of
         each decoder block in the low_memory mode.

Example:
    uv run python scripts/benchmarks/vae_decode_benchmark.py capture --model z-image-turbo -q 8 \\
        --prompts prompts.txt --out /tmp/vae-bench/z-image-turbo
    uv run python scripts/benchmarks/vae_decode_benchmark.py decode --run-dir /tmp/vae-bench/z-image-turbo --profile
"""

import argparse
import importlib
import json
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from PIL import Image

from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS
from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.vae.tiling_config import TilingConfig
from mflux.models.common.vae.vae_util import VAEUtil

# Model key prefix -> the class that generates images.
MODEL_CLASSES = {
    "flux3-": "mflux.models.flux3.variants:Flux3Klein",
    "flux2-klein": "mflux.models.flux2.variants:Flux2Klein",
    "ernie-image": "mflux.models.ernie_image.variants.txt2img.ernie_image:ErnieImage",
    "z-image": "mflux.models.z_image.variants.z_image:ZImage",
    "lens": "mflux.models.lens.variants.txt2img.lens_image:LensImage",
    "ideogram-4": "mflux.models.ideogram4.variants.txt2img.ideogram4:Ideogram4",
}
MODES = {
    "legacy": (VAEBlockDecoder.LEGACY, None),
    "tiled": (VAEBlockDecoder.LEGACY, TilingConfig()),
    "low_memory": (VAEBlockDecoder.LOW_MEMORY, None),
    "full_precision": (VAEBlockDecoder.FULL_PRECISION, None),
}
GB = 1e9


class _Captured(Exception):
    pass


class VAEDecodeBenchmark:
    @staticmethod
    def capture(args) -> None:
        args.out.mkdir(parents=True, exist_ok=True)
        prefix = next(p for p in MODEL_CLASSES if args.model.startswith(p))
        module_name, class_name = MODEL_CLASSES[prefix].split(":")
        model_class = getattr(importlib.import_module(module_name), class_name)
        model = model_class(model_config=AVAILABLE_MODELS[args.model], quantize=args.quantize, model_path=args.model_path)  # fmt: skip
        VAEDecodeBenchmark._save_vae(model.vae, getattr(model, "bits", None), args.out)
        steps = args.steps or MODEL_INFERENCE_STEPS[args.model]
        captured: dict = {}

        def save_latent(vae, latent, tiling_config=None):
            mx.eval(latent)
            captured["latent"] = latent
            raise _Captured

        prompts = [line.strip() for line in args.prompts.read_text().splitlines() if line.strip() and not line.startswith("#")]  # fmt: skip
        with patch.object(VAEUtil, "decode", staticmethod(save_latent)):
            for index, prompt in enumerate(prompts, start=1):
                path = args.out / f"p{index:02d}_s{args.seed}_{args.width}x{args.height}.safetensors"
                if index in set(args.skip) or path.exists():
                    continue
                mx.reset_peak_memory()
                start = time.perf_counter()
                try:
                    model.generate_image(seed=args.seed, prompt=prompt, num_inference_steps=steps, width=args.width, height=args.height)  # fmt: skip
                except _Captured:
                    pass
                else:
                    raise RuntimeError("VAEUtil.decode was not called")
                mx.save_safetensors(str(path), {"latent": captured.pop("latent")}, metadata={"prompt": prompt, "steps": str(steps)})  # fmt: skip
                print(json.dumps({"prompt": index, "seconds": round(time.perf_counter() - start, 1), "peak_gb": round(mx.get_peak_memory() / GB, 2)}), flush=True)  # fmt: skip
                mx.clear_cache()

    @staticmethod
    def decode(args) -> None:
        vae = VAEDecodeBenchmark._load_vae(args.run_dir)
        latents = sorted(args.run_dir.glob("p*.safetensors"))
        rows = []
        for mode in args.modes:
            out_dir = args.run_dir / mode
            out_dir.mkdir(exist_ok=True)
            vae.decode_mode, tiling_config = MODES[mode]
            results = [
                VAEDecodeBenchmark._decode_one(vae, path, tiling_config, out_dir, args.run_dir) for path in latents
            ]
            psnrs = [r["psnr_db"] for r in results if r["psnr_db"] is not None]
            rows.append(
                {
                    "mode": mode,
                    "latent_dtype": results[0]["latent_dtype"],
                    "peak_gb": max(r["peak_gb"] for r in results),
                    "median_s": statistics.median(r["seconds"] for r in results[1:] or results),
                    "psnr_min_db": min(psnrs) if psnrs else None,
                    "psnr_median_db": statistics.median(psnrs) if psnrs else None,
                    "max_mean_shift": max(r["mean_shift"] for r in results) if psnrs else None,
                }
            )
            print(json.dumps(rows[-1]), flush=True)
        (args.run_dir / "decode_results.json").write_text(json.dumps(rows, indent=2))
        if args.profile:
            VAEDecodeBenchmark._profile(vae, latents[0])

    @staticmethod
    def _decode_one(vae, path: Path, tiling_config, out_dir: Path, run_dir: Path) -> dict:
        latent = mx.load(str(path))["latent"]
        mx.eval(latent)
        mx.clear_cache()
        mx.reset_peak_memory()
        start = time.perf_counter()
        decoded = VAEUtil.decode(vae=vae, latent=latent, tiling_config=tiling_config)
        mx.eval(decoded)
        seconds = time.perf_counter() - start
        peak = mx.get_peak_memory() / GB
        pixels = VAEDecodeBenchmark._to_uint8(decoded)
        Image.fromarray(pixels).save(out_dir / f"{path.stem}.png")
        reference_path = run_dir / "legacy" / f"{path.stem}.png"
        psnr, shift = None, 0.0
        if out_dir.name != "legacy" and reference_path.exists():
            reference = np.array(Image.open(reference_path).convert("RGB")).astype(np.float64)
            diff = pixels.astype(np.float64) - reference
            mse = float((diff**2).mean())
            psnr = float("inf") if mse == 0 else round(10 * np.log10(255.0**2 / mse), 2)
            shift = float(np.abs(diff.mean(axis=(0, 1))).max())
        return {"latent_dtype": str(latent.dtype).replace("mlx.core.", ""), "seconds": seconds, "peak_gb": round(peak, 2), "psnr_db": psnr, "mean_shift": shift}  # fmt: skip

    @staticmethod
    def _profile(vae, path: Path) -> None:
        # Peak MLX memory of each block in the low_memory decode. Each block ends with an mx.eval.
        rows = []

        def wrap(name, function):
            def wrapped(*args, **kwargs):
                mx.reset_peak_memory()
                out = function(*args, **kwargs)
                rows.append((f"{name} {len(rows)}", "x".join(map(str, out.shape)), mx.get_peak_memory() / GB))
                return out

            return staticmethod(wrapped)

        latent = mx.load(str(path))["latent"]
        vae.decode_mode = VAEBlockDecoder.LOW_MEMORY
        targets = {name: getattr(VAEBlockDecoder, name) for name in ("_resnet", "_attention", "_upsample")}
        with patch.multiple(VAEBlockDecoder, **{name: wrap(name.strip("_"), fn) for name, fn in targets.items()}):
            mx.eval(VAEUtil.decode(vae=vae, latent=latent))
        print(f"{'block':<16}{'output':<20}{'peak GB':>8}")
        for name, shape, peak in rows:
            print(f"{name:<16}{shape:<20}{peak:>8.2f}")

    @staticmethod
    def _save_vae(vae, bits: int | None, out: Path) -> None:
        vae.save_weights(str(out / "vae.safetensors"))
        info = {"class": f"{type(vae).__module__}:{type(vae).__name__}", "bits": bits}
        (out / "vae.json").write_text(json.dumps(info))

    @staticmethod
    def _load_vae(run_dir: Path):
        info = json.loads((run_dir / "vae.json").read_text())
        module_name, class_name = info["class"].split(":")
        vae = getattr(importlib.import_module(module_name), class_name)()
        weights = mx.load(str(run_dir / "vae.safetensors"))
        quantized = {key.rsplit(".", 1)[0] for key in weights if key.endswith(".scales")}
        if quantized:
            # A quantized weight has in_features * bits / 32 columns and in_features / group_size scales.
            sample = next(iter(quantized))
            group_size = 64
            bits = info["bits"] or 32 * weights[f"{sample}.weight"].shape[1] // (weights[f"{sample}.scales"].shape[1] * group_size)  # fmt: skip
            nn.quantize(vae, group_size=group_size, bits=bits, class_predicate=lambda path, module: path in quantized)
        vae.load_weights(list(weights.items()))
        mx.eval(vae.parameters())
        print(f"Loaded {class_name}: {sum(v.nbytes for _, v in tree_flatten(vae.parameters())) / GB:.2f} GB of weights")
        return vae

    @staticmethod
    def _to_uint8(decoded: mx.array) -> np.ndarray:
        if decoded.ndim == 5:
            decoded = decoded[:, :, 0]
        image = mx.clip(decoded[0] / 2 + 0.5, 0, 1)
        return (np.array(mx.transpose(image, (1, 2, 0)).astype(mx.float32)) * 255).round().astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure the VAE decode of an mflux model.")
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture")
    capture.add_argument("--model", required=True, choices=sorted(k for k in AVAILABLE_MODELS if any(k.startswith(p) for p in MODEL_CLASSES)))  # fmt: skip
    capture.add_argument("--model-path", default=None)
    capture.add_argument("-q", "--quantize", type=int, default=None)
    capture.add_argument("--prompts", required=True, type=Path)
    capture.add_argument("--out", required=True, type=Path)
    capture.add_argument("--width", type=int, default=1024)
    capture.add_argument("--height", type=int, default=1024)
    capture.add_argument("--steps", type=int, default=None)
    capture.add_argument("--seed", type=int, default=42)
    capture.add_argument("--skip", type=int, nargs="*", default=[], help="1-based prompt numbers to skip")
    decode = commands.add_parser("decode")
    decode.add_argument("--run-dir", required=True, type=Path)
    decode.add_argument("--modes", nargs="+", choices=list(MODES), default=list(MODES))
    decode.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    VAEDecodeBenchmark.capture(args) if args.command == "capture" else VAEDecodeBenchmark.decode(args)


if __name__ == "__main__":
    main()
