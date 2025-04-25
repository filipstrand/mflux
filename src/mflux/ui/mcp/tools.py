import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass

from fastmcp import (
    FastMCP,
    Image as MCPImage,
)

from mflux import Config, Flux1, ModelConfig, StopImageGenerationException

# Create an MCP server
mcp = FastMCP("mflux")


def generate_image(args: argparse.Namespace) -> MCPImage:
    flux = Flux1(model_config=ModelConfig.from_name(model_name=args.model))

    try:
        generated_image = flux.generate_image(
            seed=args.seed,
            prompt=args.prompt,
            config=Config(
                num_inference_steps=args.steps,
                height=args.height,
                width=args.width,
                guidance=args.guidance,
            ),
        )
        return MCPImage(generated_image.image.tobytes(), format="png")
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


@mcp.tool()
def dev_model_generate_image(
    prompt: str, steps: int, width: int, height: int, guidance: float, seed: int = None
) -> MCPImage:
    seed = seed or int(time.time())
    args = argparse.Namespace(
        model="schnell", steps=steps, prompt=prompt, width=width, height=height, guidance=guidance, seed=seed
    )
    return generate_image(args)


@mcp.tool()
def schnell_model_generate_image(
    prompt: str, steps: int, width: int, height: int, guidance: float, seed: int = None
) -> MCPImage:
    seed = seed or int(time.time())
    args = argparse.Namespace(
        model="schnell", steps=steps, prompt=prompt, width=width, height=height, guidance=guidance, seed=seed
    )
    return generate_image(args)


@dataclass
class SystemInfo:
    chipset_name: str
    cpu_count: int
    gpu_count: int


# Get number of logical CPUs
def get_system_info() -> SystemInfo:
    try:
        info = json.loads(subprocess.check_output(["system_profiler", "-json", "SPDisplaysDataType"], text=True))[
            "SPDisplaysDataType"
        ][0]
    except (IndexError, KeyError) as e:
        return f"Error parsing system profile: {e}"
    return SystemInfo(chipset_name=info["_name"], cpu_count=os.cpu_count(), gpu_count=info["sppci_cores"])


# Add a dynamic greeting resource
@mcp.resource("greeting://{name}")
def get_greeting(name: str) -> str:
    """Get a personalized greeting"""
    return f"Hello from mflux, {name}. You are running Flux image generator on {get_system_info()}!"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
