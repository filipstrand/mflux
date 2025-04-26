import io
import json
import os
import subprocess
import time
from pathlib import Path

from fastmcp import (
    FastMCP,
    Image as MCPImage,
)
from PIL import Image as PILImage
from pydantic import BaseModel

from mflux import Config, Flux1, GeneratedImage, ModelConfig, StopImageGenerationException


class ImageGenerationArgs(BaseModel):
    model: str
    steps: int
    prompt: str
    width: int
    height: int
    guidance: float
    seed: int


# Create an MCP server
mcp = FastMCP("mflux")


@mcp.resource("config://app-version")
def get_app_version() -> str:
    """Returns the application version."""
    return GeneratedImage._get_version_from_toml()


@mcp.resource("dir://output")
def output() -> list[str]:
    """List the files in the user's desktop"""
    desktop = Path.home() / "mflux-mcp-output"
    return [str(f) for f in desktop.iterdir()]


def generate_client_output_image(
    image: PILImage, client_max_length=1024**2, format="png", size_down_step: float = 0.05
) -> MCPImage:
    """
    Generate a client-side output image optimized for transfer.

    This function takes a PIL Image and compresses it to PNG format, ensuring the resulting byte size
    is below the client_max_length threshold. If the initial compressed image exceeds this threshold,
    the function progressively resizes the image smaller until it fits within the limit.

    Args:
        image (PILImage): The PIL Image to be compressed
        client_max_length (int, optional): Maximum byte size allowed. Defaults to 1048576 (1MB).
        format (str, optional): Image format to use. Defaults to "png"
        size_down_step (float, optional): Percentage to reduce dimensions by in each iteration. Defaults to 0.05 (5%)

    Returns:
        bytes: The compressed image data as bytes
    """
    image.save(buffer := io.BytesIO(), compress_level=9, format=format, optimize=True)
    client_image: MCPImage = None
    client_image_data: str = None
    while client_image is None or len(client_image_data) > client_max_length:
        image = image.resize((int(image.width * (1 - size_down_step)), int(image.height * (1 - size_down_step))))
        image.save(buffer := io.BytesIO(), compress_level=9, format=format, optimize=True)
        client_image = MCPImage(data=buffer.getvalue(), format=format)
        client_image_data = client_image.to_image_content().data
    return client_image


def generate_image(args: ImageGenerationArgs) -> MCPImage:
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
        return generate_client_output_image(generated_image.image)
    except StopImageGenerationException as stop_exc:
        print(stop_exc)


@mcp.tool(description="Generate an image using the Flux.1-Dev model")
def dev_model_generate_image(
    prompt: str, steps: int, width: int, height: int, guidance: float, seed: int = None
) -> MCPImage:
    seed = seed or int(time.time())
    args = ImageGenerationArgs(
        model="dev", steps=steps, prompt=prompt, width=width, height=height, guidance=guidance, seed=seed
    )
    return generate_image(args)


@mcp.tool(description="Generate an image using the Flux.1-Schnell model")
def schnell_model_generate_image(
    prompt: str, steps: int, width: int, height: int, guidance: float, seed: int = None
) -> MCPImage:
    seed = seed or int(time.time())
    args = ImageGenerationArgs(
        model="schnell", steps=steps, prompt=prompt, width=width, height=height, guidance=guidance, seed=seed
    )
    return generate_image(args)


@mcp.tool(description="generate a test image for mflux integration")
def flux1_test_image() -> MCPImage:
    """Returns a test image from a static file path."""
    image_path = Path("~/Desktop/test-image.png").expanduser()
    return generate_client_output_image(PILImage.open(image_path))


class SystemInfo(BaseModel):
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
    return f"Hello from mflux, {name}. You are running Flux.1 image generator on {get_system_info()}!"


def main():
    mcp.run()


if __name__ == "__main__":
    main()
