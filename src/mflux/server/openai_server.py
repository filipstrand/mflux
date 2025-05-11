from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import base64
import os
import time
import io
from typing import List, Optional
import logging
import http.client
import argparse
import uvicorn
import random

from mflux.ui.defaults import WIDTH, HEIGHT
from mflux import Config, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from common import ModelManager, create_temp_directory, cleanup_temp_file, MODEL_KEEP_ALIVE_FOREVER, MODEL_UNLOAD_IMMEDIATELY

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

app = FastAPI()

# Initialize the model manager
model_manager = ModelManager()

class GenerationRequest(BaseModel):
    prompt: str
    model: str = "schnell"
    base_model: Optional[str] = None
    height: Optional[int] = HEIGHT
    width: Optional[int] = WIDTH
    size: Optional[str] = None
    quality: Optional[str] = None
    guidance: float = 4.0
    image_strength: Optional[float] = None
    controlnet_strength: Optional[float] = None
    seed: Optional[int] = None
    quantize: int = 4
    n: int = 1
    steps: int = 4
    lora_repo_id: Optional[str] = None
    lora_scale: float = 0.7
    low_ram: bool = False
    stepwise_output: bool = False
    metadata: bool = False
    local_path: Optional[str] = None
    response_format: Optional[str] = None
    keep_alive: int = MODEL_UNLOAD_IMMEDIATELY

class Image(BaseModel):
    b64_json: str
    
class GenerationResponse(BaseModel):
    created: int
    data: List[Image]

@app.post("/v1/images/generations", response_model=GenerationResponse)
async def generate_images(request: GenerationRequest = Body(...)):
    try:
        # Determine the image path (if an image upload feature is needed, it would require a different approach)
        image_path = None
        
        # Setup for stepwise output if requested
        stepwise_dir = None
        if request.stepwise_output:
            stepwise_dir = create_temp_directory()

        # Use the model name directly without OpenAI mapping
        model = request.model or "schnell"
        
        # For OpenAI API compatibility, we can validate sizes
        if model in ["dall-e-3", "schnell"]:
            valid_sizes = ['1024x1024', '1792x1024', '1024x1792']
            if model == "dall-e-3":
                model = "schnell"
        elif model in ['dall-e-2', 'dev']:
            valid_sizes = ['256x256', '512x512', '1024x1024']
            if model == "dall-e-2":
                model = "dev"
        else:
            # For custom models, we'll be more flexible with sizes
            valid_sizes = None
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=http.client.INTERNAL_SERVER_ERROR, detail=f"Generation failed: {e}")
        
    try:
        # Handle size parameter
        if request.size is not None:
            # Check if size is valid for the selected model
            if valid_sizes and request.size not in valid_sizes:
                raise HTTPException(
                    status_code=http.client.BAD_REQUEST, 
                    detail=f"Invalid size '{request.size}' for model '{model}'. Valid sizes: {valid_sizes}"
                )
            width_str, _, height_str = request.size.lower().partition('x')
            height = int(height_str)
            width = int(width_str)
        else:
            # Use default dimensions or provided height/width
            height = request.height or HEIGHT
            width = request.width or WIDTH
            
            # For OpenAI compatibility, check if dimensions are valid
            if valid_sizes and f"{width}x{height}" not in valid_sizes and f"{height}x{width}" not in valid_sizes:
                logger.warning(
                    f"Dimensions {width}x{height} may not be standard for model '{model}'. "
                    f"Standard sizes: {valid_sizes}"
                )
        
        # Prepare LoRA parameters
        lora_paths = None
        lora_scales = None

        if request.lora_repo_id:
            # Check if it's a local path or a repo ID
            if os.path.exists(request.lora_repo_id):
                lora_paths = [request.lora_repo_id]
            else:
                # Assume it's a HuggingFace repo ID
                lora_paths = [request.lora_repo_id]
            
            if request.lora_scale is not None:
                lora_scales = [request.lora_scale]
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=http.client.INTERNAL_SERVER_ERROR, detail=f"Generation failed: {e}")
        
    try:
        # Get the model using the model manager
        flux, model_key = model_manager.get_model(
            model_name=model,
            base_model=request.base_model,
            quantize=request.quantize,
            local_path=request.local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            keep_alive=request.keep_alive
        )
        
        registered_callbacks = []

        if stepwise_dir:
            handler = StepwiseHandler(flux=flux, output_dir=stepwise_dir)
            CallbackRegistry.register_before_loop(handler)
            registered_callbacks.append(("before_loop", handler))
            CallbackRegistry.register_in_loop(handler)
            registered_callbacks.append(("in_loop", handler))
            CallbackRegistry.register_interrupt(handler)
            registered_callbacks.append(("interrupt", handler))

        memory_saver = None
        if request.low_ram:
            memory_saver = MemorySaver(flux)
            CallbackRegistry.register_before_loop(memory_saver)
            registered_callbacks.append(("before_loop", memory_saver))
            CallbackRegistry.register_in_loop(memory_saver)
            registered_callbacks.append(("in_loop", memory_saver))
            CallbackRegistry.register_after_loop(memory_saver)
            registered_callbacks.append(("after_loop", memory_saver))
        
        images = []
        for i in range(request.n):
            if request.seed is not None:
                current_seed = request.seed + i
            else:
                current_seed = random.getrandbits(32)
            
            config = Config(
                num_inference_steps=request.steps,
                height=height,
                width=width,
                guidance=request.guidance,
                image_path=image_path,
                image_strength=request.image_strength,
                controlnet_strength=request.controlnet_strength
            )
            
            # Generate the image
            generated_image = flux.generate_image(
                seed=current_seed,
                prompt=request.prompt,
                config=config
            )
            
            # Save image to buffer via temporary file
            buffer = io.BytesIO()
            temp_img_path = f"{create_temp_directory()}/temp_img_{i}.png"

            # Save with or without metadata
            generated_image.save(
                path=temp_img_path, 
                export_json_metadata=request.metadata or False
            )

            # Read the file back
            with open(temp_img_path, "rb") as f:
                buffer.write(f.read())

            # Clean up temp file
            cleanup_temp_file(temp_img_path)

            # Encode image to base64
            buffer.seek(0)
            b64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Add the image to our response list
            images.append(Image(b64_json=b64_img))

        return GenerationResponse(
            created=int(time.time()),
            data=images
        )
                
    except StopImageGenerationException as stop_exc:
        raise HTTPException(status_code=http.client.BAD_REQUEST, detail=f"Image generation stopped: {stop_exc}")
    
    finally:
        # Clean up all registered callbacks
        for callback_type, callback in registered_callbacks:
            if callback_type == "before_loop":
                CallbackRegistry.unregister_before_loop(callback)
            elif callback_type == "in_loop":
                CallbackRegistry.unregister_in_loop(callback)
            elif callback_type == "after_loop":
                CallbackRegistry.unregister_after_loop(callback)
            elif callback_type == "interrupt":
                CallbackRegistry.unregister_interrupt(callback)
        
        # Print memory stats if using memory saver
        if memory_saver:
            logger.info(memory_saver.memory_stats())
        
        # If keep_alive is 0, unload the model immediately
        if request.keep_alive == 0:
            flux = None
            del flux
            model_manager.unload_model(model_key)
        
        # Clean up temporary files
        if image_path:
            cleanup_temp_file(image_path)


def run_server(host="0.0.0.0", port=8800):
    """
    Run the OpenAI-compatible server.
    This function is called by the CLI module.
    """
    logger.info(f"Starting OpenAI-compatible server on {host}:{port}")
    
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mflux OpenAI DALL·E-compatible API Server configuration')
    parser.add_argument('--host', default='0.0.0.0', type=str,
                        help='Host address to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', default=8800, type=int,
                        help='Port to listen on (default: 8800)')
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port)