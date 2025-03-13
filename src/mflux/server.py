from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
import base64
import os
import tempfile
import time
from typing import List, Optional
import io

# Import the Flux modules
from mflux import Config, Flux1, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler

app = FastAPI()

class GenerationRequest(BaseModel):
    prompt: str
    model: Optional[str] = "schnell"
    base_model: Optional[str] = None
    steps: Optional[int] = 2
    height: Optional[int] = 1024
    width: Optional[int] = 1024
    size: Optional[str] = None
    guidance: Optional[float] = 4.0
    image_strength: Optional[float] = None
    controlnet_strength: Optional[float] = None
    seed: Optional[int] = None
    quantize: Optional[int] = 8
    n: Optional[int] = 1
    lora_repo_id: Optional[str] = None
    lora_scale: Optional[float] = 0.7
    low_ram: Optional[bool] = False
    stepwise_output: Optional[bool] = False
    metadata: Optional[bool] = False
    local_path: Optional[str] = None
    response_format: Optional[str] = None  # Added for OpenAI compatibility

class Image(BaseModel):
    b64_json: str
    
class GenerationResponse(BaseModel):
    created: int
    data: List[Image]

@app.post("/v1/images/generations", response_model=GenerationResponse)
async def generate_images(request: GenerationRequest = Body(...)):
    # Now accepts a direct JSON body instead of form data
    
    try:
        # Determine the image path (if an image upload feature is needed, it would require a different approach)
        image_path = None
        
        # Setup for stepwise output if requested
        stepwise_dir = None
        if request.stepwise_output:
            stepwise_dir = tempfile.mkdtemp()

        # Map OpenAI model names to your internal models
        if request.model == 'dall-e-3':
            model = 'schnell'
        elif request.model == 'dall-e-2':
            model = 'dev'
        else:
            model = request.model or 'schnell'
        
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
                
        # 1. Load the model with LoRA support
        flux = Flux1(
            model_config=ModelConfig.from_name(
                model_name=model, 
                base_model=request.base_model
            ),
            quantize=request.quantize,
            local_path=request.local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )
        
        # 2. Register optional callbacks
        if stepwise_dir:
            handler = StepwiseHandler(flux=flux, output_dir=stepwise_dir)
            CallbackRegistry.register_before_loop(handler)
            CallbackRegistry.register_in_loop(handler)
            CallbackRegistry.register_interrupt(handler)

        memory_saver = None
        if request.low_ram:
            memory_saver = MemorySaver(flux)
            CallbackRegistry.register_before_loop(memory_saver)
            CallbackRegistry.register_in_loop(memory_saver)
            CallbackRegistry.register_after_loop(memory_saver)
        
        images = []
        
        try:
            for i in range(request.n or 1):
                # Use provided seed or generate a new one for each image
                current_seed = request.seed if request.seed is not None else int(time.time()) + i

                if request.size is not None:
                    width_str, height_str = request.size.lower().split('x')
                    height = int(height_str)
                    width = int(width_str)
                else:
                    height = request.height or 1024
                    width = request.width or 1024
                
                # 3. Generate the image
                generated_image = flux.generate_image(
                    seed=current_seed,
                    prompt=request.prompt,
                    config=Config(
                        num_inference_steps=request.steps or 2,
                        height=height,
                        width=width,
                        guidance=request.guidance or 4.0,
                        image_path=image_path,
                        image_strength=request.image_strength,
                        controlnet_strength=request.controlnet_strength
                    )
                )
                
                # Fix for the image saving issue
                buffer = io.BytesIO()

                # Always save to a temporary file first
                temp_img_path = f"{tempfile.mkdtemp()}/temp_img_{i}.png"

                # Save with or without metadata
                generated_image.save(
                    path=temp_img_path, 
                    export_json_metadata=request.metadata or False
                )

                # Read the file back
                with open(temp_img_path, "rb") as f:
                    buffer.write(f.read())

                # Clean up
                os.remove(temp_img_path)
                os.rmdir(os.path.dirname(temp_img_path))

                buffer.seek(0)
                b64_img = base64.b64encode(buffer.getvalue()).decode('utf-8')
                
                # Add the image to our response list
                images.append(Image(b64_json=b64_img))
                
        except StopImageGenerationException as stop_exc:
            # Handle graceful stopping if needed
            raise HTTPException(status_code=400, detail=f"Image generation stopped: {str(stop_exc)}")
        finally:
            # Print memory stats if using memory saver
            if memory_saver:
                print(memory_saver.memory_stats())
        
        # Clean up temporary files
        if image_path and os.path.exists(image_path):
            os.remove(image_path)
            os.rmdir(image_path.parent)
            
        if stepwise_dir:
            # Note: We're not deleting stepwise output as user might want to access it
            pass
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    
    return GenerationResponse(
        created=int(time.time()),
        data=images
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8800)