import logging
import threading
import os
from typing import Dict, Optional
import tempfile
import datetime

from mflux import Flux1, ModelConfig

MODEL_KEEP_ALIVE_FOREVER = -1
MODEL_UNLOAD_IMMEDIATELY = 0

# Configure the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(console_handler)

class ModelManager:
    """Manages model loading, caching, and unloading across different API servers"""
    
    def __init__(self):
        self.model_cache: Dict[str, Flux1] = {}
        self.model_timers: Dict[str, threading.Timer] = {}
    
    def get_model(self, 
        model_name: str, 
        base_model: Optional[str] = None,
        quantize: int = 8,
        local_path: Optional[str] = None,
        lora_paths: Optional[list] = None,
        lora_scales: Optional[list] = None,
        keep_alive: datetime.timedelta = datetime.timedelta(minutes=0)
    ) -> Flux1:
        """
        Get a model from cache or load it if not available
        
        Args:
            model_name: Name of the model to load
            base_model: Base model to use
            quantize: Quantization level
            local_path: Path to local model
            lora_paths: List of LoRA paths or repo IDs
            lora_scales: List of LoRA scales
            keep_alive: Minutes to keep model in memory (MODEL_KEEP_ALIVE_FOREVER for forever, 
           MODEL_UNLOAD_IMMEDIATELY for unload after use)
            
        Returns:
            Loaded Flux1 model
        """
        # Create a unique model key
        model_key = repr([model_name, base_model, quantize, local_path, lora_paths])
        
        
        # Check if we have this model in cache
        if model_key in self.model_cache:
            logger.info(f"Using cached model: {model_key}")
            flux = self.model_cache[model_key]
            
            # Cancel any existing timer for this model
            # if model_key in self.model_timers and self.model_timers[model_key]:
            #     self.model_timers[model_key].cancel()

            if timer := self.model_timers.get(model_key):
                timer.cancel()
        else:
            # Load the model with LoRA support
            logger.info(f"Loading new model: {model_key}")
            flux = Flux1(
                model_config=ModelConfig.from_name(
                    model_name=model_name, 
                    base_model=base_model
                ),
                quantize=quantize,
                local_path=local_path,
                lora_paths=lora_paths,
                lora_scales=lora_scales,
            )
            
            # Store in cache
            self.model_cache[model_key] = flux
        
        # Handle the keep_alive parameter
        self._setup_model_timer(model_key, keep_alive)
        
        return flux, model_key
    
    def _setup_model_timer(self, model_key: str, keep_alive: datetime.timedelta):
        """Set up a timer to unload the model after the specified time"""
        if keep_alive != MODEL_KEEP_ALIVE_FOREVER:  # Replace -1 with constant
            if keep_alive == MODEL_UNLOAD_IMMEDIATELY:  # Replace 0 with constant
                # Will unload after generation completes
                pass
            else:
                # Set a timer to unload the model after keep_alive minutes
                timer = threading.Timer(
                    keep_alive.total_seconds(),
                    self.unload_model, 
                    args=[model_key]
                )
                timer.daemon = True
                self.model_timers[model_key] = timer
                timer.start()
                logger.info(f"Model {model_key} will be unloaded in {keep_alive} minutes if not used again")
        else:
            logger.info(f"Model {model_key} will remain loaded indefinitely")
    
    def unload_model(self, model_key: str):
        """Unload a model from cache when its timer expires"""
        if model_key in self.model_cache:
            logger.info(f"Unloading model: {model_key}")
            model = self.model_cache[model_key]
            
            # Check if Flux1 has any explicit cleanup methods
            if hasattr(model, 'unload') and callable(model.unload):
                model.unload()
            # Or if it has a close method
            elif hasattr(model, 'close') and callable(model.close):
                model.close()
            
            # Remove from cache
            del self.model_cache[model_key]
            self.model_timers.pop(model_key, None)
                
            # Force garbage collection
            import gc
            gc.collect()
    
    def unload_model_after_use(self, model_key: str):
        """Unload a model immediately after use"""
        logger.info(f"Unloading model immediately as requested: {model_key}")
        del self.model_cache[model_key]
        self.unload_model(model_key)

def create_temp_directory():
    """Create a temporary directory and return its path"""
    return tempfile.mkdtemp()


def cleanup_temp_file(file_path):
    """Clean up a temporary file and its parent directory if empty"""
    if file_path and os.path.exists(file_path):
        os.remove(file_path)
        parent_dir = os.path.dirname(file_path)
        if os.path.exists(parent_dir) and not os.listdir(parent_dir):
            os.rmdir(parent_dir)