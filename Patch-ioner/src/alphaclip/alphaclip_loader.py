"""
AlphaCLIP Standalone Loader

This module provides a simple interface to load and use AlphaCLIP models.
It exposes the core functionality of AlphaCLIP in a standalone package.

Usage:
    from alphaclip_loader import AlphaCLIPLoader
    
    # Initialize the loader
    loader = AlphaCLIPLoader()
    
    # Load a model
    model, preprocess = loader.load_model("ViT-B/16")
    
    # Tokenize text
    tokens = loader.tokenize("A photo of a cat")
    
    # Get available models
    models = loader.available_models()
"""

import os
import sys
from typing import Union, List, Tuple, Optional

# Check for critical dependencies
missing_deps = []
try:
    import torch
except ImportError:
    missing_deps.append("torch")

try:
    from PIL import Image
except ImportError:
    missing_deps.append("Pillow")

if missing_deps:
    raise ImportError(f"Missing required dependencies: {', '.join(missing_deps)}. "
                     f"Please install them with: pip install {' '.join(missing_deps)}")

# Add the alpha_clip directory to the path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_alpha_clip_dir = os.path.join(_current_dir, 'alpha_clip')
if _alpha_clip_dir not in sys.path:
    sys.path.insert(0, _alpha_clip_dir)

# Import the alpha_clip modules
try:
    #import .alpha_clip
    from .alpha_clip import available_models, load, tokenize
except ImportError as e:
    raise ImportError(f"Failed to import alpha_clip modules: {e}. Please ensure all dependencies are installed.")


class AlphaCLIPLoader:
    """
    A convenience wrapper for AlphaCLIP functionality.
    
    This class provides a clean interface to load AlphaCLIP models and 
    perform text tokenization.
    """
    
    def __init__(self, default_device: Optional[str] = None):
        """
        Initialize the AlphaCLIP loader.
        
        Args:
            default_device: Default device to load models on. If None, will use
                          CUDA if available, otherwise CPU.
        """
        if default_device is None:
            self.default_device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.default_device = default_device
    
    def available_models(self) -> List[str]:
        """
        Get list of available AlphaCLIP model names.
        
        Returns:
            List of model names that can be used with load_model()
        """
        return available_models()
    
    def load_model(
        self, 
        name: str,
        alpha_vision_ckpt_pth: str = "None",
        device: Optional[Union[str, torch.device]] = None,
        jit: bool = False,
        download_root: Optional[str] = None,
        lora_adapt: bool = False,
        rank: int = 16
    ) -> Tuple[torch.nn.Module, callable]:
        """
        Load an AlphaCLIP model.
        
        Args:
            name: Model name (e.g., "ViT-B/16") or path to checkpoint
            alpha_vision_ckpt_pth: Path to additional vision checkpoint
            device: Device to load model on (defaults to self.default_device)
            jit: Whether to load JIT optimized model
            download_root: Directory to download models to
            lora_adapt: Whether to use LoRA adaptation
            rank: LoRA rank if lora_adapt is True
            
        Returns:
            Tuple of (model, preprocess_function)
        """
        if device is None:
            device = self.default_device
            
        return load(
            name=name,
            alpha_vision_ckpt_pth=alpha_vision_ckpt_pth,
            device=device,
            jit=jit,
            download_root=download_root,
            lora_adapt=lora_adapt,
            rank=rank
        )
    
    def tokenize(
        self, 
        texts: Union[str, List[str]], 
        context_length: int = 77, 
        truncate: bool = True
    ) -> torch.Tensor:
        """
        Tokenize text for use with AlphaCLIP models.
        
        Args:
            texts: String or list of strings to tokenize
            context_length: Maximum token length (default 77)
            truncate: Whether to truncate long texts
            
        Returns:
            Tensor of tokenized text
        """
        return tokenize(texts, context_length, truncate)
    
    def encode_text(self, model: torch.nn.Module, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Convenience method to tokenize and encode text.
        
        Args:
            model: Loaded AlphaCLIP model
            texts: Text(s) to encode
            
        Returns:
            Text embeddings tensor
        """
        tokens = self.tokenize(texts)
        if hasattr(model, 'token_embedding'):
            # Move tokens to same device as model
            device = next(model.parameters()).device
            tokens = tokens.to(device)
        
        with torch.no_grad():
            text_features = model.encode_text(tokens)
            
        return text_features
    
    def encode_image(self, model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
        """
        Convenience method to encode images.
        
        Args:
            model: Loaded AlphaCLIP model
            images: Preprocessed image tensor
            
        Returns:
            Image embeddings tensor
        """
        with torch.no_grad():
            image_features = model.encode_image(images)
            
        return image_features
    
    def get_similarity(self, text_features: torch.Tensor, image_features: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity between text and image features.
        
        Args:
            text_features: Text embedding tensor
            image_features: Image embedding tensor
            
        Returns:
            Similarity scores tensor
        """
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Compute similarity
        similarity = (text_features @ image_features.T)
        return similarity


# Convenience function for quick model loading
def load_alphaclip(
    model_name: str = "ViT-B/16",
    device: Optional[str] = None,
    alpha_vision_ckpt_pth: str = "None",
    download_root = '/raid/datasets/models_weights/alphaclip',
    **kwargs
) -> Tuple[AlphaCLIPLoader, torch.nn.Module, callable]:
    """
    Quick function to load AlphaCLIP with a loader instance.
    
    Args:
        model_name: Name of the model to load
        device: Device to use
        **kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (loader, model, preprocess_function)
    """
    loader = AlphaCLIPLoader(default_device=device)
    model, preprocess = loader.load_model(model_name, **kwargs)
    return loader, model, preprocess


# Make key functions available at module level
__all__ = [
    'AlphaCLIPLoader',
    'load_alphaclip',
    'available_models',
    'load',
    'tokenize'
]
