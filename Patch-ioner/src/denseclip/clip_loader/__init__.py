"""
DenseCLIP to CLIP Loader

A simple interface for loading DenseCLIP checkpoints as CLIP-like models
for text and image encoding.
"""

from .denseclip_loader import (
    DenseCLIPModel,
    load_denseclip_model,
    load_clip,
    load_config
)

__version__ = "1.0.0"
__all__ = [
    "DenseCLIPModel",
    "load_denseclip_model", 
    "load_clip",
    "load_config"
]
