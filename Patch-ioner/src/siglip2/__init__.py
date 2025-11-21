"""
SigLIP2 Integration Module

This module provides integration of SigLIP2 (the newer version of SigLIP) 
as a visual backbone for the Patchioner framework.

The module uses locally stored transformers.models.siglip2 code to avoid
requiring a transformers library update.
"""

from .loader import (
    load_siglip2,
    load_siglip2_processor,
    get_siglip2_tokenizer,
    load_siglip2_config
)

__all__ = [
    'load_siglip2',
    'load_siglip2_processor', 
    'get_siglip2_tokenizer',
    'load_siglip2_config'
]
