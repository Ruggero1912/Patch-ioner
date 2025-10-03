"""
AlphaCLIP Standalone Package

A standalone version of AlphaCLIP that can be used independently.
"""

from .alphaclip_loader import AlphaCLIPLoader, load_alphaclip

# Version info
__version__ = "1.0.0"
__author__ = "AlphaCLIP Team"

# Make main classes available at package level
__all__ = ['AlphaCLIPLoader', 'load_alphaclip']
