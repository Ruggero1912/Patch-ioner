"""
GroupNet entrypoint for integration with Patchioner model.

This module provides a unified interface for loading GroupNet models,
similar to viecap, clipcap, and other model integrations.
"""

import torch
import os
from typing import Dict, Any, Optional

from .model import GroupNet
from .config_loader import load_config


def load_groupnet_model(
    groupnet_config: Dict[str, Any],
    device: str = 'cpu',
    hf_repo_id: Optional[str] = None
) -> GroupNet:
    """
    Load a GroupNet model with configuration and optional weights.
    
    This function follows the same pattern as viecap, clipcap, and other model loaders:
    1. Load configuration from file or dict
    2. Create model instance
    3. Load weights from file or HuggingFace as fallback
    
    Args:
        groupnet_config: Configuration dictionary containing:
            - config: Path to GroupNet config YAML (required)
            - weights_path: Path to checkpoint file (optional)
            - embed_dim: Override embedding dimension (optional)
        device: Device to load model on ('cpu', 'cuda', etc.)
        hf_repo_id: HuggingFace repository ID for fallback weight loading
    
    Returns:
        GroupNet model instance loaded and ready for inference
    
    Example:
        >>> config = {
        ...     'config': 'groupnet/configs/transformer.yaml',
        ...     'weights_path': 'path/to/weights.pth'
        ... }
        >>> model = load_groupnet_model(config, device='cuda')
    """
    
    # Extract configuration path
    config_path = groupnet_config.get('config', None)
    if config_path is None:
        raise ValueError("GroupNet configuration must specify 'config' path")
    
    # Load GroupNet configuration from YAML
    if not os.path.isabs(config_path):
        # If relative path, make it relative to the groupnet directory
        config_path = os.path.join(os.path.dirname(__file__), 'configs', config_path)
    
    print(f"Loading GroupNet configuration from: {config_path}")
    model_config = load_config(config_path)
    
    # Override embed_dim if specified in groupnet_config
    if 'embed_dim' in groupnet_config:
        model_config['embed_dim'] = groupnet_config['embed_dim']
        print(f"Overriding GroupNet embed_dim to {groupnet_config['embed_dim']}")
    
    # Create GroupNet model instance
    print(f"Creating GroupNet model (type: {model_config.get('model_type', 'unknown')})")
    model = GroupNet(model_config)
    
    # Load weights if specified
    weights_path = groupnet_config.get('weights_path', None)
    
    if weights_path is not None:
        # Try loading from local path first
        success = _load_weights_from_file(model, weights_path, device)
        
        # Fallback to HuggingFace if local loading failed and hf_repo_id is provided
        if not success and hf_repo_id is not None:
            print(f"Local weights not found, trying HuggingFace fallback...")
            success = _load_weights_from_huggingface(model, weights_path, hf_repo_id, device)
        
        if not success:
            print("Warning: Could not load GroupNet weights, using random initialization")
    else:
        print("No weights_path specified, using random initialization")
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    print(f"GroupNet loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    return model


def _load_weights_from_file(
    model: GroupNet,
    weights_path: str,
    device: str
) -> bool:
    """
    Load weights from a local file.
    
    Args:
        model: GroupNet model instance
        weights_path: Path to checkpoint file
        device: Device for loading
    
    Returns:
        True if successful, False otherwise
    """
    # Handle relative paths
    if not os.path.isabs(weights_path):
        # Try relative to groupnet directory
        full_path = os.path.join(os.path.dirname(__file__), '..', weights_path)
        if not os.path.exists(full_path):
            # Try relative to current working directory
            full_path = weights_path
    else:
        full_path = weights_path
    
    if not os.path.exists(full_path):
        print(f"Weights file not found: {full_path}")
        return False
    
    try:
        print(f"Loading GroupNet weights from: {full_path}")
        checkpoint = torch.load(full_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print("GroupNet weights loaded successfully from local file")
        return True
        
    except Exception as e:
        print(f"Error loading weights from {full_path}: {e}")
        return False


def _load_weights_from_huggingface(
    model: GroupNet,
    weights_filename: str,
    hf_repo_id: str,
    device: str
) -> bool:
    """
    Load weights from HuggingFace as fallback.
    
    Args:
        model: GroupNet model instance
        weights_filename: Filename of weights in HF repo
        hf_repo_id: HuggingFace repository ID
        device: Device for loading
    
    Returns:
        True if successful, False otherwise
    """
    try:
        from ..hf_utils import get_model_path_with_hf_fallback
        
        print(f"Attempting to load GroupNet weights from HuggingFace repo: {hf_repo_id}")
        
        # Extract just the filename if full path was provided
        if os.path.sep in weights_filename or '/' in weights_filename:
            weights_filename = os.path.basename(weights_filename)
        
        # Download from HuggingFace
        weights_path = get_model_path_with_hf_fallback(
            model_name=hf_repo_id,
            hf_repo_id=hf_repo_id,
            filename=weights_filename
        )
        
        if weights_path is None or not os.path.exists(weights_path):
            print(f"Could not download weights from HuggingFace")
            return False
        
        # Load the downloaded weights
        checkpoint = torch.load(weights_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        print(f"GroupNet weights loaded successfully from HuggingFace")
        return True
        
    except ImportError:
        print("HuggingFace utilities not available")
        return False
    except Exception as e:
        print(f"Error loading weights from HuggingFace: {e}")
        return False
