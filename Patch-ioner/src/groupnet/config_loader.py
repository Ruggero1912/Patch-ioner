"""
Configuration loader for GroupNet models.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a GroupNet configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
    
    Returns:
        Dictionary containing the configuration
    
    Raises:
        FileNotFoundError: If the config file doesn't exist
        yaml.YAMLError: If the YAML file is malformed
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    if 'embed_dim' not in config:
        raise ValueError("Configuration must specify 'embed_dim'")
    
    return config


def get_default_config(model_type: str = 'attention', embed_dim: int = 768) -> Dict[str, Any]:
    """
    Get default configuration for a specific model type.
    
    Args:
        model_type: Type of model ('attention', 'transformer', or 'set_transformer')
        embed_dim: Embedding dimension
    
    Returns:
        Default configuration dictionary
    """
    base_config = {
        'model_type': model_type,
        'embed_dim': embed_dim,
        'num_heads': 8,
        'dropout': 0.1,
    }
    
    if model_type == 'attention':
        return {
            **base_config,
            'use_cls_token': True,
        }
    elif model_type == 'transformer':
        return {
            **base_config,
            'num_layers': 3,
            'dim_feedforward': 2048,
            'use_cls_token': True,
            'positional_encoding': 'none',
            'max_seq_len': 512,
        }
    elif model_type == 'set_transformer':
        return {
            **base_config,
            'num_layers': 3,
            'num_inducing_points': 32,
            'num_seed_vectors': 1,
            'dim_feedforward': 2048,
            'use_isab': True,
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save a configuration dictionary to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        save_path: Path where to save the YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
