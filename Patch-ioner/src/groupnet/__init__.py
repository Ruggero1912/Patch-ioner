"""
GroupNet: Neural network models for aggregating sets of patch embeddings.

This module provides flexible aggregation models that can handle variable-length
sequences of patch embeddings and produce a single aggregated feature vector.

Available models:
- AttentionLayer: Single-layer attention-based aggregation
- TransformerAggregator: Multi-layer transformer encoder
- SetTransformer: Set-based aggregation with inducing points

Usage:
    from src.groupnet import GroupNet
    from src.groupnet.config_loader import load_config
    
    # Load configuration
    config = load_config('src/groupnet/configs/transformer.yaml')
    
    # Create model
    model = GroupNet(config)
    
    # Forward pass
    patches = torch.randn(batch_size, num_patches, embed_dim)
    aggregated = model(patches)  # (batch_size, embed_dim)
"""

from .model import (
    AttentionLayer,
    TransformerAggregator,
    SetTransformer,
    GroupNet,
)

from .config_loader import (
    load_config,
    get_default_config,
    save_config,
)

from .utils import (
    extract_bbox_patches,
    extract_trace_patches,
    aggregate_with_groupnet,
)

from .entrypoint import (
    load_groupnet_model,
)

__all__ = [
    'AttentionLayer',
    'TransformerAggregator',
    'SetTransformer',
    'GroupNet',
    'load_config',
    'get_default_config',
    'save_config',
    'extract_bbox_patches',
    'extract_trace_patches',
    'aggregate_with_groupnet',
    'load_groupnet_model',
]
