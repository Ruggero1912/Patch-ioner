"""
Utility functions for converting bboxes and traces to alpha masks for AlphaClip.
"""

import torch
import math


def bbox_to_alpha_mask(bbox, grid_size, patch_size, crop_dim):
    """
    Convert a single bounding box to an alpha mask for AlphaClip.
    
    Args:
        bbox: [x_min, y_min, w, h] format in original coordinates
        grid_size: Number of patches per side (e.g., 37 for 518/14)
        patch_size: Size of each patch in pixels
        crop_dim: Size of the cropped image
    
    Returns:
        alpha_mask: Binary mask of shape (grid_size, grid_size)
    """
    alpha_mask = torch.zeros((grid_size, grid_size))
    
    # Convert bbox to patch coordinates
    x_min, y_min, w, h = bbox
    x_max = x_min + w
    y_max = y_min + h
    
    # Scale to patch grid coordinates
    x1_patch = int(x_min // patch_size)
    y1_patch = int(y_min // patch_size)
    x2_patch = int(x_max // patch_size)
    y2_patch = int(y_max // patch_size)
    
    # Clamp to grid bounds
    x1_patch = max(0, min(x1_patch, grid_size - 1))
    y1_patch = max(0, min(y1_patch, grid_size - 1))
    x2_patch = max(0, min(x2_patch, grid_size))  # Allow up to grid_size for exclusive end
    y2_patch = max(0, min(y2_patch, grid_size))
    
    # Set the region to 1 (using slice notation for proper indexing)
    if x2_patch > x1_patch and y2_patch > y1_patch:
        alpha_mask[y1_patch:y2_patch, x1_patch:x2_patch] = 1.0
    
    return alpha_mask


def bboxes_to_alpha_mask(bboxes, grid_size, patch_size, crop_dim):
    """
    Convert multiple bboxes to a single OR-ed alpha mask.
    
    Args:
        bboxes: Tensor of bboxes in [x_min, y_min, w, h] format, shape [n_boxes, 4]
        grid_size: Number of patches per side
        patch_size: Size of each patch in pixels 
        crop_dim: Size of the cropped image
        
    Returns:
        alpha_mask: Binary mask of shape (grid_size, grid_size)
    """
    alpha_mask = torch.zeros((grid_size, grid_size))
    
    for bbox in bboxes:
        # Skip dummy boxes (negative values)
        if bbox.sum().item() < 0:
            continue
            
        bbox_mask = bbox_to_alpha_mask(bbox, grid_size, patch_size, crop_dim)
        alpha_mask = torch.logical_or(alpha_mask, bbox_mask).float()
    
    return alpha_mask


def trace_to_alpha_mask(trace, grid_size):
    """
    Convert a trace to an alpha mask using the existing map_traces_to_grid function.
    
    Args:
        trace: List of trace points with 'x' and 'y' coordinates (normalized 0-1)
        grid_size: Number of patches per side
        
    Returns:
        alpha_mask: Binary mask of shape (grid_size, grid_size) 
    """
    from src.bbox_utils import map_traces_to_grid
    
    alpha_mask = map_traces_to_grid(trace, grid_size)
    # Convert to binary (any value > 0 becomes 1)
    alpha_mask = (alpha_mask > 0).float()
    
    return alpha_mask


def traces_to_alpha_mask(traces, grid_size):
    """
    Convert multiple traces to a single OR-ed alpha mask.
    
    Args:
        traces: List of traces
        grid_size: Number of patches per side
        
    Returns:
        alpha_mask: Binary mask of shape (grid_size, grid_size)
    """
    alpha_mask = torch.zeros((grid_size, grid_size))
    
    for trace in traces:
        trace_mask = trace_to_alpha_mask(trace, grid_size)
        alpha_mask = torch.logical_or(alpha_mask, trace_mask).float()
    
    return alpha_mask
