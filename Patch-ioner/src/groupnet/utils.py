"""
Utility functions for extracting and preparing patches for GroupNet aggregation.

These functions help integrate GroupNet with the existing Patchioner implementation.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


def extract_bbox_patches(
    patch_embeddings: torch.Tensor,
    bboxes: torch.Tensor,
    patch_size: int = 14,
    return_mask: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract patch embeddings within bounding boxes for GroupNet aggregation.
    
    Unlike the original extract_bboxes_feats which aggregates patches using mean/gaussian,
    this function returns the raw patches for neural aggregation.
    
    Args:
        patch_embeddings: Patch embeddings of shape (batch_size, num_patches, embed_dim)
                         where num_patches = grid_size^2
        bboxes: Bounding boxes of shape (batch_size, num_boxes, 4) in [x, y, w, h] format
                Values should be in pixel coordinates (will be converted to patch coordinates)
        patch_size: Size of patches in pixels (e.g., 14 for ViT-B/14)
        return_mask: If True, returns mask for padded patches in variable-length sequences
    
    Returns:
        bbox_patches: Tensor of shape (batch_size, num_boxes, max_patches_per_bbox, embed_dim)
                     Padded to max_patches_per_bbox across all boxes
        mask: Optional tensor of shape (batch_size, num_boxes, max_patches_per_bbox)
              True indicates padded positions, False indicates valid patches
              Only returned if return_mask=True
    
    Example:
        >>> patches = torch.randn(4, 256, 768)  # 4 images, 16x16 grid, 768-dim
        >>> bboxes = torch.tensor([[[56, 56, 112, 112], [0, 0, 56, 56]]])  # 2 boxes
        >>> bbox_patches, mask = extract_bbox_patches(patches, bboxes, patch_size=14)
        >>> print(bbox_patches.shape)  # (4, 2, max_patches, 768)
    """
    batch_size = patch_embeddings.shape[0]
    num_boxes = bboxes.shape[1]
    embed_dim = patch_embeddings.shape[-1]
    grid_size = int(patch_embeddings.shape[1] ** 0.5)
    device = patch_embeddings.device
    
    # Convert bboxes from pixel coordinates to patch coordinates
    bboxes = bboxes.clone().float() / patch_size
    bboxes = bboxes.long()
    
    # Reshape patches to spatial grid
    patch_embeddings = patch_embeddings.view(batch_size, grid_size, grid_size, embed_dim)
    
    # Extract boxes and find max patches needed
    x1, y1, w, h = bboxes.unbind(-1)
    x2 = x1 + w
    y2 = y1 + h
    
    # Find maximum number of patches in any bbox
    max_patches_per_bbox = 0
    all_patches = []
    all_lengths = []
    
    for i in range(batch_size):
        batch_patches = []
        batch_lengths = []
        
        for j in range(num_boxes):
            # Check for dummy/invalid boxes (negative values)
            if bboxes[i, j].min() < 0:
                # Dummy box - add placeholder
                batch_patches.append([])
                batch_lengths.append(0)
                continue
            
            # Extract region for this box
            y_start = max(0, y1[i, j].item())
            y_end = min(grid_size, y2[i, j].item() + 1)
            x_start = max(0, x1[i, j].item())
            x_end = min(grid_size, x2[i, j].item() + 1)
            
            region_patches = patch_embeddings[i, y_start:y_end, x_start:x_end, :]
            
            # Flatten spatial dimensions
            num_patches = region_patches.shape[0] * region_patches.shape[1]
            region_patches = region_patches.reshape(num_patches, embed_dim)
            
            batch_patches.append(region_patches)
            batch_lengths.append(num_patches)
            max_patches_per_bbox = max(max_patches_per_bbox, num_patches)
        
        all_patches.append(batch_patches)
        all_lengths.append(batch_lengths)
    
    # Pad all boxes to max_patches_per_bbox
    bbox_patches = torch.zeros(
        batch_size, num_boxes, max_patches_per_bbox, embed_dim,
        device=device, dtype=patch_embeddings.dtype
    )
    
    if return_mask:
        mask = torch.ones(batch_size, num_boxes, max_patches_per_bbox, dtype=torch.bool, device=device)
    else:
        mask = None
    
    for i in range(batch_size):
        for j in range(num_boxes):
            if all_lengths[i][j] > 0:
                patches = all_patches[i][j]
                length = all_lengths[i][j]
                bbox_patches[i, j, :length] = patches
                if return_mask:
                    mask[i, j, :length] = False
    
    if return_mask:
        return bbox_patches, mask
    else:
        return bbox_patches


def extract_trace_patches(
    patch_embeddings: torch.Tensor,
    traces: torch.Tensor,
    patch_size: int = 14,
    return_mask: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract patch embeddings along traces/scribbles for GroupNet aggregation.
    
    Args:
        patch_embeddings: Patch embeddings of shape (batch_size, num_patches, embed_dim)
        traces: Binary trace masks of shape (batch_size, num_traces, H, W)
                where H and W are image dimensions
        patch_size: Size of patches in pixels
        return_mask: If True, returns mask for padded patches
    
    Returns:
        trace_patches: Tensor of shape (batch_size, num_traces, max_patches_per_trace, embed_dim)
        mask: Optional mask tensor
    """
    batch_size = patch_embeddings.shape[0]
    num_traces = traces.shape[1]
    embed_dim = patch_embeddings.shape[-1]
    grid_size = int(patch_embeddings.shape[1] ** 0.5)
    device = patch_embeddings.device
    
    # Reshape patches to spatial grid
    patch_embeddings = patch_embeddings.view(batch_size, grid_size, grid_size, embed_dim)
    
    # Downsample traces to patch resolution
    trace_masks = F.interpolate(
        traces.float(),
        size=(grid_size, grid_size),
        mode='nearest'
    )  # (batch_size, num_traces, grid_size, grid_size)
    
    # Find maximum number of patches in any trace
    max_patches_per_trace = 0
    all_patches = []
    all_lengths = []
    
    for i in range(batch_size):
        batch_patches = []
        batch_lengths = []
        
        for j in range(num_traces):
            # Get mask for this trace
            mask = trace_masks[i, j] > 0.5
            
            # Extract patches where mask is True
            if mask.sum() == 0:
                # Empty trace
                batch_patches.append([])
                batch_lengths.append(0)
                continue
            
            # Get patches at masked locations
            trace_patch = patch_embeddings[i][mask]  # (num_masked_patches, embed_dim)
            
            num_patches = trace_patch.shape[0]
            batch_patches.append(trace_patch)
            batch_lengths.append(num_patches)
            max_patches_per_trace = max(max_patches_per_trace, num_patches)
        
        all_patches.append(batch_patches)
        all_lengths.append(batch_lengths)
    
    # Pad all traces to max_patches_per_trace
    trace_patches = torch.zeros(
        batch_size, num_traces, max_patches_per_trace, embed_dim,
        device=device, dtype=patch_embeddings.dtype
    )
    
    if return_mask:
        mask = torch.ones(batch_size, num_traces, max_patches_per_trace, dtype=torch.bool, device=device)
    else:
        mask = None
    
    for i in range(batch_size):
        for j in range(num_traces):
            if all_lengths[i][j] > 0:
                patches = all_patches[i][j]
                length = all_lengths[i][j]
                trace_patches[i, j, :length] = patches
                if return_mask:
                    mask[i, j, :length] = False
    
    if return_mask:
        return trace_patches, mask
    else:
        return trace_patches


def aggregate_with_groupnet(
    model,
    patch_embeddings: torch.Tensor,
    bboxes: Optional[torch.Tensor] = None,
    traces: Optional[torch.Tensor] = None,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    Convenience function to extract and aggregate patches using GroupNet.
    
    This is a high-level wrapper that handles both bbox and trace aggregation.
    
    Args:
        model: GroupNet model instance
        patch_embeddings: Patch embeddings of shape (batch_size, num_patches, embed_dim)
        bboxes: Optional bounding boxes of shape (batch_size, num_boxes, 4)
        traces: Optional trace masks of shape (batch_size, num_traces, H, W)
        patch_size: Size of patches in pixels
    
    Returns:
        Aggregated features of shape:
        - (batch_size, num_boxes, embed_dim) if bboxes provided
        - (batch_size, num_traces, embed_dim) if traces provided
    
    Example:
        >>> from src.groupnet import GroupNet
        >>> from src.groupnet.config_loader import get_default_config
        >>> 
        >>> config = get_default_config('transformer', embed_dim=768)
        >>> groupnet = GroupNet(config)
        >>> 
        >>> patches = torch.randn(4, 256, 768)
        >>> bboxes = torch.randint(0, 224, (4, 10, 4))
        >>> 
        >>> aggregated = aggregate_with_groupnet(groupnet, patches, bboxes=bboxes)
        >>> print(aggregated.shape)  # (4, 10, 768)
    """
    if bboxes is not None:
        # Extract bbox patches
        bbox_patches, mask = extract_bbox_patches(
            patch_embeddings, bboxes, patch_size, return_mask=True
        )
        
        # Reshape for batch processing
        batch_size, num_boxes, max_patches, embed_dim = bbox_patches.shape
        bbox_patches_flat = bbox_patches.view(batch_size * num_boxes, max_patches, embed_dim)
        mask_flat = mask.view(batch_size * num_boxes, max_patches)
        
        # Aggregate with GroupNet
        aggregated = model(bbox_patches_flat, mask=mask_flat)
        
        # Reshape back
        return aggregated.view(batch_size, num_boxes, embed_dim)
    
    elif traces is not None:
        # Extract trace patches
        trace_patches, mask = extract_trace_patches(
            patch_embeddings, traces, patch_size, return_mask=True
        )
        
        # Reshape for batch processing
        batch_size, num_traces, max_patches, embed_dim = trace_patches.shape
        trace_patches_flat = trace_patches.view(batch_size * num_traces, max_patches, embed_dim)
        mask_flat = mask.view(batch_size * num_traces, max_patches)
        
        # Aggregate with GroupNet
        aggregated = model(trace_patches_flat, mask=mask_flat)
        
        # Reshape back
        return aggregated.view(batch_size, num_traces, embed_dim)
    
    else:
        raise ValueError("Either bboxes or traces must be provided")
