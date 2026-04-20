"""
Dataset for training GroupNet on trace captioning tasks.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import numpy as np


def is_caption_too_long(caption: str, max_length: int = 77) -> bool:
    """
    Check if a caption is too long for CLIP tokenizer.
    Uses a simple heuristic based on character count and word count.
    
    Args:
        caption: Text caption
        max_length: Maximum token length (77 for CLIP)
    
    Returns:
        True if caption is likely too long
    """
    # Simple heuristics to avoid loading CLIP tokenizer
    # CLIP uses BPE tokenization, roughly 4-5 chars per token on average
    word_count = len(caption.split())
    char_count = len(caption)
    
    # Conservative estimates (leaving room for special tokens like <|startoftext|>, <|endoftext|>)
    max_words = max_length - 2  # Account for special tokens
    max_chars = (max_length - 2) * 5  # Rough upper bound
    
    return word_count > max_words or char_count > max_chars


class TraceCaptioningDataset(Dataset):
    """
    Dataset for trace captioning.
    
    Each sample contains:
    - image_path: path to the image
    - trace: list of {x, y, t} dictionaries with normalized coordinates
    - caption: text caption corresponding to the trace
    """
    
    def __init__(
        self,
        dataset_path: str,
        image_base_path: str,
        image_backup_path: Optional[str] = None,
        transform: Optional[T.Compose] = None,
        split: str = 'train',
        max_samples: Optional[int] = None
    ):
        """
        Args:
            dataset_path: Path to JSON file with traces and captions
            image_base_path: Base path to images (e.g., /raid/datasets/coco/train2017)
            image_backup_path: Backup path if image not found in base_path
            transform: Torchvision transforms to apply to images
            split: 'train', 'val', or 'test' (requires split info in JSON or separate file)
            max_samples: Maximum number of samples to use (for debugging)
        """
        self.dataset_path = dataset_path
        self.image_base_path = image_base_path
        self.image_backup_path = image_backup_path
        self.transform = transform
        self.split = split
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            self.data = json.load(f)
        
        # Build sample list: (image_id, caption_idx)
        # Filter out captions that are too long for CLIP tokenizer (max 77 tokens)
        self.samples = []
        skipped_count = 0
        
        for img_id, img_obj in self.data.items():
            # Check if this is coco (needs zero-padding)
            is_coco = 'coco' in dataset_path.lower()
            if is_coco:
                img_id = img_id.zfill(12)
            
            for i in range(len(img_obj['captions'])):
                caption = img_obj['captions'][i]
                
                # Filter out captions that are too long for CLIP tokenizer
                if is_caption_too_long(caption):
                    skipped_count += 1
                    continue
                
                self.samples.append({
                    'image_id': img_id,
                    'caption_idx': i,
                    'caption': caption,
                    'trace': img_obj['traces'][i]
                })
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} samples with captions too long for CLIP tokenizer")
        
        # Apply max_samples limit
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict with keys:
                - image: transformed image tensor
                - trace: list of {x, y, t} trace points
                - caption: text caption
                - image_path: path to image file
        """
        sample = self.samples[idx]
        
        # Get image path
        img_id = sample['image_id']
        img_path = os.path.join(self.image_base_path, f"{img_id}.jpg")
        
        # Try backup path if image not found
        if not os.path.exists(img_path) and self.image_backup_path:
            img_path = os.path.join(self.image_backup_path, f"{img_id}.jpg")
        
        # Load and transform image
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return {
            'image': image,
            'trace': sample['trace'],
            'caption': sample['caption'],
            'image_path': img_path,
            'image_id': img_id
        }


def trace_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for batching traces.
    
    Args:
        batch: List of samples from TraceCaptioningDataset
    
    Returns:
        Batched dictionary with:
            - images: tensor of shape [B, C, H, W]
            - traces: list of traces (each trace is a list of {x, y, t})
            - captions: list of caption strings
            - image_paths: list of image paths
    """
    images = torch.stack([item['image'] for item in batch])
    traces = [item['trace'] for item in batch]
    captions = [item['caption'] for item in batch]
    image_paths = [item['image_path'] for item in batch]
    image_ids = [item['image_id'] for item in batch]
    
    return {
        'images': images,
        'traces': traces,
        'captions': captions,
        'image_paths': image_paths,
        'image_ids': image_ids
    }


def map_trace_to_patches(trace: List[Dict], patch_embeddings: torch.Tensor, 
                         n_patches: int) -> torch.Tensor:
    """
    Extract patch embeddings corresponding to a trace.
    
    Args:
        trace: List of {x, y, t} dictionaries with normalized coordinates [0, 1]
        patch_embeddings: Patch embeddings of shape [n_patches, n_patches, embed_dim]
        n_patches: Number of patches per side (e.g., 37 for 518x518 image with patch_size=14)
    
    Returns:
        Tensor of shape [num_selected_patches, embed_dim] containing embeddings of patches
        touched by the trace
    """
    # Create grid to track which patches are touched
    grid = torch.zeros((n_patches, n_patches), dtype=torch.bool)
    patch_size_norm = 1.0 / n_patches
    
    # Mark patches touched by trace
    for point in trace:
        x, y = point['x'], point['y']
        if 0 <= x <= 1 and 0 <= y <= 1:
            grid_x = int(x / patch_size_norm)
            grid_y = int(y / patch_size_norm)
            grid_x = min(grid_x, n_patches - 1)
            grid_y = min(grid_y, n_patches - 1)
            grid[grid_y, grid_x] = True
    
    # Get indices of selected patches
    selected_indices = grid.nonzero(as_tuple=True)
    
    # Extract corresponding patch embeddings
    if len(selected_indices[0]) == 0:
        # No patches selected, return a dummy embedding
        return patch_embeddings[0:1, 0:1, :].reshape(1, -1)
    
    selected_patches = patch_embeddings[selected_indices[0], selected_indices[1], :]
    
    return selected_patches


def create_trace_captioning_dataloaders(
    dataset_path: str = None,
    train_dataset_path: Optional[str] = None,
    val_dataset_path: Optional[str] = None,
    image_base_path: str = None,
    image_backup_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[T.Compose] = None,
    val_split: float = 0.1,
    seed: int = 42,
    max_samples: Optional[int] = None
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and validation dataloaders for trace captioning.
    
    Supports two modes:
    1. Single dataset file with random split (use dataset_path + val_split)
    2. Separate train/val files (use train_dataset_path + val_dataset_path)
    
    Args:
        dataset_path: Path to single dataset JSON (for random split mode)
        train_dataset_path: Path to training dataset JSON (for separate files mode)
        val_dataset_path: Path to validation dataset JSON (for separate files mode)
        image_base_path: Base path to images
        image_backup_path: Backup path for images
        batch_size: Batch size
        num_workers: Number of dataloader workers
        transform: Image transforms
        val_split: Fraction of data to use for validation (only used in random split mode)
        seed: Random seed for splitting (only used in random split mode)
        max_samples: Maximum samples to use (for debugging)
    
    Returns:
        (train_loader, val_loader)
    """
    # Determine which mode to use
    if train_dataset_path is not None and val_dataset_path is not None:
        # Mode 1: Separate train/val files
        print("Using separate train and val dataset files")
        train_dataset = TraceCaptioningDataset(
            dataset_path=train_dataset_path,
            image_base_path=image_base_path,
            image_backup_path=image_backup_path,
            transform=transform,
            split='train',
            max_samples=max_samples
        )
        
        val_dataset = TraceCaptioningDataset(
            dataset_path=val_dataset_path,
            image_base_path=image_base_path,
            image_backup_path=image_backup_path,
            transform=transform,
            split='val',
            max_samples=max_samples
        )
    elif dataset_path is not None:
        # Mode 2: Single file with random split
        print(f"Using single dataset file with {val_split:.1%} validation split")
        full_dataset = TraceCaptioningDataset(
            dataset_path=dataset_path,
            image_base_path=image_base_path,
            image_backup_path=image_backup_path,
            transform=transform,
            max_samples=max_samples
        )
        
        # Split into train/val
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(seed)
        )
    else:
        raise ValueError("Must provide either 'dataset_path' OR both 'train_dataset_path' and 'val_dataset_path'")
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=trace_collate_fn,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=trace_collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader
