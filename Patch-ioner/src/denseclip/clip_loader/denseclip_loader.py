#!/usr/bin/env python3
"""
DenseCLIP to CLIP Loader

A simple interface for loading DenseCLIP checkpoints as CLIP-like models
for text and image encoding.
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Tuple, Optional, Dict, Any
from PIL import Image
import torchvision.transforms as transforms

# Import local model components
try:
    from .models import CLIPVisionTransformer, CLIPTextEncoder, ResidualAttentionBlock, LayerNorm, QuickGELU
    from .tokenizer import tokenize
except ImportError:
    # Fallback for direct execution
    from models import CLIPVisionTransformer, CLIPTextEncoder, ResidualAttentionBlock, LayerNorm, QuickGELU
    from tokenizer import tokenize


class DenseCLIPModel(nn.Module):
    """
    A CLIP-like model loaded from DenseCLIP checkpoints.
    Provides simple text and image encoding functionality.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        
        # Initialize vision encoder
        vision_config = config['model']['vision']
        self.visual = CLIPVisionTransformer(
            input_resolution=vision_config['image_resolution'],
            patch_size=vision_config['vision_patch_size'],
            width=vision_config['vision_width'],
            layers=vision_config['vision_layers'],
            heads=vision_config['vision_width'] // 64,
            output_dim=vision_config['embed_dim']
        )
        
        # Initialize text encoder
        text_config = config['model']['text']
        self.text_encoder = CLIPTextEncoder(
            context_length=text_config['context_length'],
            vocab_size=text_config['vocab_size'],
            transformer_width=text_config['transformer_width'],
            transformer_heads=text_config['transformer_heads'],
            transformer_layers=text_config['transformer_layers'],
            embed_dim=text_config['embed_dim']
        )
        
        # Store configuration for preprocessing
        self.context_length = text_config['context_length']
        self.image_resolution = vision_config['image_resolution']
        
        # Initialize preprocessing
        self._setup_preprocessing()
        
    def _setup_preprocessing(self):
        """Setup image preprocessing pipeline"""
        preprocess_config = self.config['preprocessing']
        
        self.preprocess = transforms.Compose([
            transforms.Resize(self.image_resolution, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(self.image_resolution),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=preprocess_config['image_mean'],
                std=preprocess_config['image_std']
            )
        ])
    
    def encode_image(self, images: Union[torch.Tensor, List[Image.Image], Image.Image]) -> torch.Tensor:
        """
        Encode images into feature vectors
        
        Args:
            images: PIL Images, list of PIL Images, or preprocessed tensor
            
        Returns:
            Normalized image features [batch_size, embed_dim]
        """
        if isinstance(images, (list, tuple)):
            # List of PIL Images
            image_tensors = torch.stack([self.preprocess(img) for img in images])
        elif isinstance(images, Image.Image):
            # Single PIL Image
            image_tensors = self.preprocess(images).unsqueeze(0)
        elif isinstance(images, torch.Tensor):
            # Already preprocessed tensor
            image_tensors = images
        else:
            raise ValueError(f"Unsupported image type: {type(images)}")
        
        # Move to same device as model
        device = next(self.parameters()).device
        image_tensors = image_tensors.to(device)
        
        # Encode
        with torch.no_grad():
            image_features = self.visual(image_tensors)
            image_features = F.normalize(image_features, dim=-1)
            
        return image_features
    
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode texts into feature vectors
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Normalized text features [batch_size, embed_dim]
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Tokenize if necessary
        if isinstance(texts, list):
            tokens = tokenize(texts, context_length=self.context_length)
        elif isinstance(texts, torch.Tensor):
            if texts.dim() == 1:
                # Single tokenized text
                tokens = texts.unsqueeze(0)
            else:
                tokens = texts
        else:
            raise ValueError(f"Unsupported text type: {type(texts)}")    
        # Move to same device as model
        device = next(self.parameters()).device
        tokens = tokens.to(device)
        
        # Encode
        with torch.no_grad():
            text_features = self.text_encoder(tokens)
            text_features = F.normalize(text_features, dim=-1)
            
        return text_features
    
    def compute_similarity(self, 
                         image_features: torch.Tensor, 
                         text_features: torch.Tensor,
                         temperature: float = 1.0) -> torch.Tensor:
        """
        Compute similarity between image and text features
        
        Args:
            image_features: Normalized image features [N, embed_dim]
            text_features: Normalized text features [M, embed_dim]
            temperature: Temperature for scaling similarities
            
        Returns:
            Similarity matrix [N, M]
        """
        return (image_features @ text_features.t()) / temperature
    
    def forward(self, images: torch.Tensor, texts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for both image and text encoding
        
        Args:
            images: Preprocessed image tensor [batch_size, 3, H, W]
            texts: Tokenized text tensor [batch_size, context_length]
            
        Returns:
            Tuple of (image_features, text_features)
        """
        image_features = self.visual(images)
        text_features = self.text_encoder(texts)
        
        # Normalize features
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        return image_features, text_features


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_denseclip_weights(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load DenseCLIP checkpoint and extract relevant weights"""
    print(f"Loading DenseCLIP checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'state_dict' not in checkpoint:
        raise ValueError("Checkpoint doesn't contain 'state_dict'")
    
    state_dict = checkpoint['state_dict']
    
    # Extract vision and text encoder weights
    vision_weights = {}
    text_weights = {}
    
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            # Remove 'backbone.' prefix for vision encoder
            new_key = key[len('backbone.'):]
            vision_weights[new_key] = value
        elif key.startswith('text_encoder.'):
            # Remove 'text_encoder.' prefix  
            new_key = key[len('text_encoder.'):]
            text_weights[new_key] = value
    
    print(f"Extracted {len(vision_weights)} vision parameters")
    print(f"Extracted {len(text_weights)} text parameters")
    
    return {
        'vision': vision_weights,
        'text': text_weights,
        'full_state_dict': state_dict
    }


def load_denseclip_model(config_path: str, 
                        checkpoint_path: Optional[str] = None,
                        device: str = 'auto') -> DenseCLIPModel:
    """
    Load a DenseCLIP model from configuration and checkpoint
    
    Args:
        config_path: Path to YAML configuration file
        checkpoint_path: Optional path to checkpoint (overrides config)
        device: Device to load model on ('auto', 'cpu', 'cuda')
        
    Returns:
        Loaded DenseCLIPModel ready for inference
    """
    # Load configuration
    config = load_config(config_path)
    
    # Override checkpoint path if provided
    if checkpoint_path is not None:
        config['checkpoint']['path'] = checkpoint_path
    
    # Create model
    model = DenseCLIPModel(config)
    
    # Load weights
    checkpoint_path = config['checkpoint']['path']
    if os.path.exists(checkpoint_path):
        weights = load_denseclip_weights(checkpoint_path)
        
        # Load vision encoder weights
        if weights['vision']:
            missing_v, unexpected_v = model.visual.load_state_dict(weights['vision'], strict=False)
            if missing_v:
                print(f"Missing vision keys: {len(missing_v)} (expected for FPN/post-norm components)")
            if unexpected_v:
                # Filter out expected mismatches
                important_unexpected = [k for k in unexpected_v if not any(x in k for x in ['fpn', 'ln_post', 'proj'])]
                if important_unexpected:
                    print(f"Unexpected vision keys: {important_unexpected}")
                else:
                    print(f"✓ Vision weights loaded (ignoring {len(unexpected_v)} FPN/post-norm parameters)")
        
        # Load text encoder weights  
        if weights['text']:
            missing_t, unexpected_t = model.text_encoder.load_state_dict(weights['text'], strict=False)
            if missing_t:
                print(f"Missing text keys: {len(missing_t)}")
            if unexpected_t:
                print(f"Unexpected text keys: {unexpected_t}")
        
        print("✓ Model weights loaded successfully")
    else:
        print(f"⚠ Checkpoint not found at {checkpoint_path}, using random weights")
    
    # Setup device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = model.to(device)
    model.eval()
    
    print(f"✓ Model loaded on {device}")
    return model


# Convenience function
def load_clip(config_name: str = 'denseclip_vitb16', 
              checkpoint_path: Optional[str] = None,
              device: str = 'auto') -> DenseCLIPModel:
    """
    Convenience function to load a DenseCLIP model
    
    Args:
        config_name: Name of config file (without .yaml extension)
        checkpoint_path: Optional path to checkpoint
        device: Device to load on
        
    Returns:
        Loaded DenseCLIPModel
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'configs', f'{config_name}.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return load_denseclip_model(config_path, checkpoint_path, device)
