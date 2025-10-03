# DenseCLIP to CLIP Loader

A simple interface for loading DenseCLIP checkpoints as CLIP-like models for text and image encoding.

## Overview

This module provides a clean API to load DenseCLIP models and use them like standard CLIP models for encoding text and images. It abstracts away the complexity of DenseCLIP's detection/segmentation components and exposes only the core vision-language encoding functionality.

## Features

- ✅ **Simple API**: Load DenseCLIP models with a single function call
- ✅ **CLIP-like Interface**: Familiar `encode_text()` and `encode_image()` methods
- ✅ **Flexible Configuration**: YAML-based configuration system
- ✅ **Multiple Input Types**: Support for PIL Images, image tensors, strings, and text lists
- ✅ **Automatic Preprocessing**: Built-in image preprocessing pipeline
- ✅ **Device Management**: Automatic GPU/CPU detection and placement

## Quick Start

```python
from clip_loader import load_clip

# Load DenseCLIP model with default configuration
model = load_clip('denseclip_segmentation_vitb16')

# Encode text
texts = ["a photo of a cat", "a photo of a dog"]
text_features = model.encode_text(texts)

# Encode images (PIL Images)
from PIL import Image
images = [Image.open("cat.jpg"), Image.open("dog.jpg")]
image_features = model.encode_image(images)

# Compute similarities
similarities = model.compute_similarity(image_features, text_features)
print(f"Image-text similarities: {similarities}")
```

## Configuration

Models are configured using YAML files in the `configs/` directory. The main configuration for DenseCLIP ViT-B/16 is in `configs/denseclip_vitb16.yaml`.

### Configuration Structure

```yaml
model:
  name: "denseclip_vitb16"
  type: "vit"
  
  vision:
    image_resolution: 224
    vision_layers: 12
    vision_width: 768
    vision_patch_size: 16
    embed_dim: 512
  
  text:
    context_length: 13  # DenseCLIP uses shorter context
    vocab_size: 49408
    transformer_width: 512
    transformer_heads: 8
    transformer_layers: 12
    embed_dim: 512

checkpoint:
  path: "/path/to/denseclip/checkpoint.pth"
  format: "denseclip"
  
preprocessing:
  image_mean: [0.48145466, 0.4578275, 0.40821073]
  image_std: [0.26862954, 0.26130258, 0.27577711]
  normalize: true
```

## API Reference

### Core Functions

#### `load_clip(config_name, checkpoint_path=None, device='auto')`

Load a DenseCLIP model with the specified configuration.

**Parameters:**
- `config_name` (str): Name of config file (without .yaml extension)
- `checkpoint_path` (str, optional): Path to checkpoint file (overrides config)
- `device` (str): Device to load on ('auto', 'cpu', 'cuda')

**Returns:**
- `DenseCLIPModel`: Loaded model ready for inference

#### `load_denseclip_model(config_path, checkpoint_path=None, device='auto')`

Load a DenseCLIP model from configuration file path.

### DenseCLIPModel Methods

#### `encode_text(texts)`

Encode text into feature vectors.

**Parameters:**
- `texts` (str or List[str]): Text string(s) to encode

**Returns:**
- `torch.Tensor`: Normalized text features [batch_size, embed_dim]

#### `encode_image(images)`

Encode images into feature vectors.

**Parameters:**
- `images`: PIL Image, List[PIL.Image], or preprocessed tensor

**Returns:**
- `torch.Tensor`: Normalized image features [batch_size, embed_dim]

#### `compute_similarity(image_features, text_features, temperature=1.0)`

Compute similarity between image and text features.

**Parameters:**
- `image_features` (torch.Tensor): Image features [N, embed_dim]
- `text_features` (torch.Tensor): Text features [M, embed_dim]
- `temperature` (float): Temperature scaling factor

**Returns:**
- `torch.Tensor`: Similarity matrix [N, M]

## Examples

### Basic Text-Image Retrieval

```python
from clip_loader import load_clip
from PIL import Image

# Load model
model = load_clip('denseclip_vitb16')

# Load and encode images
images = [
    Image.open("cat.jpg"),
    Image.open("dog.jpg"),
    Image.open("car.jpg")
]
image_features = model.encode_image(images)

# Encode text queries
queries = [
    "a cute cat",
    "a happy dog", 
    "a red car"
]
text_features = model.encode_text(queries)

# Find best matches
similarities = model.compute_similarity(image_features, text_features)
best_matches = similarities.argmax(dim=1)

for i, query in enumerate(queries):
    best_image_idx = best_matches[i]
    score = similarities[best_image_idx, i].item()
    print(f"Query '{query}' -> Image {best_image_idx} (score: {score:.3f})")
```

### Zero-Shot Classification

```python
from clip_loader import load_clip
from PIL import Image

model = load_clip('denseclip_vitb16')

# Load test image
image = Image.open("test_image.jpg")
image_features = model.encode_image(image)

# Define class labels
class_labels = [
    "a photo of a cat",
    "a photo of a dog", 
    "a photo of a bird",
    "a photo of a car",
    "a photo of a house"
]

# Encode labels
text_features = model.encode_text(class_labels)

# Classify
similarities = model.compute_similarity(image_features, text_features)
probabilities = similarities.softmax(dim=-1)

# Show results
for i, label in enumerate(class_labels):
    prob = probabilities[0, i].item()
    print(f"{label}: {prob:.3f}")
```

### Custom Configuration

```python
from clip_loader import load_denseclip_model

# Load with custom config
model = load_denseclip_model(
    config_path='configs/custom_config.yaml',
    checkpoint_path='/path/to/custom/checkpoint.pth',
    device='cuda:1'
)
```

## Requirements

- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- Pillow >= 8.0.0
- PyYAML >= 5.4.0

## Notes

- DenseCLIP uses a shorter text context length (13) compared to standard CLIP (77)
- The model preserves ~98% similarity with original CLIP text representations
- Image preprocessing follows CLIP's standard normalization
- All features are L2-normalized for cosine similarity computation

## Supported Models

Currently supported:
- `denseclip_vitb16`: DenseCLIP with ViT-B/16 backbone

To add support for other DenseCLIP variants, create new configuration files in the `configs/` directory.
