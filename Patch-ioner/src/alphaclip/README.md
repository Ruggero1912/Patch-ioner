# AlphaCLIP Standalone

A standalone, easy-to-use version of AlphaCLIP that can be integrated into any project without complex dependencies or setup.

## Overview

AlphaCLIP is an enhanced version of OpenAI's CLIP model that provides improved vision-language understanding capabilities. This standalone package makes it easy to use AlphaCLIP in your projects with minimal setup.

## Features

- **Easy Installation**: Simple pip install with minimal dependencies
- **Clean API**: Intuitive interface for loading models and processing data
- **Device Flexibility**: Automatic CUDA/CPU detection with manual override options
- **Model Variety**: Support for multiple AlphaCLIP model variants
- **Preprocessing Included**: Built-in image preprocessing and text tokenization

## Installation

### Requirements

- Python 3.7 or higher
- PyTorch 1.7.1 or higher
- CUDA (optional, for GPU acceleration)

### Install from source

```bash
# Clone or download this standalone package
cd alphaclip-standalone

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Core Dependencies

The package requires the following core dependencies:

```
torch>=1.7.1
torchvision
ftfy
regex
tqdm
loralib
Pillow
numpy
packaging
```

## Quick Start

### Basic Usage

```python
from alphaclip_loader import AlphaCLIPLoader

# Initialize the loader
loader = AlphaCLIPLoader()

# Load a model (this will download the model if not cached)
model, preprocess = loader.load_model("ViT-B/16")

# Tokenize text
text_tokens = loader.tokenize("A photo of a cat")

# Get text embeddings
text_features = loader.encode_text(model, "A photo of a cat")

print(f"Text features shape: {text_features.shape}")
```

### Advanced Usage

```python
import torch
from PIL import Image
from alphaclip_loader import AlphaCLIPLoader

# Initialize with specific device
loader = AlphaCLIPLoader(default_device="cuda")

# Load model with custom options
model, preprocess = loader.load_model(
    "ViT-B/16",
    device="cuda",
    lora_adapt=False,
    rank=16
)

# Process an image
image = Image.open("your_image.jpg")
image_tensor = preprocess(image).unsqueeze(0)  # Add batch dimension

# Get embeddings
with torch.no_grad():
    image_features = loader.encode_image(model, image_tensor)
    text_features = loader.encode_text(model, ["A photo of a cat", "A dog playing"])

# Compute similarities
similarities = loader.get_similarity(text_features, image_features)
print(f"Similarities: {similarities}")
```

### One-line Model Loading

```python
from alphaclip_loader import load_alphaclip

# Quick loading function
loader, model, preprocess = load_alphaclip("ViT-B/16", device="cuda")
```

## Available Models

You can check available models using:

```python
from alphaclip_loader import AlphaCLIPLoader

loader = AlphaCLIPLoader()
models = loader.available_models()
print("Available models:", models)
```

Typically includes:
- `ViT-B/32`
- `ViT-B/16`
- `ViT-L/14`
- `ViT-L/14@336px`
- `RN50`, `RN101`, `RN50x4`, `RN50x16`, `RN50x64`

## API Reference

### AlphaCLIPLoader Class

#### Methods

- **`__init__(default_device=None)`**: Initialize loader with optional default device
- **`available_models()`**: Get list of available model names
- **`load_model(name, **kwargs)`**: Load a model with preprocessing function
- **`tokenize(texts, context_length=77, truncate=True)`**: Tokenize text input
- **`encode_text(model, texts)`**: Encode text to embeddings
- **`encode_image(model, images)`**: Encode images to embeddings
- **`get_similarity(text_features, image_features)`**: Compute cosine similarity

#### load_model Parameters

- `name`: Model name or checkpoint path
- `alpha_vision_ckpt_pth`: Additional vision checkpoint path (default: "None")
- `device`: Device to load on (default: auto-detect)
- `jit`: Use JIT compilation (default: False)
- `download_root`: Model download directory (default: ~/.cache/clip)
- `lora_adapt`: Use LoRA adaptation (default: False)
- `rank`: LoRA rank if enabled (default: 16)

## Example Use Cases

### Image-Text Similarity

```python
from alphaclip_loader import load_alphaclip
from PIL import Image
import torch

loader, model, preprocess = load_alphaclip()

# Load and preprocess image
image = Image.open("cat.jpg")
image_input = preprocess(image).unsqueeze(0)

# Define candidate texts
texts = ["a cat", "a dog", "a bird", "a car"]

# Get features
image_features = loader.encode_image(model, image_input)
text_features = loader.encode_text(model, texts)

# Calculate similarities
similarities = loader.get_similarity(text_features, image_features)

# Find best match
best_match_idx = similarities.argmax()
print(f"Best match: {texts[best_match_idx]} (score: {similarities[best_match_idx]:.3f})")
```

### Batch Processing

```python
from alphaclip_loader import AlphaCLIPLoader
import torch

loader = AlphaCLIPLoader()
model, preprocess = loader.load_model("ViT-B/16")

# Process multiple texts at once
texts = [
    "A red apple on a table",
    "A dog running in the park", 
    "A beautiful sunset"
]

# Batch tokenization and encoding
text_features = loader.encode_text(model, texts)
print(f"Batch text features shape: {text_features.shape}")  # [3, 512]
```

## Performance Tips

1. **GPU Usage**: Use CUDA for better performance with larger models
2. **Batch Processing**: Process multiple texts/images together when possible
3. **Model Caching**: Models are automatically cached after first download
4. **Memory Management**: Use `torch.no_grad()` during inference to save memory

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU
2. **Model Download Fails**: Check internet connection and disk space
3. **Import Errors**: Ensure all dependencies are installed

### Dependencies Issues

If you encounter import errors, try:

```bash
pip install --upgrade torch torchvision
pip install ftfy regex tqdm loralib
```

## File Structure

```
alphaclip-standalone/
├── __init__.py              # Package initialization
├── alphaclip_loader.py      # Main loader class
├── requirements.txt         # Dependencies
├── setup.py                # Package setup
├── README.md               # This file
└── alpha_clip/             # Core AlphaCLIP modules
    ├── __init__.py
    ├── alpha_clip.py       # Main AlphaCLIP functions
    ├── model.py           # Model architectures
    ├── simple_tokenizer.py # Text tokenization
    └── bpe_simple_vocab_16e6.txt.gz  # Tokenizer vocabulary
```

## License

This standalone package maintains the same license as the original AlphaCLIP project.

## Contributing

This is a standalone distribution. For contributions to the core AlphaCLIP model, please refer to the main AlphaCLIP repository.

## Changelog

### Version 1.0.0
- Initial standalone release
- Clean API with AlphaCLIPLoader class
- Comprehensive documentation and examples
- Easy installation and setup
