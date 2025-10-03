# AlphaCLIP Standalone - Installation Guide

## Quick Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
cd alphaclip-standalone
pip install -r requirements.txt
```

### Step 2: Install the Package

```bash
# Install in development mode (recommended for testing)
pip install -e .

# OR install normally
pip install .
```

### Step 3: Test Installation

```bash
python test_installation.py
```

### Step 4: Run Example

```bash
python example.py
```

## Manual Dependency Installation

If you encounter issues with the requirements.txt, install dependencies manually:

```bash
# Core PyTorch (choose appropriate version for your system)
pip install torch torchvision torchaudio

# Text processing
pip install ftfy regex tqdm

# LoRA support
pip install loralib

# Image processing
pip install Pillow

# Utilities
pip install numpy packaging
```

## GPU Support

For CUDA support, make sure you install PyTorch with CUDA:

```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Check your CUDA version with: nvidia-smi
```

## Verification

After installation, verify everything works:

```python
from alphaclip_loader import AlphaCLIPLoader

# This should work without errors
loader = AlphaCLIPLoader()
models = loader.available_models()
print("Available models:", models)
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'loralib'**
   ```bash
   pip install loralib
   ```

2. **CUDA out of memory**
   - Use CPU: `AlphaCLIPLoader(default_device="cpu")`
   - Or use a smaller model like "ViT-B/32"

3. **Model download fails**
   - Check internet connection
   - Ensure you have enough disk space (~1GB per model)
   - Models are cached in `~/.cache/clip/`

4. **Permission errors**
   - Use `--user` flag: `pip install --user -e .`

### Getting Help

If you encounter issues:
1. Check that all dependencies are properly installed
2. Run the test script: `python test_installation.py`
3. Check CUDA compatibility if using GPU
4. Ensure Python version is 3.7+
