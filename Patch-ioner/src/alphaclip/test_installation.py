#!/usr/bin/env python3
"""
Test script for AlphaCLIP Standalone

This script tests the basic functionality of the standalone package
to ensure everything is working correctly.
"""

import sys
import os

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__} imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import PyTorch: {e}")
        return False
    
    try:
        import torchvision
        print(f"✓ Torchvision imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import torchvision: {e}")
        return False
    
    try:
        from alphaclip_loader import AlphaCLIPLoader
        print("✓ AlphaCLIPLoader imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import AlphaCLIPLoader: {e}")
        return False
    
    try:
        import loralib
        print("✓ LoraLib imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import loralib: {e}")
        return False
    
    return True

def test_model_loading():
    """Test loading a model."""
    print("\nTesting model loading...")
    
    try:
        from alphaclip_loader import AlphaCLIPLoader
        
        loader = AlphaCLIPLoader(default_device="cpu")  # Use CPU for testing
        models = loader.available_models()
        print(f"✓ Available models: {models}")
        
        # Try to load the smallest model for testing
        print("Loading ViT-B/32 model (this may take a while for first download)...")
        model, preprocess = loader.load_model("ViT-B/32", device="cpu")
        print("✓ Model loaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False

def test_tokenization():
    """Test text tokenization."""
    print("\nTesting tokenization...")
    
    try:
        from alphaclip_loader import AlphaCLIPLoader
        
        loader = AlphaCLIPLoader()
        test_text = "a photo of a cat"
        tokens = loader.tokenize(test_text)
        print(f"✓ Tokenized '{test_text}' to shape {tokens.shape}")
        
        # Test batch tokenization
        test_texts = ["a cat", "a dog", "a bird"]
        batch_tokens = loader.tokenize(test_texts)
        print(f"✓ Batch tokenized {len(test_texts)} texts to shape {batch_tokens.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed tokenization test: {e}")
        return False

def test_text_encoding():
    """Test text encoding with a loaded model."""
    print("\nTesting text encoding...")
    
    try:
        from alphaclip_loader import AlphaCLIPLoader
        
        loader = AlphaCLIPLoader(default_device="cpu")
        model, preprocess = loader.load_model("ViT-B/32", device="cpu")
        
        test_text = "a photo of a cat"
        features = loader.encode_text(model, test_text)
        print(f"✓ Encoded text to features with shape {features.shape}")
        
        # Test batch encoding
        test_texts = ["a cat", "a dog"]
        batch_features = loader.encode_text(model, test_texts)
        print(f"✓ Batch encoded {len(test_texts)} texts to shape {batch_features.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Failed text encoding test: {e}")
        return False

def main():
    """Run all tests."""
    print("AlphaCLIP Standalone Test Suite")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_tokenization,
        test_model_loading,
        test_text_encoding,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{'='*40}")
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! AlphaCLIP Standalone is working correctly.")
        return 0
    else:
        print("❌ Some tests failed. Please check the error messages above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
