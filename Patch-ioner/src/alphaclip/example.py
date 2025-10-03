#!/usr/bin/env python3
"""
Example usage of AlphaCLIP Standalone

This script demonstrates basic usage of the AlphaCLIP standalone package.
"""

import torch
import numpy as np
from alphaclip_loader import AlphaCLIPLoader, load_alphaclip

def main():
    print("AlphaCLIP Standalone Example")
    print("=" * 40)
    
    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Method 1: Using the loader class
    print("\n1. Using AlphaCLIPLoader class:")
    loader = AlphaCLIPLoader(default_device=device)
    
    # Show available models
    models = loader.available_models()
    print(f"Available models: {models}")
    
    # Load a model
    print("\nLoading ViT-B/16 model...")
    model, preprocess = loader.load_model("ViT-B/16")
    print(f"Model loaded successfully!")
    
    # Test text encoding
    test_texts = [
        "a photo of a cat",
        "a dog running in the park",
        "a beautiful sunset over the ocean"
    ]
    
    print(f"\nEncoding {len(test_texts)} texts...")
    text_features = loader.encode_text(model, test_texts)
    print(f"Text features shape: {text_features.shape}")
    
    # Compute similarities between texts
    print("\nComputing text-to-text similarities:")
    similarities = loader.get_similarity(text_features, text_features)
    
    for i, text1 in enumerate(test_texts):
        for j, text2 in enumerate(test_texts):
            if i <= j:  # Only show upper triangle
                sim = similarities[i, j].item()
                print(f"  '{text1}' <-> '{text2}': {sim:.3f}")
    
    # Method 2: Using the quick loader function
    print("\n\n2. Using quick loader function:")
    loader2, model2, preprocess2 = load_alphaclip("ViT-B/16", device=device)
    
    # Test single text
    single_text = "a red apple on a wooden table"
    single_features = loader2.encode_text(model2, single_text)
    print(f"Single text '{single_text}' encoded to shape: {single_features.shape}")
    
    # Test tokenization
    print("\n3. Tokenization example:")
    tokens = loader.tokenize(test_texts)
    print(f"Tokenized {len(test_texts)} texts to shape: {tokens.shape}")
    
    # Show some token examples
    print("First few tokens for each text:")
    for i, text in enumerate(test_texts):
        print(f"  '{text}': {tokens[i][:10].tolist()}...")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
