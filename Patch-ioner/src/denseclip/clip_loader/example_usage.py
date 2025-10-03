#!/usr/bin/env python3
"""
Example usage of the DenseCLIP to CLIP loader
"""

import sys
import os

# Add the clip_loader to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from denseclip_loader import load_clip
import torch


def main():
    print("🚀 DenseCLIP to CLIP Loader Example")
    print("=" * 50)
    
    # Load model
    print("Loading DenseCLIP model...")
    try:
        model = load_clip('denseclip_segmentation_vitb16')
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Text context length: {model.context_length}")
    print(f"Image resolution: {model.image_resolution}")
    
    # Test text encoding
    print("\n📝 Testing text encoding...")
    texts = [
        "a photo of a cat",
        "a photo of a dog", 
        "a photo of a car",
        "a person walking",
        "a beautiful sunset"
    ]
    
    text_features = model.encode_text(texts)
    print(f"Text features shape: {text_features.shape}")
    print(f"Text features norm: {text_features.norm(dim=-1)}")  # Should be ~1.0 (normalized)
    
    # Test text-text similarities
    print("\n🔍 Text-to-text similarities:")
    text_similarities = model.compute_similarity(text_features, text_features)
    
    print(f"{'Text':<20} {'Self-sim':<10} {'vs Cat':<10} {'vs Dog':<10}")
    print("-" * 50)
    for i, text in enumerate(texts):
        self_sim = text_similarities[i, i].item()
        cat_sim = text_similarities[i, 0].item()
        dog_sim = text_similarities[i, 1].item()
        print(f"{text:<20} {self_sim:<10.3f} {cat_sim:<10.3f} {dog_sim:<10.3f}")
    
    # Test zero-shot classification concepts
    print("\n🎯 Zero-shot classification example:")
    test_queries = [
        "an animal",
        "a vehicle", 
        "a person",
        "nature scene"
    ]
    
    query_features = model.encode_text(test_queries)
    classification_similarities = model.compute_similarity(text_features, query_features)
    
    print(f"{'Original Text':<20} {'Animal':<8} {'Vehicle':<8} {'Person':<8} {'Nature':<8}")
    print("-" * 60)
    for i, text in enumerate(texts):
        sims = classification_similarities[i]
        print(f"{text:<20} {sims[0]:<8.3f} {sims[1]:<8.3f} {sims[2]:<8.3f} {sims[3]:<8.3f}")
    
    # Test feature statistics
    print("\n📊 Feature statistics:")
    print(f"Text feature mean: {text_features.mean():.6f}")
    print(f"Text feature std: {text_features.std():.6f}")
    print(f"Text feature min: {text_features.min():.6f}")
    print(f"Text feature max: {text_features.max():.6f}")
    
    # Test model components
    print("\n🔧 Model architecture:")
    print(f"Vision encoder: {type(model.visual).__name__}")
    print(f"Text encoder: {type(model.text_encoder).__name__}")
    
    # Count parameters
    vision_params = sum(p.numel() for p in model.visual.parameters())
    text_params = sum(p.numel() for p in model.text_encoder.parameters())
    total_params = vision_params + text_params
    
    print(f"\n📈 Parameter count:")
    print(f"Vision encoder: {vision_params:,}")
    print(f"Text encoder: {text_params:,}")
    print(f"Total: {total_params:,}")
    
    print("\n✅ All tests completed successfully!")
    print("\n💡 Usage tip:")
    print("   from clip_loader import load_clip")
    print("   model = load_clip('denseclip_vitb16')")
    print("   features = model.encode_text(['your text here'])")


if __name__ == "__main__":
    main()
