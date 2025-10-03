import torch
import torch.nn.functional as F
import skimage.io as io
from PIL import Image
import pickle
import json
import os
from tqdm import tqdm
import argparse
import torchvision.transforms as T
import numpy as np
import yaml
import clip
import sys

# Add the src directory to the path so we can import ProjectionLayer
sys.path.append(os.path.join(os.path.dirname(__file__), '../..', 'src'))


# Container to store intermediate outputs for feature extraction
feats = {}

def get_self_attention(module, input, output):
    """Hook to capture self-attention weights"""
    global qkv_attention_out
    qkv_attention_out = output

def get_layer_n_output(module, input, output):
    """Hook to capture intermediate layer output"""
    feats['intermediate_output'] = output

def transform_to_standard_dino_out(x, model, num_global_tokens=1):
    """Transform raw DINO output to standardized format"""
    x_norm = model.norm(x)
    if num_global_tokens == 1:
        # Standard model without registers
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": None,
            "x_norm_patchtokens": x_norm[:, 1:],
            "x_prenorm": x,
        }
    else:
        # Model with registers (num_global_tokens = 5)
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1:num_global_tokens],
            "x_norm_patchtokens": x_norm[:, num_global_tokens:],
            "x_prenorm": x,
        }

def process_self_attention(output, batch_size, num_tokens, num_attn_heads, embed_dim, scale, num_global_tokens, ret_self_attn_maps=False):
    """Process self-attention output to compute attention weights"""
    qkv = output.reshape(batch_size, num_tokens, 3, num_attn_heads, embed_dim // num_attn_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv[0] * scale, qkv[1], qkv[2]
    attn = q @ k.transpose(-2, -1)
    self_attn_maps = attn[:, :, 0, num_global_tokens:]  # CLS token attention to patches
    self_attn = self_attn_maps.mean(dim=1)  # Average over attention heads
    self_attn = self_attn.softmax(dim=-1)
    if ret_self_attn_maps:
        return self_attn, self_attn_maps
    else:
        return self_attn


# Global variables to store hook outputs
dino_layer_n_output = None
qkv_attention_out = None

def get_layer_n_output(module, input, output):
    """Hook to capture intermediate layer output"""
    global dino_layer_n_output
    dino_layer_n_output = output


def select_most_significant_patch(dino_outs, self_attn, criteria, cls_token=None, caption_embedding=None):
    """
    Select the most significant patch token based on different criteria.
    
    Args:
        dino_outs: Dictionary containing normalized DINO outputs
        self_attn: Self-attention weights from CLS to patches [batch_size, num_patches]
        criteria: Selection criteria ('max_attention', 'most_similar_to_cls', etc.)
        cls_token: CLS token embeddings [batch_size, embed_dim]
        caption_embedding: Text caption embeddings [batch_size, embed_dim]
        
    Returns:
        selected_patches: [batch_size, embed_dim] - Selected patch embeddings
    """
    patch_tokens = dino_outs['x_norm_patchtokens']  # [batch_size, num_patches, embed_dim]
    batch_size, num_patches, embed_dim = patch_tokens.shape
    
    if criteria == "max_attention":
        # Select patch with highest attention weight from CLS token
        if self_attn is None:
            raise ValueError("self_attn required for max_attention criteria")
        max_attn_indices = self_attn.argmax(dim=1)  # [batch_size]
        selected_patches = patch_tokens[torch.arange(batch_size), max_attn_indices]
        
    elif criteria == "most_similar_to_cls":
        # Select patch most similar to CLS token using cosine similarity
        if cls_token is None:
            raise ValueError("cls_token required for most_similar_to_cls criteria")
        # Compute cosine similarity between CLS and all patches
        cls_normalized = F.normalize(cls_token, p=2, dim=1)  # [batch_size, embed_dim]
        patches_normalized = F.normalize(patch_tokens, p=2, dim=2)  # [batch_size, num_patches, embed_dim]
        similarities = torch.bmm(patches_normalized, cls_normalized.unsqueeze(2)).squeeze(2)  # [batch_size, num_patches]
        max_sim_indices = similarities.argmax(dim=1)  # [batch_size]
        selected_patches = patch_tokens[torch.arange(batch_size), max_sim_indices]
        
    elif criteria == "most_similar_to_caption":
        # Select patch most similar to caption embedding
        if caption_embedding is None:
            raise ValueError("caption_embedding required for most_similar_to_caption criteria")
        caption_normalized = F.normalize(caption_embedding, p=2, dim=1)  # [batch_size, embed_dim]
        patches_normalized = F.normalize(patch_tokens, p=2, dim=2)  # [batch_size, num_patches, embed_dim]
        similarities = torch.bmm(patches_normalized, caption_normalized.unsqueeze(2)).squeeze(2)  # [batch_size, num_patches]
        max_sim_indices = similarities.argmax(dim=1)  # [batch_size]
        selected_patches = patch_tokens[torch.arange(batch_size), max_sim_indices]
        
    elif criteria == "max_norm":
        # Select patch with highest L2 norm
        patch_norms = torch.norm(patch_tokens, p=2, dim=2)  # [batch_size, num_patches]
        max_norm_indices = patch_norms.argmax(dim=1)  # [batch_size]
        selected_patches = patch_tokens[torch.arange(batch_size), max_norm_indices]
        
    elif criteria == "centroid_distance":
        # Select patch farthest from the centroid of all patches
        centroid = patch_tokens.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
        distances = torch.norm(patch_tokens - centroid, p=2, dim=2)  # [batch_size, num_patches]
        max_dist_indices = distances.argmax(dim=1)  # [batch_size]
        selected_patches = patch_tokens[torch.arange(batch_size), max_dist_indices]
        
    else:
        raise ValueError(f"Unknown patch selection criteria: {criteria}")
    
    return selected_patches


def load_text_encoder(text_encoder_path, device, config_path=None):
    """
    Load a text encoder model for caption similarity.
    Supports Talk2Dino, CLIP, and DINO.txt-based text encoders.
    """
    if text_encoder_path is None:
        return None
    
    print(f"Loading text encoder from: {text_encoder_path}")
    
    # Check for DINO.txt model
    if text_encoder_path.lower() == 'dinotxt' or text_encoder_path.lower() == 'dino.txt':
        # Load DINO.txt model
        try:
            from src.dinotxt_utils import get_tokenizer
            
            print("Loading DINO.txt model...")
            dinotxt_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l')
            dinotxt_model.eval()
            dinotxt_model.to(device)
            
            tokenizer = get_tokenizer()
            
            return {
                'type': 'dinotxt',
                'model': dinotxt_model,
                'tokenizer': tokenizer
            }
            
        except ImportError:
            raise ImportError("Could not import dinotxt_utils. Make sure src/dinotxt_utils.py is accessible.")
        except Exception as e:
            raise RuntimeError(f"Failed to load DINO.txt model: {e}")
    
    # Check if it's a Talk2Dino model (expect config and weights)
    elif text_encoder_path.endswith('.pth') or text_encoder_path.endswith('.pt'):
        # Use provided config or auto-find
        if config_path is None:
            # Look for corresponding config file
            base_path = text_encoder_path.rsplit('.', 1)[0]
            config_path = base_path + '.yaml'
            
            # Alternative config path patterns
            if not os.path.exists(config_path):
                # Try configs_talk2dino directory
                config_name = os.path.basename(base_path) + '.yaml'
                config_path = os.path.join(os.path.dirname(__file__), 'configs_talk2dino', config_name)
                
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Could not find config file for {text_encoder_path}. "
                                   f"Expected at {config_path} or specify --text_encoder_config.")
        
        # Load Talk2Dino model
        try:
            from src.model import ProjectionLayer
            
            print(f"Using config: {config_path}")
            
            # Load the projection layer
            talk2dino = ProjectionLayer.from_config(config_path)
            talk2dino.load_state_dict(torch.load(text_encoder_path, map_location=device))
            talk2dino.to(device)
            talk2dino.eval()
            
            # Load CLIP model for text encoding
            clip_model, _ = clip.load("ViT-B/32", device=device)
            clip_model.eval()
            
            return {
                'type': 'talk2dino',
                'talk2dino': talk2dino,
                'clip_model': clip_model,
                'config_path': config_path
            }
            
        except ImportError:
            raise ImportError("Could not import ProjectionLayer. Make sure src/model.py is accessible.")
    
    else:
        # Assume it's a direct model path (CLIP or other)
        try:
            # Try loading as a CLIP model
            clip_model, _ = clip.load(text_encoder_path, device=device)
            clip_model.eval()
            
            return {
                'type': 'clip',
                'clip_model': clip_model
            }
        except:
            raise ValueError(f"Could not load text encoder from {text_encoder_path}. "
                           f"Supported formats: 1) 'dinotxt' or 'dino.txt' for DINO.txt model, "
                           f"2) Talk2Dino (.pth/.pt), 3) CLIP model names.")


def encode_caption(caption, text_encoder, device):
    """
    Encode a text caption using the loaded text encoder.
    """
    if text_encoder is None:
        return None
    
    if text_encoder['type'] == 'dinotxt':
        # Use DINO.txt pipeline: tokenize + encode + extract patch-aligned features
        with torch.no_grad():
            # Tokenize with DINO.txt tokenizer
            text_tokens = text_encoder['tokenizer'].tokenize([caption]).to(device)
            
            # Encode with DINO.txt model
            dinotxt_features = text_encoder['model'].encode_text(text_tokens)
            
            # Extract patch-aligned text embeddings (dimensions 1024:)
            # DINO.txt concatenates standard text features [0:1024] and patch-aligned features [1024:]
            patch_aligned_features = dinotxt_features[:, 1024:]
            
            # Normalize the features to match DINO feature space
            patch_aligned_features = F.normalize(patch_aligned_features, p=2, dim=-1)
            return patch_aligned_features
            
    elif text_encoder['type'] == 'talk2dino':
        # Use Talk2Dino pipeline: CLIP text encoding + Talk2Dino projection
        with torch.no_grad():
            # Tokenize and encode with CLIP
            text_tokens = clip.tokenize([caption]).to(device)
            clip_text_features = text_encoder['clip_model'].encode_text(text_tokens)
            
            # Project through Talk2Dino to DINO space
            dino_text_features = text_encoder['talk2dino'].project_clip_txt(clip_text_features)
            
            # Normalize the encoded text to match DINO feature space
            dino_text_features = F.normalize(dino_text_features, p=2, dim=-1)
            return dino_text_features
            
    elif text_encoder['type'] == 'clip':
        # Use CLIP directly
        with torch.no_grad():
            text_tokens = clip.tokenize([caption]).to(device)
            clip_text_features = text_encoder['clip_model'].encode_text(text_tokens)
            
            # Normalize the features
            clip_text_features = F.normalize(clip_text_features, p=2, dim=-1)
            return clip_text_features
    
    else:
        raise ValueError(f"Unknown text encoder type: {text_encoder['type']}")


def main(dino_model_type: str, resize_dim: int = 518, crop_dim: int = 518, 
         coco_images_dir: str = "/raid/datasets/coco/", captions_file: str = "/raid/datasets/coco/train_split_karpathy.json",
         output_file: str = None, feature_type: str = "cls", extract_attention: bool = False,
         patch_selection_criteria: str = "max_attention", text_encoder_path: str = None, text_encoder_config: str = None):
    """
    Extract DINO features from COCO images for ClipCap training.
    
    Args:
        feature_type: Type of features to extract
            - "cls": CLS token features (default)
            - "avg_patch": Mean pooled patch token features
            - "avg_self_attn": Self-attention weighted patch token features
            - "most_significant_patch": Single most important patch token
        extract_attention: Whether to extract self-attention weights (required for avg_self_attn)
        patch_selection_criteria: Criteria for selecting most significant patch
        text_encoder_path: Path to text encoder for caption similarity
    """
    device = torch.device('cuda:0')
    dino_model_name = dino_model_type.replace('/', '_')
    
    # Determine model properties
    num_global_tokens = 1 if "reg" not in dino_model_type else 5
    patch_size = 14  # DINOv2 uses 14x14 patches
    num_patch_tokens = (crop_dim // patch_size) * (crop_dim // patch_size)
    num_tokens = num_global_tokens + num_patch_tokens
    
    # Get embedding dimension based on model type
    if 'vitl' in dino_model_type:
        embed_dim = 1024
        num_attn_heads = 16
    elif 'vitb' in dino_model_type:
        embed_dim = 768
        num_attn_heads = 12
    elif 'vits' in dino_model_type:
        embed_dim = 384
        num_attn_heads = 6
    elif 'vitg' in dino_model_type:
        embed_dim = 1536
        num_attn_heads = 24
    else:
        raise ValueError(f"Unknown model type: {dino_model_type}")
    
    scale = (embed_dim // num_attn_heads) ** -0.5
    
    # Set default output path if not specified
    if output_file is None:
        if feature_type == "cls":
            feature_suffix = ""
        elif feature_type == "most_significant_patch":
            if patch_selection_criteria == "most_similar_to_caption" and text_encoder_path is not None:
                # Determine text encoder type from path to create unique filename
                if text_encoder_path.lower() in ['dinotxt', 'dino.txt']:
                    text_encoder_suffix = "_dinotxt"
                elif text_encoder_path.endswith('.pth') or text_encoder_path.endswith('.pt'):
                    text_encoder_suffix = "_t2d"  # Talk2Dino
                else:
                    # CLIP or other models
                    text_encoder_suffix = "_clip"
                feature_suffix = f"_{feature_type}_{patch_selection_criteria}{text_encoder_suffix}"
            else:
                feature_suffix = f"_{feature_type}_{patch_selection_criteria}"
        else:
            feature_suffix = f"_{feature_type}"
        output_file = f"/raid/datasets/models_weights/clipcap/training-features/coco_karpathy_split_{dino_model_name}{feature_suffix}_train.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load DINO model
    print(f"Loading DINO model: {dino_model_type}")
    print(f"Feature type: {feature_type}")
    if feature_type == "most_significant_patch":
        print(f"Patch selection criteria: {patch_selection_criteria}")
    print(f"Model properties: embed_dim={embed_dim}, num_heads={num_attn_heads}, num_global_tokens={num_global_tokens}")
    
    if 'dinov2' in dino_model_type:
        model_family = 'facebookresearch/dinov2'
        dino_model = torch.hub.load(model_family, dino_model_type)
    else:
        raise ValueError(f"Unsupported DINO model type: {dino_model_type}")
    
    # Setup transforms for DINO
    image_transforms = T.Compose([
        T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(crop_dim),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    dino_model.eval()
    dino_model.to(device)
    
    # Register hooks if we need attention or intermediate outputs
    if feature_type == "avg_self_attn" or extract_attention or \
       (feature_type == "most_significant_patch" and patch_selection_criteria in ["max_attention", "most_similar_to_caption"]):
        print("Registering hooks for attention extraction...")
        dino_model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
    
    if feature_type in ["avg_patch", "avg_self_attn", "most_significant_patch"]:
        print("Registering hooks for intermediate output extraction...")
        dino_model.blocks[-1].register_forward_hook(get_layer_n_output)
    
    # Load caption data
    print(f"Loading captions from: {captions_file}")
    with open(captions_file, 'r') as f:
        data = json.load(f)
    
    # Handle different annotation formats
    if isinstance(data, list):
        # Original ClipCap format: list of dicts with 'image_id' and 'caption'
        annotations = data
        print(f"{len(annotations)} captions loaded from json (ClipCap format)")
    elif isinstance(data, dict) and 'annotations' in data:
        # Karpathy format: dict with 'annotations' key
        annotations = data['annotations']
        print(f"{len(annotations)} captions loaded from json (Karpathy format)")
        
        # Create image ID to filename mapping for faster lookup
        if 'images' in data:
            image_id_to_filename = {img['id']: img['file_name'] for img in data['images']}
        else:
            image_id_to_filename = {}
    else:
        raise ValueError("Unsupported annotation format")
    
    # Load text encoder if needed for caption similarity
    text_encoder = None
    if feature_type == "most_significant_patch" and patch_selection_criteria == "most_similar_to_caption":
        if text_encoder_path is None:
            raise ValueError("text_encoder_path required for most_similar_to_caption criteria")
        text_encoder = load_text_encoder(text_encoder_path, device, text_encoder_config)
        print(f"Loaded text encoder from: {text_encoder_path}")
    
    all_embeddings = []
    all_captions = []
    
    print(f"Processing images from: {coco_images_dir}")
    print(f"Output will be saved to: {output_file}")
    
    for i, annotation in enumerate(tqdm(annotations)):
        img_id = annotation["image_id"]
        
        # Determine filename based on format
        if isinstance(data, list):
            # Original format: construct filename from image_id
            filename = os.path.join(coco_images_dir, "train2014", f"COCO_train2014_{int(img_id):012d}.jpg")
            if not os.path.isfile(filename):
                filename = os.path.join(coco_images_dir, "val2014", f"COCO_val2014_{int(img_id):012d}.jpg")
        else:
            # Karpathy format: use filename from images mapping or construct it
            if img_id in image_id_to_filename:
                if 'train' in image_id_to_filename[img_id]:
                    fold = "train2014"
                else: 
                    fold = "val2014"
                filename = os.path.join(coco_images_dir, fold, image_id_to_filename[img_id])
            else:
                # Fallback: try to construct filename
                filename = os.path.join(coco_images_dir, "train2014", f"COCO_train2014_{int(img_id):012d}.jpg")
                if not os.path.isfile(filename):
                    filename = os.path.join(coco_images_dir, "val2014", f"COCO_val2014_{int(img_id):012d}.jpg")
        
        if not os.path.isfile(filename):
            print(f"Warning: Image not found: {filename}")
            continue
        
        # Load and process image
        try:
            image = io.imread(filename)
            if len(image.shape) == 2:  # grayscale
                image = Image.fromarray(image).convert('RGB')
            else:
                image = Image.fromarray(image)
        except Exception as e:
            print(f"Warning: Failed to load image {filename}: {e}")
            continue
        
        # Apply DINO transforms
        image_tensor = image_transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Clear any previous stored data
            global dino_layer_n_output, qkv_attention_out
            dino_layer_n_output = None
            qkv_attention_out = None
            
            # Extract DINO features
            if feature_type == "cls":
                # Standard CLS token extraction
                features = dino_model(image_tensor)
                # For DINOv2, the output is the CLS token by default
                if len(features.shape) == 3:  # If we get [batch, seq_len, dim]
                    features = features[:, 0, :]  # Take CLS token
                prefix = features.cpu()
            else:
                # For patch-based features, we need intermediate outputs
                _ = dino_model(image_tensor)  # Forward pass to trigger hooks
                
                if dino_layer_n_output is None:
                    raise RuntimeError("No intermediate output captured. Check hook registration.")
                
                # Transform to standard format
                dino_outs = transform_to_standard_dino_out(dino_layer_n_output, dino_model, num_global_tokens)
                
                if feature_type == "avg_patch":
                    # Average of patch tokens (excluding global tokens)
                    prefix = dino_outs['x_norm_patchtokens'].mean(dim=1)  # [B, D]
                elif feature_type == "avg_self_attn":
                    # Self-attention weighted average of patch tokens
                    if qkv_attention_out is None:
                        raise RuntimeError("No attention output captured. Check hook registration.")
                    
                    # Process self-attention to get attention weights
                    batch_size = qkv_attention_out.shape[0]
                    self_attn = process_self_attention(
                        qkv_attention_out, 
                        batch_size,
                        num_tokens, 
                        num_attn_heads, 
                        embed_dim,
                        scale, 
                        num_global_tokens
                    )
                    
                    # Compute attention-weighted average
                    prefix = (self_attn.unsqueeze(-1) * dino_outs['x_norm_patchtokens']).mean(dim=1)
                elif feature_type == "most_significant_patch":
                    # Select single most significant patch based on criteria
                    self_attn = None
                    cls_token = None
                    caption_embedding = None
                    
                    # Prepare required inputs based on criteria
                    if patch_selection_criteria in ["max_attention", "most_similar_to_caption"]:
                        if qkv_attention_out is None:
                            raise RuntimeError("No attention output captured. Check hook registration.")
                        batch_size = qkv_attention_out.shape[0]
                        self_attn = process_self_attention(
                            qkv_attention_out, 
                            batch_size,
                            num_tokens, 
                            num_attn_heads, 
                            embed_dim,
                            scale, 
                            num_global_tokens
                        )
                    
                    if patch_selection_criteria == "most_similar_to_cls":
                        cls_token = dino_outs['x_norm_clstoken']
                    
                    if patch_selection_criteria == "most_similar_to_caption":
                        if text_encoder is not None:
                            caption_embedding = encode_caption(annotation["caption"], text_encoder, device)
                    
                    # Select the most significant patch
                    prefix = select_most_significant_patch(
                        dino_outs, 
                        self_attn, 
                        patch_selection_criteria,
                        cls_token=cls_token,
                        caption_embedding=caption_embedding
                    )
                else:
                    raise ValueError(f"Unknown feature type: {feature_type}")
                
                prefix = prefix.cpu()
        
        # Create annotation in ClipCap format for compatibility
        caption_entry = {
            "image_id": img_id,
            "caption": annotation["caption"],
            "clip_embedding": i  # Index for the embedding
        }
        
        all_embeddings.append(prefix)
        all_captions.append(caption_entry)
        
        if (i + 1) % 10000 == 0:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'wb') as f:
                pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump({"clip_embedding": torch.cat(all_embeddings, dim=0), "captions": all_captions}, f)

    print('Done')
    print("%0d embeddings saved " % len(all_embeddings))
    print(f"Feature dimension: {all_embeddings[0].shape[-1]}")
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract DINO features from COCO images for ClipCap training')
    parser.add_argument('--dino_model_type', default="dinov2_vitb14", 
                       choices=('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14',
                               'dinov2_vits14_reg', 'dinov2_vitb14_reg', 'dinov2_vitl14_reg', 'dinov2_vitg14_reg'),
                       help='DINO model type to use for feature extraction')
    parser.add_argument('--feature_type', default="cls", 
                       choices=('cls', 'avg_patch', 'avg_self_attn', 'most_significant_patch'),
                       help='Type of features to extract: cls (CLS token), avg_patch (mean pooled patches), avg_self_attn (attention-weighted patches), most_significant_patch (single most important patch)')
    parser.add_argument('--patch_selection_criteria', default="max_attention",
                       choices=('max_attention', 'most_similar_to_cls', 'most_similar_to_caption', 'max_norm', 'centroid_distance'),
                       help='Criteria for selecting the most significant patch (only used with most_significant_patch feature_type)')
    parser.add_argument('--text_encoder_path', type=str, default=None,
                       help='Path to text encoder for caption similarity. Supports: 1) "dinotxt" or "dino.txt" for DINO.txt model, 2) Talk2Dino weights (.pth/.pt) - will auto-find config, 3) CLIP model names (e.g., "ViT-B/32")')
    parser.add_argument('--text_encoder_config', type=str, default=None,
                       help='Optional: explicit config path for Talk2Dino models (if not auto-found)')
    parser.add_argument('--resize_dim', type=int, default=518, help='Resize dimension for images')
    parser.add_argument('--crop_dim', type=int, default=518, help='Crop dimension for images')
    parser.add_argument('--coco_images_dir', type=str, default="/raid/datasets/coco", 
                       help='Path to COCO images directory (should contain train2014/ and val2014/ subdirs)')
    parser.add_argument('--captions_file', type=str, default="/raid/datasets/coco/train_split_karpathy.json",
                       help='Path to COCO captions JSON file (supports both Karpathy and ClipCap formats)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output pickle file path (default: auto-generated based on model and feature type)')
    parser.add_argument('--extract_attention', action='store_true',
                       help='Extract attention weights (automatically enabled for avg_self_attn feature type)')
    
    args = parser.parse_args()
    
    main(args.dino_model_type, args.resize_dim, args.crop_dim, 
         args.coco_images_dir, args.captions_file, args.output_file, 
         args.feature_type, args.extract_attention,
         args.patch_selection_criteria, args.text_encoder_path, args.text_encoder_config)