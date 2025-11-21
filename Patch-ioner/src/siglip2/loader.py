"""
SigLIP2 Model Loader

This module provides functions to load SigLIP2 models from HuggingFace checkpoints
using the local transformers.models.siglip2 code without requiring a transformers library update.

SigLIP2 is the newer version of SigLIP with improved vision-language alignment.
"""

import torch
from typing import Union, Tuple
import os
import sys


try:
    from transformers.models import Siglip2Model
    from transformers.models import Siglip2Config
    from transformers.models import Siglip2Processor
    from transformers.models import Siglip2ImageProcessor
    from transformers.models import Siglip2VisionModel
    from transformers.models import Siglip2TextModel

    from transformers.models import SiglipModel
    from transformers.models import SiglipVisionModel
    from transformers.models import SiglipTextModel
    from transformers.models import SiglipConfig
    from transformers.models import SiglipProcessor
    from transformers.models import SiglipImageProcessor
except ImportError:
    # Add the local siglip2 transformers code to the path
    _SIGLIP2_LOCAL_PATH = os.path.join(os.path.dirname(__file__), 'transfomers_siglip2')
    if _SIGLIP2_LOCAL_PATH not in sys.path:
        sys.path.insert(0, _SIGLIP2_LOCAL_PATH)

    # Import from local siglip2 code
    from modeling_siglip2 import Siglip2Model, Siglip2VisionModel, Siglip2TextModel
    from configuration_siglip2 import Siglip2Config
    from processing_siglip2 import Siglip2Processor
    from image_processing_siglip2 import Siglip2ImageProcessor


def needs_siglip2_classes(model_id: str) -> bool:
    return 'naflex' in model_id


def load_siglip2(
    config: Union[dict, str],
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    load_full_model: bool = False
) -> Union[Siglip2Model, Siglip2VisionModel]:
    """
    Load a SigLIP2 model from HuggingFace checkpoint using local siglip2 code.
    
    Args:
        config (Union[dict, str]): Configuration dictionary or HuggingFace model ID.
            If dict, should contain:
                - model_id (str): HuggingFace model ID (e.g., 'google/siglip2-base-patch16-512')
                - checkpoint_path (str, optional): Local checkpoint path (overrides model_id)
                - vision_only (bool, optional): Load only vision model. Defaults to True
                - patch_size (int, optional): Override patch size from config
                - embed_dim (int, optional): Override embedding dimension from config
                - image_size (int, optional): Override image size from config
            If str, treated as HuggingFace model ID
        device (Union[str, torch.device]): Device to load the model on.
        load_full_model (bool): If True, load the full Siglip2Model (vision + text).
                                If False, load only Siglip2VisionModel.
    
    Returns:
        Union[Siglip2Model, Siglip2VisionModel]: Loaded SigLIP2 model
    
    Raises:
        ValueError: If required parameters are missing from config
        RuntimeError: If model loading fails
    
    Example:
        # Load from HuggingFace
        config = {
            'model_id': 'google/siglip2-base-patch16-512',
            'vision_only': True,
            'image_size': 512
        }
        model = load_siglip2(config, device='cuda')
        
        # Or simply
        model = load_siglip2('google/siglip2-base-patch16-512', device='cuda')
    """
    
    # Handle string config (just model_id)
    if isinstance(config, str):
        config = {'model_id': config, 'vision_only': not load_full_model}
    
    # Extract parameters
    model_id = config.get('model_id', 'google/siglip2-base-patch16-512')
    checkpoint_path = config.get('checkpoint_path', None)
    vision_only = config.get('vision_only', not load_full_model)
    
    try:
        # Load from local checkpoint or HuggingFace
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading SigLIP2 model from local checkpoint: {checkpoint_path}")
            
            # Load config and model from local checkpoint
            siglip2_config = Siglip2Config.from_pretrained(checkpoint_path) if needs_siglip2_classes(model_id) else SiglipConfig.from_pretrained(checkpoint_path)
            
            if vision_only:
                model = Siglip2VisionModel.from_pretrained(checkpoint_path, config=siglip2_config) if needs_siglip2_classes(model_id) else SiglipVisionModel.from_pretrained(checkpoint_path, config=siglip2_config)
            else:
                model = Siglip2Model.from_pretrained(checkpoint_path, config=siglip2_config) if needs_siglip2_classes(model_id) else SiglipModel.from_pretrained(checkpoint_path, config=siglip2_config)
        else:
            print(f"Loading SigLIP2 model from HuggingFace: {model_id}")
            
            # Load config from HuggingFace
            siglip2_config = Siglip2Config.from_pretrained(model_id) if needs_siglip2_classes(model_id) else SiglipConfig.from_pretrained(model_id)

            if siglip2_config is None:
                raise ValueError(f"Failed to load SigLIP2 config for model ID: {model_id}")
            
            print(f"Loaded SigLIP2 config: {siglip2_config}")

            # Apply config overrides if provided
            if 'patch_size' in config:
                siglip2_config.vision_config.patch_size = config['patch_size']
            if 'embed_dim' in config:
                siglip2_config.vision_config.hidden_size = config['embed_dim']
            if 'image_size' in config:
                siglip2_config.vision_config.image_size = config['image_size']
            
            # Load the appropriate model
            if vision_only:
                model = Siglip2VisionModel.from_pretrained(model_id, config=siglip2_config.vision_config) if needs_siglip2_classes(model_id) else SiglipVisionModel.from_pretrained(model_id, config=siglip2_config.vision_config)
            else:
                print(f"Loading full SigLIP2 model (vision + text)")
                
                try:
                    model = Siglip2Model.from_pretrained(model_id, config=siglip2_config) if needs_siglip2_classes(model_id) else SiglipModel.from_pretrained(model_id, config=siglip2_config)
                except Exception as e:
                    raise e
                    Siglip2Model.config_class = Siglip2Config if needs_siglip2_classes(model_id) else SiglipConfig
                    model = Siglip2Model.from_pretrained(model_id, config=model_id) if needs_siglip2_classes(model_id) else SiglipModel.from_pretrained(model_id, config=model_id)
        
        # Move to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        print(f"Successfully loaded SigLIP2 {'vision' if vision_only else 'full'} model")
        print(f"  - Image size: {model.config.image_size if hasattr(model.config, 'image_size') else siglip2_config.vision_config.image_size}")
        print(f"  - Patch size: {model.config.patch_size if hasattr(model.config, 'patch_size') else siglip2_config.vision_config.patch_size}")
        print(f"  - Hidden size: {model.config.hidden_size if hasattr(model.config, 'hidden_size') else siglip2_config.vision_config.hidden_size}")
        
        return model
        
    except Exception as e:
        raise e
        raise RuntimeError(f"Failed to load SigLIP2 model: {str(e)}")


def load_siglip2_processor(
    config: Union[dict, str],
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[Union[Siglip2Model, Siglip2VisionModel], Siglip2Processor]:
    """
    Load a SigLIP2 model and its processor together.
    
    Args:
        config (Union[dict, str]): Configuration dictionary or HuggingFace model ID
        device (Union[str, torch.device]): Device to load the model on
    
    Returns:
        Tuple[Union[Siglip2Model, Siglip2VisionModel], Siglip2Processor]: 
            Loaded model and processor
    
    Example:
        model, processor = load_siglip2_processor('google/siglip2-base-patch16-512', device='cuda')
        
        # Process image
        from PIL import Image
        image = Image.open("path/to/image.jpg")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs.to(device))
    """
    
    # Handle string config
    if isinstance(config, str):
        model_id = config
        config = {'model_id': config}
    else:
        model_id = config.get('model_id', 'google/siglip2-base-patch16-512')
    
    # Load model
    model = load_siglip2(config, device)
    
    # Load processor
    checkpoint_path = config.get('checkpoint_path', None)
    if checkpoint_path and os.path.exists(checkpoint_path):
        processor = Siglip2Processor.from_pretrained(checkpoint_path) if needs_siglip2_classes(model_id) else SiglipProcessor.from_pretrained(checkpoint_path)
    else:
        processor = Siglip2Processor.from_pretrained(model_id) if needs_siglip2_classes(model_id) else SiglipProcessor.from_pretrained(model_id)
    
    return model, processor


def get_siglip2_tokenizer(
    config: Union[dict, str]
):
    """
    Get the tokenizer for SigLIP2 text encoding.
    
    Args:
        config (Union[dict, str]): Configuration dictionary or HuggingFace model ID
    
    Returns:
        Tokenizer function compatible with the training script
    
    Example:
        tokenizer = get_siglip2_tokenizer('google/siglip2-base-patch16-512')
        tokens = tokenizer(["a photo of a cat", "a photo of a dog"])
    """
    
    if isinstance(config, str):
        model_id = config
        checkpoint_path = None
    else:
        model_id = config.get('model_id', 'google/siglip2-base-patch16-512')
        checkpoint_path = config.get('checkpoint_path', None)
    
    # Load processor to get tokenizer
    if checkpoint_path and os.path.exists(checkpoint_path):
        processor = Siglip2Processor.from_pretrained(checkpoint_path) if needs_siglip2_classes(model_id) else SiglipProcessor.from_pretrained(checkpoint_path)
    else:
        processor = Siglip2Processor.from_pretrained(model_id) if needs_siglip2_classes(model_id) else SiglipProcessor.from_pretrained(model_id)
    
    # Return the tokenizer function
    return processor.tokenizer


def load_siglip2_config(config_name_or_path: str) -> dict:
    """
    Load SigLIP2 configuration from a config name or path.
    
    This function maintains compatibility with the existing config loading pattern
    used by other backbones (RegionCLIP, DenseCLIP, etc.).
    
    Args:
        config_name_or_path (str): Path to config file or HuggingFace model ID
    
    Returns:
        dict: Configuration dictionary with vision and text config details
    
    Example:
        config = load_siglip2_config('google/siglip2-base-patch16-512')
        print(f"Patch size: {config['vision']['patch_size']}")
        print(f"Hidden size: {config['vision']['hidden_size']}")
    """
    
    try:
        # Try loading as HuggingFace model ID or local path
        siglip2_config = Siglip2Config.from_pretrained(config_name_or_path) if needs_siglip2_classes(config_name_or_path) else SiglipConfig.from_pretrained(config_name_or_path)
        
        # Convert to dict format similar to other backbones
        config_dict = {
            'model_type': 'siglip2',
            'vision': {
                'hidden_size': siglip2_config.vision_config.hidden_size,
                'image_size': siglip2_config.vision_config.image_size,
                'patch_size': siglip2_config.vision_config.patch_size,
                'num_hidden_layers': siglip2_config.vision_config.num_hidden_layers,
                'num_attention_heads': siglip2_config.vision_config.num_attention_heads,
                'intermediate_size': siglip2_config.vision_config.intermediate_size,
                'num_channels': siglip2_config.vision_config.num_channels,
            },
            'text': {
                'hidden_size': siglip2_config.text_config.hidden_size,
                'vocab_size': siglip2_config.text_config.vocab_size,
                'max_position_embeddings': siglip2_config.text_config.max_position_embeddings,
                'num_hidden_layers': siglip2_config.text_config.num_hidden_layers,
                'num_attention_heads': siglip2_config.text_config.num_attention_heads,
                'intermediate_size': siglip2_config.text_config.intermediate_size,
            },
            'projection_dim': siglip2_config.projection_dim if hasattr(siglip2_config, 'projection_dim') else siglip2_config.vision_config.hidden_size,
        }
        
        return config_dict
        
    except Exception as e:
        raise ValueError(f"Failed to load SigLIP2 config from {config_name_or_path}: {str(e)}")


def siglip2_vision_forward_with_patches(
    model: Siglip2VisionModel,
    pixel_values: torch.FloatTensor,
    return_patches: bool = True,
    output_attentions: bool = False,
    output_hidden_states: bool = False,
    project_patches_using_mlp: bool = False,
    project_patches_using_attention_pooling_head_to_each_patch: bool = False
):
    """
    Forward pass for SigLIP2 vision model with option to return patch tokens.
    
    This function wraps the SigLIP2VisionModel forward method to handle the
    spatial_shapes parameter and optionally return patch tokens in a format
    compatible with other vision backbones (DenseCLIP, INViTE, etc.).
    
    Args:
        model (Siglip2VisionModel): The SigLIP2 vision model
        pixel_values (torch.FloatTensor): Input images of shape (batch_size, num_channels, height, width)
        return_patches (bool): If True, return all tokens (CLS + patches). If False, return only pooled output.
        output_attentions (bool): Whether to return attention weights
        output_hidden_states (bool): Whether to return hidden states
    
    Returns:
        torch.Tensor: 
            - If return_patches=True: Tensor of shape (batch_size, num_tokens, hidden_size) 
              where num_tokens = 1 + num_patches (CLS token + patch tokens)
            - If return_patches=False: Tensor of shape (batch_size, hidden_size) (pooled output)
    
    Example:
        model = load_siglip2('google/siglip2-base-patch16-512', device='cuda')
        output = siglip2_vision_forward_with_patches(model, images, return_patches=True)
        cls_token = output[:, 0, :]  # CLS token
        patch_tokens = output[:, 1:, :]  # Patch tokens
    """
    
    # Calculate spatial shapes from input images
    batch_size, num_channels, height, width = pixel_values.shape
    
    # Get patch size from model config
    patch_size = model.config.patch_size
    
    # Calculate number of patches in each dimension
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    
    # Create spatial_shapes tensor: (batch_size, 2) containing (height, width) for each image
    spatial_shapes = torch.tensor(
        [[num_patches_height, num_patches_width]] * batch_size,
        dtype=torch.long,
        device=pixel_values.device
    )
    
    # Create pixel_attention_mask (all ones since we're not masking anything)
    # Shape: (batch_size, num_patches_height * num_patches_width)
    num_patches = num_patches_height * num_patches_width
    pixel_attention_mask = torch.ones(
        (batch_size, num_patches),
        dtype=torch.long,
        device=pixel_values.device
    )
    
    # Forward pass through the model
    outputs = model(
        pixel_values=pixel_values,
        pixel_attention_mask=pixel_attention_mask,
        spatial_shapes=spatial_shapes,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states
    )
    
    if return_patches:
        # Return all tokens (CLS + patches)
        # SigLIP2 outputs: last_hidden_state has shape (batch_size, num_patches, hidden_size)
        # The first token is typically treated as the CLS token

        # we should apply the linear projection to the patch tokens to make them compatible with text
        projected_patches = (outputs.last_hidden_state)

        if project_patches_using_mlp:
            #projected_patches = model.vision_model.head.layernorm(projected_patches)
            #projected_patches = model.vision_model.head.mlp(projected_patches)
            residual = projected_patches
            projected_patches = model.vision_model.head.layernorm(projected_patches)
            projected_patches = residual + model.vision_model.head.mlp(projected_patches)
        elif project_patches_using_attention_pooling_head_to_each_patch:
            # use the attention pooling head to project each patch token
            # to do so, we reshape the tokens to (batch_size * num_patches, 1, hidden_size)
            batch_size, num_patches, hidden_size = projected_patches.shape
            projected_patches_reshaped = projected_patches.reshape(batch_size * num_patches, 1, hidden_size)
            #attention_mask_reshaped = torch.ones((batch_size * num_patches, 1), dtype=torch.long, device=pixel_values.device)
            pooler_output = model.vision_model.head(projected_patches_reshaped) # , attention_mask_reshaped
            # reshape back to (batch_size, num_patches, hidden_size)
            projected_patches = pooler_output.reshape(batch_size, num_patches, hidden_size)
        # we want to return also the CLS token, that is in outputs.
        cls_token = outputs.pooler_output.unsqueeze(1)  # Shape: (batch_size, 1, hidden_size)
        return cls_token, projected_patches
    else:
        # Return only pooled output (CLS-like representation)
        if outputs.pooler_output is not None:
            return outputs.pooler_output
        else:
            # Fallback: use the first token from last_hidden_state
            return outputs.last_hidden_state[:, 0, :]

