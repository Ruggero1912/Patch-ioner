import torch
from typing import Union
from .clipfolder.clip import load as invite_clip_load, tokenize as invite_clip_tokenize


def load_invite_clip(config: dict, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    """
    Load an INViTE CLIP model based on the provided configuration.
    
    This method loads an INViTE CLIP model similar to how RegionCLIP is loaded in the Patchioner class.
    
    Args:
        config (dict): Configuration dictionary containing the following keys:
            - name (str): Model name listed by `clip.available_models()`, or path to a model checkpoint
            - jit (bool, optional): Whether to load the optimized JIT model. Defaults to False
            - download_root (str, optional): Path to download model files. Defaults to '/raid/datasets/models_weights/INViTE'
            - extract_last_k_th_token (int, optional): Extract last k-th token. Defaults to -1
            - viz (bool, optional): Visualization flag. Defaults to False
        device (Union[str, torch.device], optional): Device to load the model on. 
            Defaults to "cuda" if available, else "cpu"
    
    Returns:
        tuple: (model, preprocess_transform, tokenize_fn)
            - model: The loaded INViTE CLIP model
            - preprocess_transform: Torchvision transform for preprocessing images
            - tokenize_fn: Tokenization function for text processing
    
    Raises:
        KeyError: If required 'name' key is missing from config
        RuntimeError: If model loading fails
    
    Example:
        config = {
            'name': 'ViT-B/32',
            'jit': False,
            'download_root': '/raid/datasets/models_weights/INViTE',  # optional, this is the default
            'extract_last_k_th_token': -1,
            'viz': False
        }
        model, preprocess, tokenize = load_invite_clip(config, device='cuda')
    """
    
    # Validate required parameters
    if 'name' not in config:
        raise KeyError("'name' key is required in config dictionary")
    
    # Extract parameters with defaults
    name = config['name']
    jit = config.get('jit', False)
    download_root = config.get('download_root', '/raid/datasets/models_weights/INViTE')
    extract_last_k_th_token = config.get('extract_last_k_th_token', -1)
    viz = config.get('viz', False)

    image_resolution = config.get('resolution', None)  # Default resolution if not specified
    
    # Load the INViTE CLIP model using the clip.load function
    try:
        model, preprocess_transform = invite_clip_load(
            name=name,
            device=device,
            jit=jit,
            download_root=download_root,
            extract_last_k_th_token=extract_last_k_th_token,
            viz=viz,
            image_resolution=image_resolution
        )
        
        # Return model, preprocess transform, and tokenize function
        return model, preprocess_transform, invite_clip_tokenize
        
    except Exception as e:
        raise RuntimeError(f"Failed to load INViTE CLIP model '{name}': {str(e)}")
