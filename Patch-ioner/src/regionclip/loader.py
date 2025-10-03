from .clip_backbone import CLIP, convert_weights
import yaml
import os
import torch

"""
Use the method load_regionclip_from_checkpoint to load a RegionCLIP model from a checkpoint file.
This function will automatically handle the conversion of RegionCLIP-specific state_dict keys to the standard CLIP format.
It also allows you to specify a configuration file to set parameters like out_features and freeze_at.
"""


def load_regionclip_config(config_name):
    """
    Load RegionCLIP configuration from YAML file.
    
    Args:
        config_name (str): Name of the YAML configuration file (from the regionclip/configs directory)
    
    Returns:
        dict: Configuration dictionary
    """
    config_path = os.path.join(os.path.dirname(__file__), 'configs', config_name)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    #print(f"Loading RegionCLIP config from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    #print(f"Successfully loaded configuration")
    #print(f"   - Model architecture: {config.get('MODEL', {}).get('META_ARCHITECTURE', 'Unknown')}")
    #print(f"   - Backbone: {config.get('MODEL', {}).get('BACKBONE', {}).get('NAME', 'Unknown')}")
    #print(f"   - Freeze at: {config.get('MODEL', {}).get('BACKBONE', {}).get('FREEZE_AT', 'Unknown')}")

    return config

def load_regionclip_from_checkpoint(checkpoint_path, device='cpu', config=None, override_config=None):
    """
    Load CLIP model from a checkpoint file using build_model function.
    
    Args:
        checkpoint_path (str): Path to the .pth checkpoint file
        device (str): Device to load the model on ('cpu', 'cuda', etc.)
        config (dict | str): RegionCLIP configuration dictionary from YAML file or name of the config file.
                            If a string is provided, it will be loaded using load_regionclip_config.
        override_config (dict): Optional dictionary to override specific configuration parameters.
    
    Returns:
        CLIP model with loaded weights
    """

    if isinstance(config, str):
        # If config is a string, load it from the YAML file
        config = load_regionclip_config(config)
    
    if override_config:
        # Override specific configuration parameters if provided
        if config is None:
            config = {}
        # handle case of nested dictionaries
        def recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    recursive_update(d[k], v)
                else:
                    d[k] = v
        recursive_update(config, override_config)
    
    #print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract state_dict if it's wrapped in a checkpoint structure
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        #print("Found 'state_dict' in checkpoint")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        #print("Found 'model' in checkpoint")
    else:
        # Assume the checkpoint is directly the state_dict
        state_dict = checkpoint
        #print("Using checkpoint as state_dict directly")
    
    # Convert RegionCLIP format to standard CLIP format if needed
    if any(k.startswith('lang_encoder.') or k.startswith('backbone.') for k in state_dict.keys()):
        #print("Converting RegionCLIP format to standard CLIP format...")
        converted_state_dict = {}
        
        for key, value in state_dict.items():
            if key.startswith('lang_encoder.'):
                # Remove lang_encoder prefix for text encoder
                new_key = key.replace('lang_encoder.', '')
                converted_state_dict[new_key] = value
            elif key.startswith('backbone.'):
                # Convert backbone to visual
                new_key = key.replace('backbone.', 'visual.')
                converted_state_dict[new_key] = value
        
        clip_state_dict = converted_state_dict
        #print(f"Extracted {len(clip_state_dict)} CLIP-specific parameters")
    else:
        # Filter to only CLIP-related keys
        clip_keys = [k for k in state_dict.keys() if any(clip_prefix in k for clip_prefix in [
            'visual.', 'transformer.', 'token_embedding', 'positional_embedding', 
            'ln_final', 'text_projection', 'logit_scale'
        ])]
        
        if clip_keys:
            clip_state_dict = {k: state_dict[k] for k in clip_keys}
            #print(f"Extracted {len(clip_keys)} CLIP-specific parameters")
        else:
            # This checkpoint doesn't contain standard CLIP weights
            print("No CLIP weights found in this checkpoint")
            raise ValueError("No CLIP weights found in checkpoint")
    
    # Add missing logit_scale if not present
    if 'logit_scale' not in clip_state_dict:
        import numpy as np
        print("Adding missing logit_scale parameter")
        clip_state_dict['logit_scale'] = torch.ones([]) * np.log(1 / 0.07)
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    # Use build_model function directly with custom wrapper
    try:
        # Create a custom build_model that provides the missing parameters
        model = build_model_with_defaults(clip_state_dict, config)
        model.to(device)
        #print("Successfully created model using build_model()")
        return model
    except Exception as e:
        print(f"build_model() failed: {e}")
        raise e

def build_model_with_defaults(state_dict, config=None):
    """
    Wrapper around build_model that provides the required out_features and freeze_at parameters
    
    Args:
        state_dict: Model state dictionary
        config: RegionCLIP configuration dictionary from YAML file
    """
    
    # Get configuration parameters
    if config:
        model_config = config.get('MODEL', {})
        resnets_config = model_config.get('RESNETS', {})
        backbone_config = model_config.get('BACKBONE', {})
        
        # Extract configuration values
        out_features = resnets_config.get('OUT_FEATURES', ['res4'])
        freeze_at = backbone_config.get('FREEZE_AT', 0)

        depth = resnets_config.get('DEPTH', None)  # Optional depth parameter

        image_resolution = resnets_config.get('IMAGE_RESOLUTION', None)
        
        #print(f"Using config values - out_features: {out_features}, freeze_at: {freeze_at}")
    else:
        # Default values if no config is provided
        out_features = ['res4']
        freeze_at = 0
        
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        if image_resolution is None:
            image_resolution = output_width * 32
        else:
            if image_resolution / 32 != output_width:
                # The positional embedding is not compatible with the image resolution
                # Remove it from state_dict and let the model create a new one
                print(f"Warning: Removing incompatible positional embedding from checkpoint.")
                print(f"  Checkpoint spatial size: {output_width}x{output_width} (for image resolution {output_width * 32})")
                print(f"  Config image resolution: {image_resolution} (requires {image_resolution // 32}x{image_resolution // 32})")
                if "visual.attnpool.positional_embedding" in state_dict:
                    del state_dict["visual.attnpool.positional_embedding"]
                # Update output_width to match the config
                output_width = image_resolution // 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    # Create CLIP model with the required parameters
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        out_features=out_features,  # Use configuration parameter
        freeze_at=freeze_at,         # Use configuration parameter
        depth=depth                   # Use configuration parameter
    )

    # Clean up state_dict
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    
    # Load state dict with flexibility for missing or incompatible keys
    incompatible_keys = model.load_state_dict(state_dict, strict=False)
    
    if incompatible_keys.missing_keys:
        print(f"Note: Missing keys in checkpoint (will use model defaults): {incompatible_keys.missing_keys}")
    if incompatible_keys.unexpected_keys:
        print(f"Note: Unexpected keys in checkpoint (ignored): {incompatible_keys.unexpected_keys}")
    
    return model.eval()