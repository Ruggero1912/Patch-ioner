from .clip_loader.denseclip_loader import load_clip, load_config
from .clip_loader.denseclip_loader import DenseCLIPModel
from .clip_loader.tokenizer import tokenize as DenseCLIP_tokenize
import os

def load_denseclip(config_name: str, device: str = "cuda") -> DenseCLIPModel:
    """
    Load a DenseCLIP model.

    Args:
        model_name (str): The name of the DenseCLIP model to load.
        device (str): The device to load the model onto, default is "cuda".

    Returns:
        The loaded DenseCLIP model.
    """
    return load_clip(config_name=config_name, device=device)

def load_denseclip_config(config_name: str) -> dict:
    """
    Load the configuration for a DenseCLIP model.

    Args:
        config_name (str): The name of the DenseCLIP configuration to load.

    Returns:
        dict: The loaded configuration dictionary.
    """
    config_name = config_name + '.yaml' if not config_name.endswith('.yaml') else config_name
    config_path = os.path.join(os.path.dirname(__file__), 'clip_loader/configs', f'{config_name}')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"DenseClip configuration file {config_path} does not exist.")
    return load_config(config_path=config_path)