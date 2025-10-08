"""
Utility functions for HuggingFace Hub integration with decap-dino models.

This module provides functionality to download model weights from HuggingFace Hub
as a fallback when local checkpoint files are not available.
"""

import os
import torch
from typing import Optional, Union
from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import HfHubHTTPError
import logging

logger = logging.getLogger(__name__)


def get_model_path_with_hf_fallback(local_path: str,
    hf_repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    cache_dir: Optional[str] = None
) -> str:
    """
    Get the path to a model checkpoint, downloading from HuggingFace Hub if necessary.
    Args:
        local_path: Path to local checkpoint file
        hf_repo_id: HuggingFace repository ID (e.g., 'username/model-name')
        filename: Filename in the HF repository (if None, uses basename of local_path)
        cache_dir: Directory to cache downloaded files (if None, uses default HF cache)
    Returns:
        Path to the model checkpoint file
    Raises:
        FileNotFoundError: If neither local file nor HF repo is available
        Exception: If download fails
    """
    # Try to use the local path if it exists
    if os.path.exists(local_path):
        logger.info(f"Using local model path: {local_path}")
        return local_path

    # If local path doesn't exist, check if we have a HF repo ID to fall back on
    if hf_repo_id is None:
        raise FileNotFoundError(
            f"Local checkpoint not found at {local_path} and no hf_repo_id provided for fallback"
        )

    # Use basename of local_path as filename if not specified
    if filename is None:
        filename = os.path.basename(local_path)

    logger.info(f"Attempting to download from HuggingFace Hub: {hf_repo_id}/{filename}")

    try:
        # Check if the file exists in the repository
        api = HfApi()
        try:
            repo_files = api.list_repo_files(repo_id=hf_repo_id)
            if filename not in repo_files:
                raise FileNotFoundError(
                    f"File '{filename}' not found in HuggingFace repository '{hf_repo_id}'. "
                    f"Available files: {repo_files}"
                )
        except HfHubHTTPError as e:
            if e.response.status_code == 404:
                raise FileNotFoundError(
                    f"HuggingFace repository '{hf_repo_id}' not found or not accessible"
                )
            raise

        # Download the file
        downloaded_path = hf_hub_download(
            repo_id=hf_repo_id,
            filename=filename,
            cache_dir=cache_dir
        )

        logger.info(f"Successfully downloaded model from HF Hub to: {downloaded_path}")
        return downloaded_path

    except Exception as e:
        logger.error(f"Failed to download from HuggingFace Hub: {e}")
        raise


def load_model_with_hf_fallback(
    local_path: str,
    hf_repo_id: Optional[str] = None,
    filename: Optional[str] = None,
    map_location: Union[str, torch.device] = 'cpu',
    cache_dir: Optional[str] = None
) -> torch.Tensor:
    """
    Load a PyTorch model checkpoint with HuggingFace Hub fallback.
    
    Args:
        local_path: Path to local checkpoint file
        hf_repo_id: HuggingFace repository ID (e.g., 'username/model-name')
        filename: Filename in the HF repository (if None, uses basename of local_path)
        map_location: Device to map the tensors to
        cache_dir: Directory to cache downloaded files (if None, uses default HF cache)
        
    Returns:
        Loaded model state dict
        
    Raises:
        FileNotFoundError: If neither local file nor HF repo is available
        Exception: If download or loading fails
    """
    model_path = get_model_path_with_hf_fallback(
        local_path=local_path,
        hf_repo_id=hf_repo_id,
        filename=filename,
        cache_dir=cache_dir
    )
    
    try:
        state_dict = torch.load(model_path, map_location=map_location)
        logger.info(f"Successfully loaded model from: {model_path}")
        return state_dict
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {e}")
        raise
