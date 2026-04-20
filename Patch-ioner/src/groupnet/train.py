"""
Training script for GroupNet on trace captioning task.

This script trains a GroupNet model to aggregate patch embeddings selected by traces
and match them to textual captions.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.transforms as T
import yaml

# Add parent directory to path for imports
# This allows us to import from decap/src when running from groupnet directory
decap_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(decap_root))
print(f"DEBUG: Added to sys.path: {decap_root}")
print(f"DEBUG: Checking if src/talk2dino exists: {(decap_root / 'src' / 'talk2dino').exists()}")



from model import GroupNet
from config_loader import load_config
from dataset import (
    TraceCaptioningDataset,
    trace_collate_fn,
    create_trace_captioning_dataloaders,
    map_trace_to_patches
)
print(f"DEBUG: Successfully imported GroupNet, load_config, and dataset functions.  ")

# Import ClipCap for captioning loss
try:
    import sys, os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    from src.clipcap.entrypoint import ClipCapModel, ClipCaptionModel, ClipCaptionPrefix, MappingType
    from transformers import GPT2Tokenizer
    CLIPCAP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ClipCap not available: {e}")
    CLIPCAP_AVAILABLE = False

# Import CIDEr metric for captioning evaluation
try:
    from speaksee.evaluation import Cider
    CIDER_AVAILABLE = True
except ImportError:
    try:
        from pycocoevalcap.cider.cider import Cider
        CIDER_AVAILABLE = True
    except ImportError:
        print("Warning: CIDEr metric not available. Install speaksee or pycocoevalcap for captioning metrics.")
        CIDER_AVAILABLE = False

class TextEncoder(nn.Module):
    """Wrapper for various text encoders (CLIP, Talk2DINO, etc.)"""
    
    def __init__(self, encoder_type: str, model_name: str, device: str = 'cuda'):
        super().__init__()
        self.encoder_type = encoder_type
        self.model_name = model_name
        self.device = device
        
        if encoder_type == 'clip':
            self._init_clip(model_name)
        elif encoder_type == 'open_clip':
            self._init_open_clip(model_name)
        elif encoder_type == 'talk2dino':
            self._init_talk2dino(model_name)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
    
    def _init_clip(self, model_name: str):
        """Initialize OpenAI CLIP text encoder"""
        import clip
        self.model, _ = clip.load(model_name, device=self.device)
        self.model.eval()
        self.embed_dim = self.model.ln_final.normalized_shape[0]
        
        # Tokenizer
        self.tokenize = clip.tokenize
    
    def _init_open_clip(self, model_name: str):
        """Initialize OpenCLIP text encoder"""
        from open_clip import create_model_and_transforms, get_tokenizer
        
        model, _, _ = create_model_and_transforms(
            model_name=model_name,
            pretrained="laion2b_s32b_b79k",
            device=self.device
        )
        model.eval()
        
        self.model = model
        self.embed_dim = model.text.text_projection.shape[1]
        self.tokenize = get_tokenizer(model_name.replace("/", "-"))
    
    def _init_talk2dino(self, config_path: str):
        """Initialize Talk2DINO text encoder (CLIP + Talk2DINO projection)"""
        import clip
        from src.talk2dino.talk2dino import ProjectionLayer
        
        # Load CLIP model first
        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
        self.clip_model.eval()
        
        # Load Talk2DINO projection layer
        self.talk2dino_projection = ProjectionLayer.from_config(config_path)
        self.talk2dino_projection.to(self.device)
        self.talk2dino_projection.eval()
        
        # Set embedding dimension (after projection to DINO space)
        self.embed_dim = self.talk2dino_projection.linear_layer.out_features
        
        # Tokenizer
        self.tokenize = clip.tokenize
    
    @torch.no_grad()
    def encode(self, texts):
        """
        Encode text to embeddings.
        
        Args:
            texts: List of text strings
        
        Returns:
            Tensor of shape [batch_size, embed_dim]
        """
        if self.encoder_type in ['clip', 'open_clip']:
            tokens = self.tokenize(texts, truncate=True).to(self.device)
            if self.encoder_type == 'clip':
                text_features = self.model.encode_text(tokens)
            else:  # open_clip
                text_features = self.model.encode_text(tokens)
            return text_features
        elif self.encoder_type == 'talk2dino':
            # Encode with CLIP then project to DINO space
            tokens = self.tokenize(texts, truncate=True).to(self.device)
            clip_features = self.clip_model.encode_text(tokens)
            # Project to DINO space using Talk2DINO
            dino_features = self.talk2dino_projection.project_clip_txt(clip_features)
            return dino_features


class VisualEncoder(nn.Module):
    """Wrapper for various visual encoders (CLIP, DINO, etc.)"""
    
    def __init__(self, encoder_type: str, model_name: str, device: str = 'cuda',
                 resize_dim: int = 518, crop_dim: int = 518):
        super().__init__()
        self.encoder_type = encoder_type
        self.model_name = model_name
        self.device = device
        self.resize_dim = resize_dim
        self.crop_dim = crop_dim
        
        if encoder_type == 'dino':
            self._init_dino(model_name)
        elif encoder_type == 'clip':
            self._init_clip(model_name)
        elif encoder_type == 'open_clip':
            self._init_open_clip(model_name)
        else:
            raise ValueError(f"Unknown visual encoder type: {encoder_type}")
        
        # Determine patch size and embedding dimension
        self._determine_patch_info()
    
    def _init_dino(self, model_name: str):
        """Initialize DINOv2 visual encoder"""
        if 'dinov2' in model_name:
            self.model = torch.hub.load('facebookresearch/dinov2', model_name)
            self.model.eval()
            self.model.to(self.device)
            
            # DINOv2 transforms
            self.transform = T.Compose([
                T.Resize(self.resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(self.crop_dim),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
        else:
            raise ValueError(f"Unknown DINO model: {model_name}")
    
    def _init_clip(self, model_name: str):
        """Initialize OpenAI CLIP visual encoder"""
        import clip
        self.model, self.transform = clip.load(model_name, device=self.device)
        self.model.eval()
    
    def _init_open_clip(self, model_name: str):
        """Initialize OpenCLIP visual encoder"""
        from open_clip import create_model_and_transforms
        
        model, _, preprocess = create_model_and_transforms(
            model_name=model_name,
            pretrained="laion2b_s32b_b79k",
            device=self.device
        )
        model.eval()
        
        self.model = model
        self.transform = preprocess
    
    def _determine_patch_info(self):
        """Determine patch size and embedding dimension"""
        if self.encoder_type == 'dino':
            if 'dinov2_vits14' in self.model_name:
                self.patch_size = 14
                self.embed_dim = 384
            elif 'dinov2_vitb14' in self.model_name:
                self.patch_size = 14
                self.embed_dim = 768
            elif 'dinov2_vitl14' in self.model_name:
                self.patch_size = 14
                self.embed_dim = 1024
            elif 'dinov2_vitg14' in self.model_name:
                self.patch_size = 14
                self.embed_dim = 1536
            else:
                raise ValueError(f"Unknown DINO model: {self.model_name}")
            
            # Check if model uses register tokens
            self.has_registers = 'reg' in self.model_name
        
        elif self.encoder_type in ['clip', 'open_clip']:
            # Determine from model name
            if 'ViT-B/32' in self.model_name or 'ViT-B-32' in self.model_name:
                self.patch_size = 32
                self.embed_dim = 768
            elif 'ViT-B/16' in self.model_name or 'ViT-B-16' in self.model_name:
                self.patch_size = 16
                self.embed_dim = 768
            elif 'ViT-L/14' in self.model_name or 'ViT-L-14' in self.model_name:
                self.patch_size = 14
                self.embed_dim = 1024
            elif 'ViT-H-14' in self.model_name:
                self.patch_size = 14
                self.embed_dim = 1280
            else:
                # Default
                self.patch_size = 16
                self.embed_dim = 768
            
            self.has_registers = False
        
        self.n_patches = self.crop_dim // self.patch_size
    
    @torch.no_grad()
    def encode(self, images):
        """
        Encode images to patch embeddings.
        
        Args:
            images: Tensor of shape [B, C, H, W]
        
        Returns:
            Tensor of shape [B, n_patches, n_patches, embed_dim]
        """
        if self.encoder_type == 'dino':
            output = self.model(images, is_training=True)
            patch_tokens = output['x_norm_patchtokens']  # [B, N, D]
            # Reshape to grid
            B = patch_tokens.shape[0]
            patch_tokens = patch_tokens.view(B, self.n_patches, self.n_patches, self.embed_dim)
            return patch_tokens
        
        elif self.encoder_type == 'clip':
            # Use CLIP visual encoder
            x = self.model.visual.conv1(images)
            x = x.reshape(x.shape[0], x.shape[1], -1)
            x = x.permute(0, 2, 1)
            x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
            x = x + self.model.visual.positional_embedding.to(x.dtype)
            x = self.model.visual.ln_pre(x)
            x = x.permute(1, 0, 2)
            x = self.model.visual.transformer(x)
            x = x.permute(1, 0, 2)
            
            # Extract patch tokens (skip CLS token)
            patch_tokens = x[:, 1:, :]
            B = patch_tokens.shape[0]
            patch_tokens = patch_tokens.view(B, self.n_patches, self.n_patches, self.embed_dim)
            return patch_tokens
        
        elif self.encoder_type == 'open_clip':
            # Use OpenCLIP visual encoder
            x = self.model.visual(images)
            # Extract patches (implementation depends on OpenCLIP version)
            # For now, use a simplified approach
            # TODO: Implement proper patch extraction for OpenCLIP
            raise NotImplementedError("OpenCLIP patch extraction not yet fully implemented")


def compute_cross_entropy_loss(
    patch_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Compute cross-entropy loss between patch and text embeddings.
    
    Args:
        patch_embeddings: Tensor of shape [batch_size, embed_dim]
        text_embeddings: Tensor of shape [batch_size, embed_dim]
        temperature: Temperature for scaling logits
    
    Returns:
        Loss scalar
    """
    # Normalize embeddings
    patch_embeddings = F.normalize(patch_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(patch_embeddings, text_embeddings.t()) / temperature
    
    # Labels: diagonal elements (i-th patch should match i-th text)
    labels = torch.arange(len(logits), device=logits.device)
    
    # Cross-entropy loss (symmetric: patch-to-text and text-to-patch)
    loss_p2t = F.cross_entropy(logits, labels)
    loss_t2p = F.cross_entropy(logits.t(), labels)
    loss = (loss_p2t + loss_t2p) / 2
    
    return loss


def compute_infonce_loss(
    patch_embeddings: torch.Tensor,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07,
    use_hard_negatives: bool = False
) -> torch.Tensor:
    """
    Compute InfoNCE (contrastive) loss.
    
    Args:
        patch_embeddings: Tensor of shape [batch_size, embed_dim]
        text_embeddings: Tensor of shape [batch_size, embed_dim]
        temperature: Temperature for scaling
        use_hard_negatives: Whether to use hard negative mining
    
    Returns:
        Loss scalar
    """
    # Normalize
    patch_embeddings = F.normalize(patch_embeddings, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Compute similarity
    logits = torch.matmul(patch_embeddings, text_embeddings.t()) / temperature
    
    # InfoNCE loss
    labels = torch.arange(len(logits), device=logits.device)
    loss = F.cross_entropy(logits, labels)
    
    return loss


def compute_centroid_baseline(
    patch_embeddings_list: list,
    text_embeddings: torch.Tensor,
    temperature: float = 0.07
) -> Dict[str, float]:
    """
    Compute metrics for simple centroid (mean) baseline.
    
    This is useful when training ResidualCentroidGroupNet to compare
    against the baseline that the model starts with (residual = 0).
    
    Args:
        patch_embeddings_list: List of patch embeddings, one per sample
                               Each element is [num_patches, embed_dim]
        text_embeddings: Text embeddings [batch_size, embed_dim]
        temperature: Temperature for scaling logits
    
    Returns:
        Dictionary with 'acc' (top-1) and 'top5_acc' (top-5) accuracy
    """
    # Compute centroids (mean over patches)
    centroids = []
    for patches in patch_embeddings_list:
        centroid = patches.mean(dim=0)  # [embed_dim]
        centroids.append(centroid)
    
    centroids = torch.stack(centroids, dim=0)  # [batch_size, embed_dim]
    
    # Normalize
    centroids = F.normalize(centroids, dim=-1)
    text_embeddings = F.normalize(text_embeddings, dim=-1)
    
    # Compute similarity
    logits = torch.matmul(centroids, text_embeddings.t()) / temperature
    
    # Top-1 accuracy
    preds = logits.argmax(dim=1)
    labels = torch.arange(len(preds), device=logits.device)
    acc = (preds == labels).float().mean().item()
    
    # Top-5 accuracy
    _, top5_preds = logits.topk(5, dim=1)
    top5_acc = (top5_preds == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    
    return {'acc': acc, 'top5_acc': top5_acc}


def train_one_epoch(
    model: GroupNet,
    visual_encoder: VisualEncoder,
    text_encoder: Optional[TextEncoder],
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    epoch: int,
    loss_type: str = 'cross_entropy',
    temperature: float = 0.07,
    writer: Optional[SummaryWriter] = None,
    is_master: bool = True,
    distributed: bool = False,
    clipcap_model: Optional['ClipCapModel'] = None
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        clipcap_model: Optional ClipCap model for captioning loss
        text_encoder: Optional text encoder (not used when loss_type='captioning')
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_master, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        try:
            images = batch['images'].to(device)
            traces = batch['traces']
            captions = batch['captions']
            
            optimizer.zero_grad()
            
            # Encode images to patches
            patch_embeddings_grid = visual_encoder.encode(images)  # [B, n_patches, n_patches, embed_dim]
            
            # Extract patches for each trace and aggregate with GroupNet
            batch_aggregated = []
            for i, trace in enumerate(traces):
                # Get patches for this trace
                trace_patches = map_trace_to_patches(
                    trace, 
                    patch_embeddings_grid[i], 
                    visual_encoder.n_patches
                )  # [num_selected_patches, embed_dim]
                
                # Aggregate with GroupNet
                aggregated = model(trace_patches.unsqueeze(0).to(device))  # [1, embed_dim]
                batch_aggregated.append(aggregated)
            
            # Stack aggregated embeddings
            patch_embeddings = torch.cat(batch_aggregated, dim=0)  # [B, embed_dim]
            
            # Compute loss based on type
            if loss_type == 'captioning':
                # Use ClipCap captioning loss
                if clipcap_model is None:
                    raise ValueError("clipcap_model must be provided when using loss_type='captioning'")
                
                try:
                    loss = clipcap_model.compute_captioning_loss(patch_embeddings, captions)
                except RuntimeError as e:
                    if "too long" in str(e).lower():
                        print(f"\nWarning: Skipping batch {batch_idx} - caption too long")
                        # Dummy backward for DDP sync
                        try:
                            p = next(model.parameters())
                            dummy_loss = p.sum() * 0.0
                        except StopIteration:
                            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)
                        dummy_loss.backward()
                        optimizer.step()
                        continue
                    else:
                        raise
            else:
                # Use contrastive losses (cross_entropy or infonce)
                if text_encoder is None:
                    raise ValueError("text_encoder must be provided when using contrastive losses")
                
                # Encode captions (with error handling for overly long captions)
                try:
                    text_embeddings = text_encoder.encode(captions)  # [B, embed_dim]
                except RuntimeError as e:
                    if "too long for context length" in str(e):
                        print(f"\nWarning: Skipping batch {batch_idx} - caption too long")
                        # To keep DDP gradients/collectives in sync across ranks, perform a dummy backward
                        # so every rank participates in the gradient all-reduce. Create a zero loss that
                        # depends on model parameters to produce zero gradients.
                        try:
                            p = next(model.parameters())
                            dummy_loss = p.sum() * 0.0
                        except StopIteration:
                            dummy_loss = torch.tensor(0.0, device=device, requires_grad=True)

                        # Backward and optimizer step to keep DDP in sync
                        dummy_loss.backward()
                        optimizer.step()
                        # Do not include this batch in metrics
                        continue
                    else:
                        raise
                
                # Compute contrastive loss
                if loss_type == 'cross_entropy':
                    loss = compute_cross_entropy_loss(patch_embeddings, text_embeddings, temperature)
                elif loss_type == 'infonce':
                    loss = compute_infonce_loss(patch_embeddings, text_embeddings, temperature)
                else:
                    raise ValueError(f"Unknown loss type: {loss_type}")
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # Compute accuracy (only for contrastive losses)
            if loss_type in ['cross_entropy', 'infonce']:
                with torch.no_grad():
                    patch_emb_norm = F.normalize(patch_embeddings, dim=-1)
                    text_emb_norm = F.normalize(text_embeddings, dim=-1)
                    logits = torch.matmul(patch_emb_norm, text_emb_norm.t())
                    preds = logits.argmax(dim=1)
                    acc = (preds == torch.arange(len(preds), device=device)).float().mean()
            else:
                # For captioning loss, we don't have a simple accuracy metric
                # Just set acc to 0 or could use perplexity
                acc = torch.tensor(0.0, device=device)
            
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1
            
        except Exception as e:
            # Unexpected error: in distributed mode fail fast so other ranks don't hang on collectives
            print(f"\nError in batch {batch_idx}: {e}")
            import traceback
            traceback.print_exc()
            if distributed and dist.is_initialized():
                try:
                    # try to gracefully destroy process group
                    dist.destroy_process_group()
                except Exception:
                    pass
                # exit immediately
                os._exit(1)
            else:
                print(f"Skipping batch and continuing...")
                continue
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / num_batches,
            'acc': total_acc / num_batches
        })
        
        # Log to tensorboard
        if writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('train/loss', loss.item(), global_step)
            writer.add_scalar('train/acc', acc.item(), global_step)
    
    # If no batches were processed, return zeros
    if num_batches == 0:
        print("Warning: No batches were processed in this epoch; returning zeros")
        return {'loss': 0.0, 'acc': 0.0}

    # Aggregate across processes if distributed
    if distributed and dist.is_initialized():
        try:
            tensor = torch.tensor([total_loss, total_acc, num_batches], device=device, dtype=torch.float64)
            # Debug: optionally print rank/tensor before collective
            if os.environ.get('DDEBUG') == '1' or is_master:
                print(f"[rank {dist.get_rank()}] all_reduce send tensor: {tensor.cpu().tolist()}")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if os.environ.get('DDEBUG') == '1' or is_master:
                print(f"[rank {dist.get_rank()}] all_reduce recv tensor: {tensor.cpu().tolist()}")

            total_loss_sum = float(tensor[0].item())
            total_acc_sum = float(tensor[1].item())
            total_batches_sum = float(tensor[2].item())

            # Guard against division by zero if all ranks had zero batches
            if total_batches_sum == 0:
                if is_master:
                    print("Warning: total_batches_sum == 0 after all_reduce; returning zeros")
                return {'loss': 0.0, 'acc': 0.0}

            return {
                'loss': total_loss_sum / total_batches_sum,
                'acc': total_acc_sum / total_batches_sum
            }
        except Exception as e:
            # If collective fails (e.g., NCCL timeout or a rank crashed), handle gracefully
            print(f"Distributed metric aggregation failed on rank {dist.get_rank()}: {e}")
            # Fallback to local metrics (best-effort)
            return {
                'loss': total_loss / max(1, num_batches),
                'acc': total_acc / max(1, num_batches)
            }
    else:
        return {
            'loss': total_loss / num_batches,
            'acc': total_acc / num_batches
        }


@torch.no_grad()
def validate(
    model: GroupNet,
    visual_encoder: VisualEncoder,
    text_encoder: Optional[TextEncoder],
    val_loader: DataLoader,
    device: str,
    loss_type: str = 'cross_entropy',
    temperature: float = 0.07,
    is_master: bool = True,
    distributed: bool = False,
    compute_centroid_metrics: bool = False,
    clipcap_model: Optional['ClipCapModel'] = None,
    compute_cider: bool = False,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    num_qualitative_samples: int = 10
) -> Dict[str, float]:
    """
    Validate the model.
    
    Args:
        compute_centroid_metrics: If True, also compute and return centroid baseline metrics
                                  Useful for ResidualCentroidGroupNet models
        clipcap_model: Optional ClipCap model for captioning loss
        text_encoder: Optional text encoder (not used when loss_type='captioning')
        compute_cider: If True, generate captions and compute CIDEr score (only for captioning loss)
        writer: TensorBoard writer for logging qualitative samples
        epoch: Current epoch number for logging
        num_qualitative_samples: Number of qualitative samples to log to TensorBoard
    
    Returns:
        Dictionary with validation metrics (and optionally centroid baseline metrics and CIDEr)
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_top5_acc = 0.0
    num_batches = 0
    
    # For centroid baseline
    total_centroid_acc = 0.0
    total_centroid_top5_acc = 0.0
    
    # For CIDEr computation
    all_generated_captions = {}
    all_reference_captions = {}
    sample_counter = 0
    
    # For qualitative logging
    qualitative_samples = []
    
    # Note: caller should pass is_master to control progress printing; default True for compatibility
    pbar = tqdm(val_loader, desc="Validation", disable=not is_master, dynamic_ncols=True)
    for batch_idx, batch in enumerate(pbar):
        images = batch['images'].to(device)
        traces = batch['traces']
        captions = batch['captions']
        
        # Encode images to patches
        patch_embeddings_grid = visual_encoder.encode(images)
        
        # Extract and aggregate patches for each trace
        batch_aggregated = []
        batch_trace_patches = []  # Store for centroid baseline
        for i, trace in enumerate(traces):
            trace_patches = map_trace_to_patches(
                trace,
                patch_embeddings_grid[i],
                visual_encoder.n_patches
            )
            batch_trace_patches.append(trace_patches)
            aggregated = model(trace_patches.unsqueeze(0).to(device))
            batch_aggregated.append(aggregated)
        
        patch_embeddings = torch.cat(batch_aggregated, dim=0)
        
        # Compute loss based on type
        if loss_type == 'captioning':
            # Use ClipCap captioning loss
            if clipcap_model is None:
                raise ValueError("clipcap_model must be provided when using loss_type='captioning'")
            
            try:
                loss = clipcap_model.compute_captioning_loss(patch_embeddings, captions)
            except RuntimeError as e:
                if "too long" in str(e).lower():
                    print(f"\nWarning: Skipping batch {batch_idx} - caption too long")
                    continue
                else:
                    raise
            
            # Generate captions if CIDEr computation is requested or for qualitative samples
            if compute_cider or (len(qualitative_samples) < num_qualitative_samples and is_master):
                try:
                    generated_captions = clipcap_model.forward(patch_embeddings, compute_scores=False)
                    
                    # Store for CIDEr computation
                    if compute_cider:
                        for i, (gen_cap, ref_cap) in enumerate(zip(generated_captions, captions)):
                            sample_id = f"{batch_idx}_{i}"
                            all_generated_captions[sample_id] = [gen_cap]
                            all_reference_captions[sample_id] = [ref_cap]
                    
                    # Collect qualitative samples for TensorBoard
                    if len(qualitative_samples) < num_qualitative_samples and is_master:
                        for i in range(min(len(generated_captions), num_qualitative_samples - len(qualitative_samples))):
                            qualitative_samples.append({
                                'image': images[i].cpu(),
                                'generated': generated_captions[i],
                                'reference': captions[i],
                                'batch_idx': batch_idx
                            })
                except Exception as e:
                    print(f"\nWarning: Failed to generate captions for batch {batch_idx}: {e}")
            
            # For captioning, no contrastive accuracy
            acc = torch.tensor(0.0, device=device)
            top5_acc = torch.tensor(0.0, device=device)
        else:
            # Use contrastive losses
            if text_encoder is None:
                raise ValueError("text_encoder must be provided when using contrastive losses")
            
            # Encode captions
            try:
                text_embeddings = text_encoder.encode(captions)
            except RuntimeError as e:
                if "too long for context length" in str(e):
                    print(f"\nWarning: Skipping batch {batch_idx} - caption too long")
                    continue
                else:
                    # Unexpected runtime error: fail fast in distributed mode
                    print(f"Error encoding captions on batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    if distributed and dist.is_initialized():
                        try:
                            dist.destroy_process_group()
                        except Exception:
                            pass
                        os._exit(1)
                    else:
                        raise

            # Compute loss
            if loss_type == 'cross_entropy':
                loss = compute_cross_entropy_loss(patch_embeddings, text_embeddings, temperature)
            elif loss_type == 'infonce':
                loss = compute_infonce_loss(patch_embeddings, text_embeddings, temperature)
            else:
                raise ValueError(f"Unknown loss type: {loss_type}")
            
            # Compute accuracy
            patch_emb_norm = F.normalize(patch_embeddings, dim=-1)
            text_emb_norm = F.normalize(text_embeddings, dim=-1)
            logits = torch.matmul(patch_emb_norm, text_emb_norm.t())
            
            # Top-1 accuracy
            preds = logits.argmax(dim=1)
            acc = (preds == torch.arange(len(preds), device=device)).float().mean()
            
            # Top-5 accuracy
            _, top5_preds = logits.topk(min(5, len(logits)), dim=1)
            labels = torch.arange(len(preds), device=device).unsqueeze(1)
            top5_acc = (top5_preds == labels).any(dim=1).float().mean()
        
        total_loss += loss.item()
        total_acc += acc.item()
        total_top5_acc += top5_acc.item()
        num_batches += 1
        
        # Compute centroid baseline metrics if requested (only for contrastive losses)
        if compute_centroid_metrics and loss_type in ['cross_entropy', 'infonce']:
            centroid_metrics = compute_centroid_baseline(
                batch_trace_patches, 
                text_embeddings, 
                temperature
            )
            total_centroid_acc += centroid_metrics['acc']
            total_centroid_top5_acc += centroid_metrics['top5_acc']
        
        postfix = {
            'loss': total_loss / num_batches,
            'acc': total_acc / num_batches,
            'top5_acc': total_top5_acc / num_batches
        }
        if compute_centroid_metrics:
            postfix['cent_acc'] = total_centroid_acc / num_batches
        pbar.set_postfix(postfix)
    
    # If no batches processed, return zeros
    if num_batches == 0:
        result = {'loss': 0.0, 'acc': 0.0, 'top5_acc': 0.0}
        if compute_centroid_metrics:
            result['centroid_acc'] = 0.0
            result['centroid_top5_acc'] = 0.0
        return result

    # Aggregate across processes if distributed
    if distributed and dist.is_initialized():
        try:
            if compute_centroid_metrics:
                tensor = torch.tensor([total_loss, total_acc, total_top5_acc, 
                                     total_centroid_acc, total_centroid_top5_acc, num_batches], 
                                    device=device, dtype=torch.float64)
            else:
                tensor = torch.tensor([total_loss, total_acc, total_top5_acc, num_batches], 
                                    device=device, dtype=torch.float64)
            
            if os.environ.get('DDEBUG') == '1' or is_master:
                print(f"[rank {dist.get_rank()}] val all_reduce send tensor: {tensor.cpu().tolist()}")
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            if os.environ.get('DDEBUG') == '1' or is_master:
                print(f"[rank {dist.get_rank()}] val all_reduce recv tensor: {tensor.cpu().tolist()}")

            if compute_centroid_metrics:
                total_loss_sum = float(tensor[0].item())
                total_acc_sum = float(tensor[1].item())
                total_top5_sum = float(tensor[2].item())
                total_centroid_acc_sum = float(tensor[3].item())
                total_centroid_top5_sum = float(tensor[4].item())
                total_batches_sum = float(tensor[5].item())
            else:
                total_loss_sum = float(tensor[0].item())
                total_acc_sum = float(tensor[1].item())
                total_top5_sum = float(tensor[2].item())
                total_batches_sum = float(tensor[3].item())

            if total_batches_sum == 0:
                if is_master:
                    print("Warning: total_batches_sum == 0 after val all_reduce; returning zeros")
                result = {'loss': 0.0, 'acc': 0.0, 'top5_acc': 0.0}
                if compute_centroid_metrics:
                    result['centroid_acc'] = 0.0
                    result['centroid_top5_acc'] = 0.0
                return result

            result = {
                'loss': total_loss_sum / total_batches_sum,
                'acc': total_acc_sum / total_batches_sum,
                'top5_acc': total_top5_sum / total_batches_sum
            }
            if compute_centroid_metrics:
                result['centroid_acc'] = total_centroid_acc_sum / total_batches_sum
                result['centroid_top5_acc'] = total_centroid_top5_sum / total_batches_sum
            
            # Compute CIDEr (only on master in distributed setting)
            if compute_cider and loss_type == 'captioning' and is_master:
                if not CIDER_AVAILABLE:
                    print("Warning: CIDEr computation requested but metric not available. Skipping.")
                elif len(all_generated_captions) > 0:
                    try:
                        cider_scorer = Cider()
                        cider_score, _ = cider_scorer.compute_score(all_reference_captions, all_generated_captions)
                        result['cider'] = float(cider_score)
                        print(f"\nCIDEr score: {cider_score:.4f}")
                    except Exception as e:
                        print(f"Warning: Failed to compute CIDEr score: {e}")
                        result['cider'] = 0.0
            
            # Log qualitative samples to TensorBoard (only on master)
            if writer is not None and is_master and len(qualitative_samples) > 0:
                try:
                    import torchvision
                    for idx, sample in enumerate(qualitative_samples):
                        # Denormalize image for visualization
                        img = sample['image']
                        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                        img = img * std + mean
                        img = torch.clamp(img, 0, 1)
                        
                        caption_text = f"Generated: {sample['generated']}\nReference: {sample['reference']}"
                        
                        writer.add_image(f'val_samples/{idx}', img, global_step=epoch, dataformats='CHW')
                        writer.add_text(f'val_captions/{idx}', caption_text, global_step=epoch)
                except Exception as e:
                    print(f"Warning: Failed to log qualitative samples to TensorBoard: {e}")
            
            return result
        except Exception as e:
            print(f"Distributed validation aggregation failed on rank {dist.get_rank()}: {e}")
            result = {
                'loss': total_loss / max(1, num_batches),
                'acc': total_acc / max(1, num_batches),
                'top5_acc': total_top5_acc / max(1, num_batches)
            }
            if compute_centroid_metrics:
                result['centroid_acc'] = total_centroid_acc / max(1, num_batches)
                result['centroid_top5_acc'] = total_centroid_top5_acc / max(1, num_batches)
            return result
    else:
        result = {
            'loss': total_loss / num_batches,
            'acc': total_acc / num_batches,
            'top5_acc': total_top5_acc / num_batches
        }
        if compute_centroid_metrics:
            result['centroid_acc'] = total_centroid_acc / num_batches
            result['centroid_top5_acc'] = total_centroid_top5_acc / num_batches
    
    # Compute CIDEr score if requested (only for captioning loss)
    if compute_cider and loss_type == 'captioning' and is_master:
        if not CIDER_AVAILABLE:
            print("Warning: CIDEr computation requested but metric not available. Skipping.")
        elif len(all_generated_captions) > 0:
            try:
                cider_scorer = Cider()
                cider_score, _ = cider_scorer.compute_score(all_reference_captions, all_generated_captions)
                result['cider'] = float(cider_score)
                print(f"\nCIDEr score: {cider_score:.4f}")
            except Exception as e:
                print(f"Warning: Failed to compute CIDEr score: {e}")
                result['cider'] = 0.0
        else:
            print("Warning: No captions generated for CIDEr computation")
            result['cider'] = 0.0
    
    # Log qualitative samples to TensorBoard
    if writer is not None and is_master and len(qualitative_samples) > 0:
        try:
            import torchvision
            for idx, sample in enumerate(qualitative_samples):
                # Denormalize image for visualization
                img = sample['image']
                # Standard ImageNet normalization used by most models
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                img = torch.clamp(img, 0, 1)
                
                # Create caption text
                caption_text = f"Generated: {sample['generated']}\nReference: {sample['reference']}"
                
                # Log to TensorBoard
                writer.add_image(
                    f'val_samples/{idx}',
                    img,
                    global_step=epoch,
                    dataformats='CHW'
                )
                writer.add_text(
                    f'val_captions/{idx}',
                    caption_text,
                    global_step=epoch
                )
        except Exception as e:
            print(f"Warning: Failed to log qualitative samples to TensorBoard: {e}")
    
    return result


def main(args):
    # Distributed setup
    distributed = False
    # Prefer command-line --local_rank, but fall back to environment variables (torchrun sets LOCAL_RANK / WORLD_SIZE)
    local_rank = getattr(args, 'local_rank', None)
    env_local_rank = os.environ.get('LOCAL_RANK')
    env_world_size = os.environ.get('WORLD_SIZE')

    if local_rank is None and env_local_rank is not None:
        try:
            local_rank = int(env_local_rank)
        except Exception:
            local_rank = None

    # If either local_rank was provided or WORLD_SIZE indicates >1, enable distributed
    if local_rank is not None:
        distributed = True
    elif env_world_size is not None:
        try:
            if int(env_world_size) > 1:
                distributed = True
                # ensure we have a local rank (default to 0)
                local_rank = int(env_local_rank) if env_local_rank is not None else 0
        except Exception:
            distributed = False

    if distributed:
        # Initialize process group (torchrun/torch.distributed will provide rendezvous info)
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Ensure local_rank is an int and set device accordingly
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_master = (rank == 0)
        if is_master:
            print(f"Distributed training initialized: rank {rank}/{world_size}, device {device}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        is_master = True
        print(f"Using device: {device}")
    
    # Load GroupNet config
    groupnet_config = load_config(args.groupnet_config)

    if not distributed or (distributed and dist.is_initialized() and is_master):
        print(f"GroupNet config: {groupnet_config}")
    
    # Initialize visual encoder
        print(f"Loading visual encoder: {args.visual_encoder_type} - {args.visual_encoder_name}")
    visual_encoder = VisualEncoder(
        encoder_type=args.visual_encoder_type,
        model_name=args.visual_encoder_name,
        device=device,
        resize_dim=args.resize_dim,
        crop_dim=args.crop_dim
    )
    visual_encoder.model.eval()  # Keep visual encoder frozen
    
    # Initialize text encoder and/or ClipCap model based on loss type
    text_encoder = None
    clipcap_model = None
    
    if args.loss_type == 'captioning':
        # Use ClipCap for captioning loss
        if not CLIPCAP_AVAILABLE:
            raise ValueError("ClipCap is not available. Cannot use loss_type='captioning'")
        if args.clipcap_config is None or args.clipcap_weights is None:
            raise ValueError("Must provide --clipcap_config and --clipcap_weights when using loss_type='captioning'")
        
        if not distributed or (distributed and dist.is_initialized() and is_master):
            print(f"Loading ClipCap model for captioning loss")
            print(f"  Config: {args.clipcap_config}")
            print(f"  Weights: {args.clipcap_weights}")
        
        # Load ClipCap config
        with open(args.clipcap_config, 'r') as f:
            clipcap_config = yaml.safe_load(f)
        
        # Override weight_path with command line argument
        clipcap_config['weight_path'] = args.clipcap_weights
        
        # Initialize ClipCap model
        clipcap_model = ClipCapModel(
            args=clipcap_config,
            device=device,
            dino_feature_dim=visual_encoder.embed_dim
        )
        
        # Freeze or unfreeze ClipCap
        if args.freeze_clipcap:
            clipcap_model.model.eval()
            for param in clipcap_model.model.parameters():
                param.requires_grad = False
            if not distributed or (distributed and dist.is_initialized() and is_master):
                print("ClipCap model frozen")
        else:
            clipcap_model.model.train()
            if not distributed or (distributed and dist.is_initialized() and is_master):
                print("ClipCap model trainable")
    else:
        # Initialize text encoder for contrastive losses
        if not distributed or (distributed and dist.is_initialized() and is_master):
            print(f"Loading text encoder: {args.text_encoder_type} - {args.text_encoder_name}")
        
        # Handle Talk2DINO config path
        text_encoder_config = args.text_encoder_name
        if args.text_encoder_type == 'talk2dino':
            # Use talk2dino_config if provided, otherwise use text_encoder_name
            text_encoder_config = args.talk2dino_config or args.text_encoder_name
            if text_encoder_config is None:
                # Default Talk2DINO config path (adjust as needed)
                text_encoder_config = '/path/to/default/talk2dino/config.yaml'
                print(f"Warning: No Talk2DINO config specified, using default: {text_encoder_config}")
        elif args.text_encoder_name is None:
            # Default for CLIP models
            text_encoder_config = 'ViT-B/16'
        
        text_encoder = TextEncoder(
            encoder_type=args.text_encoder_type,
            model_name=text_encoder_config,
            device=device
        )
        text_encoder.eval()  # Keep text encoder frozen
        if hasattr(text_encoder, 'clip_model'):
            text_encoder.clip_model.eval()
        if hasattr(text_encoder, 'talk2dino_projection'):
            text_encoder.talk2dino_projection.eval()
    
    # Update GroupNet config with correct embedding dimension
    groupnet_config['embed_dim'] = visual_encoder.embed_dim
    
    # Initialize GroupNet
    if not distributed or (distributed and dist.is_initialized() and is_master):
        print(f"Initializing GroupNet with embed_dim={groupnet_config['embed_dim']}")
    model = GroupNet(groupnet_config).to(device)
    
    if not distributed or (distributed and dist.is_initialized() and is_master):
        print(f"GroupNet parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Validate dataset arguments
    if args.train_dataset_path and args.val_dataset_path:
        if not distributed or (distributed and dist.is_initialized() and is_master):
            print(f"Loading separate train/val datasets:")
            print(f"  Train: {args.train_dataset_path}")
            print(f"  Val: {args.val_dataset_path}")
    elif args.dataset_path:
        if not distributed or (distributed and dist.is_initialized() and is_master):
            print(f"Loading dataset from {args.dataset_path}")
            print(f"  Will split randomly with val_split={args.val_split}")
    else:
        raise ValueError("Must provide either --dataset_path OR both --train_dataset_path and --val_dataset_path")
    
    # Create datasets and dataloaders
    train_loader, val_loader = create_trace_captioning_dataloaders(
        dataset_path=args.dataset_path,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        image_base_path=args.image_base_path,
        image_backup_path=args.image_backup_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        transform=visual_encoder.transform,
        val_split=args.val_split,
        seed=args.seed,
        max_samples=args.max_samples
    )
    
    if not distributed or (distributed and dist.is_initialized() and is_master):
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
    
    # Optimizer - use parameter groups with different learning rates
    param_groups = [
        {'params': model.parameters(), 'lr': args.lr}
    ]
    
    if clipcap_model is not None and not args.freeze_clipcap:
        param_groups.append({
            'params': clipcap_model.model.parameters(),
            'lr': args.clipcap_lr
        })
        if not distributed or (distributed and dist.is_initialized() and is_master):
            print(f"ClipCap parameters: {sum(p.numel() for p in clipcap_model.model.parameters()):,}")
            print(f"GroupNet LR: {args.lr:.2e}, ClipCap LR: {args.clipcap_lr:.2e}")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Tensorboard
    writer = None
    if args.log_dir and is_master:
        writer = SummaryWriter(args.log_dir)

    # If distributed, replace samplers and wrap model with DDP
    if distributed:
        # Recreate dataloaders with DistributedSampler to ensure each process sees unique data
        from torch.utils.data import DistributedSampler

        # train_loader.dataset may be a Subset (from random_split), which already has .dataset attribute
        def _get_dataset(dl):
            ds = dl.dataset
            # If it's a Subset, return underlying dataset
            if hasattr(ds, 'dataset'):
                return ds
            return ds

        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset

        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=trace_collate_fn,
            pin_memory=True,
            sampler=train_sampler
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=trace_collate_fn,
            pin_memory=True,
            sampler=val_sampler
        )

    # Wrap model in DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Check if we should compute centroid baseline metrics
    is_residual_centroid = groupnet_config.get('model_type', '').lower() == 'residual_centroid'
    if is_residual_centroid and is_master:
        print("\n" + "="*60)
        print("Using ResidualCentroidGroupNet:")
        print("  - Centroid baseline metrics will be tracked during validation")
        print("  - Model output = centroid + learned residual")
        print("  - Residual branch initialized to output zeros")
        print("="*60 + "\n")
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.epochs):
        
        if not distributed or (distributed and dist.is_initialized() and is_master):
            print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        if distributed:
            # Set epoch for sampler for shuffling
            train_loader.sampler.set_epoch(epoch)
        
        # Train
        train_metrics = train_one_epoch(
            model=model,
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            loss_type=args.loss_type,
            temperature=args.temperature,
            writer=writer,
            is_master=is_master,
            distributed=distributed,
            clipcap_model=clipcap_model
        )
        
        if is_master:
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
        
        # Validate
        # Compute CIDEr every N epochs or at the last epoch
        should_compute_cider = (
            args.loss_type == 'captioning' and 
            args.eval_cider_every > 0 and 
            ((epoch + 1) % args.eval_cider_every == 0 or epoch == args.epochs - 1)
        )
        
        val_metrics = validate(
            model=model,
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            val_loader=val_loader,
            device=device,
            loss_type=args.loss_type,
            temperature=args.temperature,
            is_master=is_master,
            distributed=distributed,
            compute_centroid_metrics=is_residual_centroid,
            clipcap_model=clipcap_model,
            compute_cider=should_compute_cider,
            writer=writer,
            epoch=epoch,
            num_qualitative_samples=args.num_qualitative_samples
        )
        
        if is_master:
            log_msg = f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}, Top5 Acc: {val_metrics['top5_acc']:.4f}"
            if is_residual_centroid:
                log_msg += f" | Centroid Baseline - Acc: {val_metrics.get('centroid_acc', 0):.4f}, Top5: {val_metrics.get('centroid_top5_acc', 0):.4f}"
            if 'cider' in val_metrics:
                log_msg += f" | CIDEr: {val_metrics['cider']:.4f}"
            print(log_msg)
        
        # Log to tensorboard
        if writer is not None:
            if is_master:
                writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                writer.add_scalar('val/acc', val_metrics['acc'], epoch)
                writer.add_scalar('val/top5_acc', val_metrics['top5_acc'], epoch)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], epoch)
                
                # Log CIDEr if available
                if 'cider' in val_metrics:
                    writer.add_scalar('val/cider', val_metrics['cider'], epoch)
                
                # Log centroid baseline if available
                if is_residual_centroid:
                    writer.add_scalar('val/centroid_acc', val_metrics.get('centroid_acc', 0), epoch)
                    writer.add_scalar('val/centroid_top5_acc', val_metrics.get('centroid_top5_acc', 0), epoch)
                    # Log improvement over baseline
                    improvement = val_metrics['acc'] - val_metrics.get('centroid_acc', 0)
                    writer.add_scalar('val/improvement_over_centroid', improvement, epoch)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            if args.save_dir and is_master:
                os.makedirs(args.save_dir, exist_ok=True)
                checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
                # If model is DDP, unwrap to get underlying module
                state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
                
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': state_dict,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_metrics['acc'],
                    'val_loss': val_metrics['loss'],
                    'config': groupnet_config if isinstance(groupnet_config, dict) else groupnet_config.__dict__ if hasattr(groupnet_config, '__dict__') else dict(groupnet_config)
                }
                
                # Save ClipCap model if it was trained
                if clipcap_model is not None and not args.freeze_clipcap:
                    checkpoint_dict['clipcap_state_dict'] = clipcap_model.model.state_dict()
                    print(f"Saved best model (including ClipCap) to {checkpoint_path}")
                else:
                    print(f"Saved best model to {checkpoint_path}")
                
                torch.save(checkpoint_dict, checkpoint_path)
        
        # Save periodic checkpoint
        if args.save_dir and (epoch + 1) % args.save_every == 0 and is_master:
            os.makedirs(args.save_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            checkpoint_dict = {
                'epoch': epoch,
                'model_state_dict': state_dict,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['acc'],
                'val_loss': val_metrics['loss'],
                'config': groupnet_config if isinstance(groupnet_config, dict) else groupnet_config.__dict__ if hasattr(groupnet_config, '__dict__') else dict(groupnet_config)
            }
            
            # Save ClipCap model if it was trained
            if clipcap_model is not None and not args.freeze_clipcap:
                checkpoint_dict['clipcap_state_dict'] = clipcap_model.model.state_dict()
            
            torch.save(checkpoint_dict, checkpoint_path)
    
    print(f"\nTraining complete! Best val accuracy: {best_val_acc:.4f}")
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GroupNet for trace captioning')
    
    # Dataset args
    parser.add_argument('--dataset_path', type=str, default=None,
                        help='Path to single trace captioning dataset JSON (will be split randomly)')
    parser.add_argument('--train_dataset_path', type=str, default=None,
                        help='Path to training dataset JSON (use with --val_dataset_path)')
    parser.add_argument('--val_dataset_path', type=str, default=None,
                        help='Path to validation dataset JSON (use with --train_dataset_path)')
    parser.add_argument('--image_base_path', type=str, required=True,
                        help='Base path to images')
    parser.add_argument('--image_backup_path', type=str, default=None,
                        help='Backup path for images')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Validation split fraction (only used with --dataset_path)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max samples for debugging')
    
    # Model args
    parser.add_argument('--groupnet_config', type=str, required=True,
                        help='Path to GroupNet config YAML')
    parser.add_argument('--visual_encoder_type', type=str, default='dino',
                        choices=['dino', 'clip', 'open_clip'],
                        help='Type of visual encoder')
    parser.add_argument('--visual_encoder_name', type=str, default='dinov2_vitb14_reg',
                        help='Name/version of visual encoder (e.g., dinov2_vitb14_reg, ViT-B/16)')
    parser.add_argument('--text_encoder_type', type=str, default='talk2dino',
                        choices=['clip', 'open_clip', 'talk2dino'],
                        help='Type of text encoder')
    parser.add_argument('--text_encoder_name', type=str, default=None,
                        help='Name/version of text encoder (for CLIP models: ViT-B/16, for Talk2DINO: path to config YAML)')
    parser.add_argument('--talk2dino_config', type=str, default=None,
                        help='Path to Talk2DINO config YAML (alternative to text_encoder_name for talk2dino)')
    parser.add_argument('--resize_dim', type=int, default=518,
                        help='Resize dimension for images')
    parser.add_argument('--crop_dim', type=int, default=518,
                        help='Crop dimension for images')
    
    # Training args
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for GroupNet')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--loss_type', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'infonce', 'captioning'],
                        help='Type of loss function')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Temperature for similarity scaling')
    
    # ClipCap args (for captioning loss)
    parser.add_argument('--clipcap_config', type=str, default=None,
                        help='Path to ClipCap config YAML (required for loss_type=captioning)')
    parser.add_argument('--clipcap_weights', type=str, default=None,
                        help='Path to ClipCap weights (required for loss_type=captioning)')
    parser.add_argument('--freeze_clipcap', action='store_true',
                        help='Freeze ClipCap weights during training')
    parser.add_argument('--clipcap_lr', type=float, default=2e-5,
                        help='Learning rate for ClipCap (used only when not frozen, default matches ClipCap training)')
    
    # Evaluation args
    parser.add_argument('--eval_cider_every', type=int, default=0,
                        help='Compute CIDEr score every N epochs (0=disabled, -1=only at end). Requires caption generation.')
    parser.add_argument('--num_qualitative_samples', type=int, default=10,
                        help='Number of qualitative samples to log to TensorBoard during validation')
    
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of dataloader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    # Distributed
    parser.add_argument('--local_rank', type=int, default=None,
                        help='Local rank for distributed training (provided by torchrun)')
    
    # Negative sampling args
    parser.add_argument('--negative_sampling', type=str, default='in_batch',
                        choices=['in_batch', 'hard'],
                        help='Negative sampling strategy')
    
    # Saving args
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory for tensorboard logs')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    main(args)
