import timm
import torch
import torch.nn as nn
import math
import yaml
import os
import pickle
import random
import torchvision.transforms as T


from .decap.decap import decoding_batched, DeCap, MLP
from .decap.decap import get_decap_model
from .dino_extraction import get_self_attention, process_self_attention, transform_to_standard_dino_out, get_layer_n_output, feats
from .decap.im2txtprojection.im2txtprojection import Im2TxtProjector, ProjectionType
from .bbox_utils import extract_bboxes_feats, extract_bboxes_feats_double_dino, process_bboxes, map_traces_to_grid
from .talk2dino.talk2dino import ProjectionLayer
from .proxyclip.proxyclip import ProxyCLIP
from transformers import GPT2LMHeadModel
from .embedding_utils import get_pseudo_inverse, revert_transformation


import math
from tqdm import tqdm

import torch.nn.functional as F


# Container to store outputs
patch_embeddings = {}

# Hook function
def save_patch_embeddings(module, input, output):
    """
    module: the module being hooked (the transformer)
    input: input to the module
    output: output from the module
    """
    # output shape: (batch_size, 1 + num_patches, embedding_dim)
    patch_tokens = output[:, 1:, :]  # remove the CLS token
    patch_embeddings['tokens'] = patch_tokens
    patch_embeddings['cls'] = output[:, 0, :]
    patch_embeddings['full'] = output

def compute_region_means(patch_embeddings, variance):
    """
    Compute weighted region means for a batch of patch embeddings.

    Args:
        patch_embeddings (torch.Tensor): Tensor of shape (N, H, W, embed_dim).
        variance (float): Variance for the Gaussian weighting. If 0, select the center patch.
                         If variance > 100, use uniform weights.

    Returns:
        region_means (torch.Tensor): Weighted means for each region, shape (N, embed_dim).
        patch_weights (torch.Tensor): The weights applied, shape (N, H, W).
    """
    N = patch_embeddings.shape[0]
    grid_size = int(patch_embeddings.shape[1]**0.5)

    W = H = grid_size

    patch_embeddings = patch_embeddings.view(N, grid_size, grid_size, -1)  # Shape (N, grid_size, grid_size, embed_dim)
    device = patch_embeddings.device

    # Create coordinate grid once
    y = torch.linspace(-1, 1, grid_size, device=device)
    x = torch.linspace(-1, 1, grid_size, device=device)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    if variance == 0:
        # One-hot weight at the center
        patch_weights = torch.zeros(N, H, W, device=device)
        center_y_options = [grid_size // 2] if grid_size % 2 == 1 else [grid_size // 2 - 1, grid_size // 2]
        center_x_options = [grid_size // 2] if grid_size % 2 == 1 else [grid_size // 2 - 1, grid_size // 2]
        for i in range(N):
            cy = random.choice(center_y_options)
            cx = random.choice(center_x_options)
            patch_weights[i, cy, cx] = 1.0
    elif variance >= 100:
        # Uniform weights
        patch_weights = torch.full((N, H, W), 1 / (H * W), device=device)
    else:
        # Gaussian weights
        distances = xx**2 + yy**2
        weights = torch.exp(-distances / variance)
        weights = weights / weights.sum()  # Normalize
        patch_weights = weights.unsqueeze(0).expand(N, -1, -1)

    # Compute the weighted sum (i.e., the weighted mean)
    weighted_patches = patch_embeddings * patch_weights.unsqueeze(-1)
    region_means = weighted_patches.sum(dim=(1, 2))

    return region_means

class Patchioner(nn.Module):

    def __init__(self, decoder_weights, device, prefix_size, linear_talk2dino, support_memory_size, projection_type = None, 
                 dino_model=None, proxyclip_clipmodel=None, proxyclip_vfm=None, use_talk2dino_project=True, normalize=True, attention_type='qkv', talk2dino_config=None, 
                 talk2dino_weights=None, resize_dim=518, crop_dim=518, talk2dino_attn_type='qkv', calculate_argmax_text=False,
                 online_texts=None, clip_model_name=None, use_open_clip=False, viecap_config=None, regionclip_config=None, invite_config=None, denseclip_config=None, alphaclip_config=None, clipcap_config=None, hf_repo_id=None,
                 **kwargs):
        super().__init__(**kwargs)

        self.decoding_method = None

        if viecap_config is not None:
            if viecap_config.get('meacap', False):
                from .meacap.entrypoint import MeaCap
                self.viecap = MeaCap(viecap_config, device, clip_model_name)
            else:
                from .viecap.entrypoint import VieCap
                self.viecap = VieCap(viecap_config, device, clip_model_name)
        else:
            self.viecap = None

        if clipcap_config is not None:
            # Determine DINO feature dimension based on model type
            dino_feature_dim = prefix_size  # Use prefix_size as DINO feature dimension
            if dino_model is not None:
                if 'dinov2_vits14' in dino_model:
                    dino_feature_dim = 384
                elif 'dinov2_vitb14' in dino_model:
                    dino_feature_dim = 768
                elif 'dinov2_vitl14' in dino_model:
                    dino_feature_dim = 1024
                elif 'dinov2_vitg14' in dino_model:
                    dino_feature_dim = 1536
                
            
            from src.clipcap.entrypoint import ClipCapModel
            self.clipcap = ClipCapModel(clipcap_config, device, dino_feature_dim)
        else:
            self.clipcap = None

        if dino_model is not None and 'dinotxt' in dino_model:
            clip_model_name = 'DINO.txt'

        if alphaclip_config is not None:
            print(f"Using AlphaCLIP model {alphaclip_config}")
            # AlphaClip will be loaded later after determining patch sizes
            
        # decoder initialization
        if online_texts is not None:
            projection_type_enum = ProjectionType.ONLINE_TEXTS
        elif projection_type == 'coco':
            projection_type_enum = ProjectionType.COCO_CAPTIONS
        elif projection_type == 'msmarco':
            projection_type_enum = ProjectionType.MS_MARCO_QUERIES_A
        elif projection_type == 'blip':
            projection_type_enum = ProjectionType.CC3M_BLIP
        elif projection_type == 'vg':
            projection_type_enum = ProjectionType.VISUAL_GENOME
        elif projection_type == 'vg-test':
            projection_type_enum = ProjectionType.VISUAL_GENOME_TEST
        elif support_memory_size == 0:
            projection_type_enum = None
        elif os.path.exists(projection_type):
            print(f"Loading memory bank from {projection_type}")
            projection_type_enum = projection_type
        else:
            raise Exception("The projection_type field must be 'coco', 'msmarco', 'blip' or 'vg'")

        self.calculate_argmax_text = calculate_argmax_text
        if not self.calculate_argmax_text and decoder_weights is not None:
            self.decoder = get_decap_model(device, decoder_weights, prefix_size, hf_repo_id)
        if support_memory_size > 0:
            self.im_proj = Im2TxtProjector(
                type=projection_type_enum,
                use_talk2dino=use_talk2dino_project,
                linear_talk2dino=linear_talk2dino,
                support_memory_size=support_memory_size,
                device_str=device,
                normalize_memory_embs=(dino_model is not None) and ('dinov2' not in dino_model),
                talk2dino_attn_type=talk2dino_attn_type,
                online_texts=online_texts,
                clip_modelname=clip_model_name,
                use_open_clip=use_open_clip,
                regionclip_config=regionclip_config,
                invite_config=invite_config,
                denseclip_config=denseclip_config,
                hf_repo_id=hf_repo_id,  # Pass HF repo ID for memory bank downloading

                )
        else:
            self.im_proj = None
            
        self.normalize = normalize
        # ProxyCLIP initialization
        if proxyclip_clipmodel:
            self.proxyclip = ProxyCLIP(clip_type='openai', model_type=proxyclip_clipmodel, vfm_model=proxyclip_vfm, device=device)
            self.patch_size = self.proxyclip.vfm.patch_embed.patch_size
            if isinstance(self.patch_size, tuple):
                self.patch_size = self.patch_size[0]
        # DINOv2 initialization
        self.resize_dim=resize_dim
        self.crop_dim=crop_dim
        self.num_global_tokens = 1 if dino_model is None or "reg" not in dino_model else 5  

        if dino_model is not None:
            if 'dinov2' in dino_model:
                patch_size = 14
            elif 'patch16' in dino_model:
                patch_size = 16
            elif 'patch14' in dino_model:
                patch_size = 14
            elif 'patch32' in dino_model:
                patch_size = 32
            elif dino_model is None:
                pass 
            elif use_open_clip:
                patch_size = int(dino_model.split('/')[-1])
                assert patch_size > 0, "Patch size must be a positive integer, got {}".format(patch_size)
            elif regionclip_config is not None:
                # For RegionCLIP ResNet, use effective patch size of 32 (spatial downsampling factor)
                patch_size = regionclip_config.get('patch_size', 32)
            elif invite_config is not None:
                # For INViTE CLIP ViT, extract patch size from model name
                model_name = invite_config.get('name', 'ViT-B/32')
                if 'ViT-B/32' in model_name:
                    patch_size = 32
                elif 'ViT-B/16' in model_name:
                    patch_size = 16
                elif 'ViT-L/14' in model_name:
                    patch_size = 14
                else:
                    # Default patch size for ViT models
                    print(f"Unknown INViTE model {model_name}, using default patch size 32")
                    patch_size = 32
            elif denseclip_config is not None:
                # For DenseClip ViT, extract patch size from config
                from src.denseclip.loader import load_denseclip_config
                denseclip_config_dict = load_denseclip_config(denseclip_config)
                patch_size = denseclip_config_dict.get('model', {}).get('vision', {}).get('vision_patch_size', 16)
            elif alphaclip_config is not None:
                # For AlphaClip, extract patch size from model name
                model_name = alphaclip_config.get('name', 'ViT-B/16')
                patch_size = alphaclip_config.get('patch_size', None)
                if patch_size is None:
                    if 'ViT-B/32' in model_name:
                        patch_size = 32
                    elif 'ViT-B/16' in model_name:
                        patch_size = 16
                    elif 'ViT-L/14' in model_name:
                        patch_size = 14
                    else:
                        print(f"Unknown AlphaClip model {model_name}, using default patch size 16")
                        patch_size = 16
            elif clip_model_name == 'ResNet50x4' and dino_model == 'RN50x4':
                patch_size = 32  # Effective patch size for ResNet50x4
            else:
                raise Exception("Unknown patch size")

            if regionclip_config is not None:
                # For RegionCLIP ResNet, calculate spatial dimensions differently
                # ResNet reduces input by factor of 32, so for crop_dim=224, final spatial size is 7x7
                spatial_size = crop_dim // patch_size
                self.num_patch_tokens = spatial_size * spatial_size
                self.num_tokens = self.num_global_tokens + self.num_patch_tokens
                
                # RegionCLIP ResNet typically uses different embedding dimensions
                # This should be determined from the loaded model
                self.embed_dim = regionclip_config.get('embed_dim', 1024)  # Common for ResNet-50 CLIP models, but should be verified
            elif invite_config is not None:
                # For INViTE CLIP ViT, calculate patch dimensions like standard ViT
                self.num_patch_tokens = (crop_dim // patch_size) * (crop_dim // patch_size)
                self.num_tokens = self.num_global_tokens + self.num_patch_tokens
                
                # INViTE CLIP ViT embedding dimensions based on model architecture
                model_name = invite_config.get('name', 'ViT-B/32')
                if 'ViT-L' in model_name:
                    self.embed_dim = 768  # ViT-L/14 uses 768-dim embeddings in CLIP
                elif 'ViT-B' in model_name:
                    self.embed_dim = 512  # ViT-B uses 512-dim embeddings in CLIP  
                else:
                    self.embed_dim = 512  # Default for CLIP ViT models
            elif denseclip_config is not None:
                # For DenseClip ViT, calculate patch dimensions like standard ViT
                self.num_patch_tokens = (crop_dim // patch_size) * (crop_dim // patch_size)
                self.num_tokens = self.num_global_tokens + self.num_patch_tokens
                
                # DenseClip embedding dimensions from config
                self.embed_dim = denseclip_config_dict.get('model', {}).get('vision', {}).get('embed_dim', 512)
            else:
                self.num_patch_tokens = crop_dim // patch_size * crop_dim // patch_size
                self.num_tokens = self.num_global_tokens + self.num_patch_tokens
            
            if regionclip_config is not None:
                # RegionCLIP ResNet typically uses different embedding dimensions
                # This should be determined from the loaded model
                self.embed_dim = regionclip_config.get('embed_dim', 1024)  # Common for ResNet-50 CLIP models, but should be verified
            elif invite_config is not None:
                # INViTE CLIP ViT embedding dimensions based on model architecture
                model_name = invite_config.get('name', 'ViT-B/32')
                if 'ViT-L' in model_name:
                    self.embed_dim = 768  # ViT-L/14 uses 768-dim embeddings in CLIP
                elif 'ViT-B' in model_name:
                    self.embed_dim = 512  # ViT-B uses 512-dim embeddings in CLIP  
                else:
                    print(f"Unknown INViTE model {model_name}, using default embedding dimension 512")
                    self.embed_dim = 512  # Default for CLIP ViT models
            elif denseclip_config is not None:
                # DenseClip embedding dimensions from config
                self.embed_dim = denseclip_config_dict.get('model', {}).get('vision', {}).get('embed_dim', 512)
            elif alphaclip_config is not None:
                # AlphaClip embedding dimensions based on model architecture
                model_name = alphaclip_config.get('name', 'ViT-B/16')
                embed_dim = alphaclip_config.get('embed_dim', None)
                if embed_dim is not None:
                    self.embed_dim = embed_dim
                else:
                    if 'ViT-L' in model_name:
                        self.embed_dim = 768  # ViT-L uses 768-dim embeddings in CLIP
                    elif 'ViT-B' in model_name:
                        self.embed_dim = 512  # ViT-B uses 512-dim embeddings in CLIP
                    else:
                        print(f"Unknown AlphaClip model {model_name}, using default embedding dimension 512")
                        self.embed_dim = 512
                
                # For AlphaClip, calculate patch dimensions
                self.num_patch_tokens = (crop_dim // patch_size) * (crop_dim // patch_size)
                self.num_tokens = self.num_global_tokens + self.num_patch_tokens
            elif 'vitl' in dino_model or 'vit_large' in dino_model or 'ViT-L' in dino_model or 'ViT-H' in dino_model:
                self.embed_dim = 1024
            elif 'vitb' in dino_model or 'vit_base' in dino_model or 'ViT-B' in dino_model:
                self.embed_dim = 768
            elif 'vits' in dino_model or 'vit_small' in dino_model:
                self.embed_dim = 384
            elif prefix_size is not None:
                print("[FALLBACK] Using prefix_size as embed_dim:", prefix_size)
                self.embed_dim = prefix_size
            else:
                raise Exception("Unknown ViT model")

        self.model_name = dino_model if dino_model is not None else 'proxyclip'
        self.num_attn_heads = 16 if dino_model is not None and not 'vits' in dino_model else 6
        self.scale = 0.125
        if dino_model is not None:
            if 'dinov2' in dino_model:
                self.num_global_tokens = 1 if "reg" not in dino_model else 5  
                
                model_family = 'facebookresearch/dinov2'
                self.dino = torch.hub.load(model_family, dino_model)
                
                if 'dinotxt' in dino_model:
                    self.dino = self.dino.visual_model.backbone.model
                self.image_transforms = T.Compose([
                    T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(crop_dim),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
                self.image_transforms_no_crop = T.Compose([
                    T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ])
            elif 'openai' in dino_model:
                print(f"Loading OpenAI model {dino_model} using timm.create_model...")
                # we use this case to test DeCap original architecture (with CLIP instead of DINOv2)

                # timm uses GELU while original OpenAI model uses QuickGELU
                # https://github.com/huggingface/pytorch-image-models/issues/1754
                # we fix the activation function because DeCap is trained using OpenAI interface
                class QuickGELU(torch.nn.Module):
                    def forward(self, x: torch.Tensor):
                        return x * torch.sigmoid(1.702 * x)

                print(f"timm.list_models(pretrained=True) contains dino_model ({dino_model}):", dino_model in timm.list_models(pretrained=True))

                timm_model = timm.create_model(dino_model, pretrained=True, act_layer=QuickGELU, img_size=resize_dim)

                print(f"timm_model is instance of {type(timm_model)}")

                self.dino = timm_model.to(device)

                assert hasattr(self.dino, 'blocks'), f"The model does not have 'blocks' attribute. dino is instance of {type(self.dino)}"

                self.image_transforms = T.Compose([
                    T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(crop_dim),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])
                
                self.image_transforms_no_crop = T.Compose([
                    T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])
            elif use_open_clip:

                print(f"""
                -------------------------------------------
                Using OpenCLIP model {dino_model} 
                -------------------------------------------
    """)
                # load open clip weights
                from open_clip import create_model_and_transforms, get_tokenizer
                open_clip, preprocess_train, preprocess_val = create_model_and_transforms(
                    model_name=dino_model,
                    pretrained="laion2b_s32b_b79k",
                    device=device,
                    #image_size=224,
                    #context_length=77,
                    #vocab_size=49408,
                )
                tokenizer = get_tokenizer(dino_model.replace("/", "-"))


                open_clip.eval()

                image_transforms_open_clip = preprocess_train

                self.dino = open_clip
                self.image_transforms = image_transforms_open_clip
                self.image_transforms_no_crop = T.Compose([
                    T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])

                self.decoding_method = tokenizer.decode
            elif regionclip_config is not None:
                # load regionclip model
                from src.regionclip.loader import load_regionclip_from_checkpoint

                regionclip_checkpoint = regionclip_config.get('checkpoint', None)
                if regionclip_checkpoint is None:
                    raise Exception("RegionCLIP checkpoint not specified in the configuration")
                regionclip_config_name = regionclip_config.get('config_name', None)

                self.dino = load_regionclip_from_checkpoint(regionclip_checkpoint, device=device, config=regionclip_config_name, override_config=regionclip_config)

                # use standard clip preprocessing transforms
                self.image_transforms = T.Compose([
                    T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(crop_dim),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])
                self.image_transforms_no_crop = T.Compose([
                    T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])
                
                # For RegionCLIP ResNet, compute effective patch size based on spatial downsampling
                # ResNet reduces spatial resolution by factor of 32 (2^5: stem avgpool + 4 layers with stride 2)
                # The final feature map from res4 (before attnpool) has resolution input_size // 32
                # So effective patch size is 32 for mapping between image coordinates and feature map coordinates
                self.patch_size = regionclip_config.get('patch_size', 32)

            elif invite_config is not None:
                # load INViTE CLIP model
                from src.INViTE.loader import load_invite_clip

                # Load INViTE CLIP model using the config
                self.dino, preprocess_transform, tokenize_method = load_invite_clip(invite_config, device=device)

                # Use the preprocess transform from INViTE CLIP
                self.image_transforms = preprocess_transform
                self.image_transforms_no_crop = T.Compose([
                    T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])
                
                # Extract patch size from model name for coordinate mapping
                model_name = invite_config.get('name', 'ViT-B/32')
                if 'ViT-B/32' in model_name:
                    self.patch_size = 32
                elif 'ViT-B/16' in model_name:
                    self.patch_size = 16
                elif 'ViT-L/14' in model_name:
                    self.patch_size = 14
                else:
                    print(f"Unknown INViTE model {model_name}, using default patch size 32")
                    self.patch_size = 32

            elif denseclip_config is not None:
                # load DenseClip model
                from src.denseclip.loader import load_denseclip

                # Load DenseClip model using the config
                checkpoint_path = denseclip_config_dict.get('checkpoint_path', None)
                config_name = denseclip_config_dict.get('config_name', 'denseclip_vitb16')
                
                self.dino = load_denseclip(config_name=denseclip_config, device=device)

                # Use standard CLIP preprocessing transforms
                self.image_transforms = T.Compose([
                    T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(crop_dim),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])
                self.image_transforms_no_crop = T.Compose([
                    T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])
                
                # Extract patch size from config for coordinate mapping
                self.patch_size = denseclip_config_dict.get('model', {}).get('vision', {}).get('vision_patch_size', 16)

            elif alphaclip_config is not None:
                # load AlphaClip model
                from src.alphaclip.alphaclip_loader import load_alphaclip

                # Load AlphaClip model using the config
                model_name = alphaclip_config.get('name', None)
                alpha_vision_checkpoint = alphaclip_config.get('alpha_vision_checkpoint', None)
                loader, self.dino, preprocess_transform = load_alphaclip(model_name=model_name, device=device, alpha_vision_ckpt_pth=alpha_vision_checkpoint)

                # Use standard CLIP preprocessing transforms
                self.image_transforms = T.Compose([
                    T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(crop_dim),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])
                self.image_transforms_no_crop = T.Compose([
                    T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                std=(0.26862954, 0.26130258, 0.27577711)),
                ])
                
                # Extract patch size from model name for coordinate mapping
                patch_size = alphaclip_config.get('patch_size', None)
                if patch_size is not None:
                    self.patch_size = patch_size
                else:
                    if 'ViT-B/32' in model_name:
                        self.patch_size = 32
                    elif 'ViT-B/16' in model_name:
                        self.patch_size = 16
                    elif 'ViT-L/14' in model_name:
                        self.patch_size = 14
                    else:
                        print(f"Unknown AlphaClip model {model_name}, using default patch size 16")
                        self.patch_size = 16

            else:
                raise Exception("Model family unsupported")
        else:
            self.image_transforms = T.Compose([
                T.Resize(resize_dim, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(crop_dim),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])
            self.image_transforms_no_crop = T.Compose([
                T.Resize((resize_dim, resize_dim), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ])

        if attention_type != 'qkv':
            # in case kkv_attention is True, we perform the attention of the last block using Keys as Queries
            original_qkv = self.dino.blocks[-1].attn.qkv
            embed_dim = original_qkv.in_features
            
            weights = {}
            biases = {}
            weights['q'], weights['k'], weights['v'] = original_qkv.weight.reshape(3, embed_dim, embed_dim)
            biases['q'], biases['k'], biases['v'] = original_qkv.bias.reshape(3, embed_dim)
            
            new_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
            new_qkv.weight.data.copy_(torch.cat([weights[x] for x in attention_type], dim=0))
            new_qkv.bias.data.copy_(torch.cat([biases[x] for x in attention_type], dim=0))
            self.dino.blocks[-1].attn.qkv = new_qkv
        
        if dino_model is not None:

            if self.dino is not None:
                self.dino.eval()

            if hasattr(self.dino, 'blocks'):
                self.dino.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
            # elif hasattr(self.dino, 'visual_model'):
            #     self.dino.visual_model.backbone.model.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
            # need patch_size
            if 'dino' in dino_model:
                self.patch_size = self.dino.patch_size
            elif 'openai' in dino_model:
                # in the case self.dino is a timm model, we need to get patch_size from 
                # the model's configuration
                # should get patch size from dino_model, which is a string with the following format:
                # 'vit_base_patch32_clip_224.openai'
                self.patch_size = int(dino_model.split('_')[2].replace('patch', ''))
            elif regionclip_config is not None:
                # For RegionCLIP ResNet, patch_size was already set above to 32
                pass  # self.patch_size = 32 was set earlier
            elif invite_config is not None:
                # For INViTE CLIP ViT, patch_size was already set above based on model name
                pass  # self.patch_size was set during model loading
            elif denseclip_config is not None:
                # For DenseClip ViT, patch_size was already set above from config
                pass  # self.patch_size was set during model loading
            elif alphaclip_config is not None:
                # AlphaClip initialization
                if self.dino is not None:
                    self.dino.eval()
                # AlphaClip patch_size was already set during model loading
                # No attention hooks needed for AlphaClip since we don't access self-attention

            if talk2dino_weights is not None:
                # Talk2DINO initialization
                talk2dino = ProjectionLayer.from_config(talk2dino_config)
                talk2dino.load_state_dict(torch.load((talk2dino_weights), device))

                self.embed_inversion = True
                self.talk2dino_A_pinv = get_pseudo_inverse(talk2dino.linear_layer.weight).to(device)
                self.talk2dino_b = talk2dino.linear_layer.bias.to(device)
            else:
                self.embed_inversion = False
        else:
            self.embed_inversion = False

        # Determine backbone type based on configuration
        if proxyclip_clipmodel is not None:
            self.backbone_type = 'CLIP'  # ProxyCLIP uses CLIP
        elif regionclip_config is not None:
            self.backbone_type = 'RegionCLIP'
            self.regionclip_config = regionclip_config.copy()  # Store config for later use
        elif invite_config is not None:
            self.backbone_type = 'INViTE'
        elif denseclip_config is not None:
            self.backbone_type = 'DenseClip'
            self.denseclip_config = denseclip_config_dict.copy()  # Store config for later use
        elif alphaclip_config is not None:
            self.backbone_type = 'AlphaClip'
            self.alphaclip_config = alphaclip_config.copy()  # Store config for later use
        elif use_open_clip and dino_model is not None:
            self.backbone_type = 'OpenCLIP'
        elif dino_model is not None:
            if 'dinotxt' in dino_model:
                self.backbone_type = 'DINO.txt'
            elif 'dinov2' in dino_model:
                self.backbone_type = 'DINO'
            elif 'openai' in dino_model:

                self.backbone_type = 'CLIP'
            else:
                self.backbone_type = 'DINO'  # Default for other DINO variants
        else:
            self.backbone_type = 'CLIP'  # Default fallback
        
        if not hasattr(self, 'dino'):
            print(f"Warning: No DINO model loaded!")
            self.dino = None



    @classmethod
    def from_config(cls, config, device='cpu', online_texts=None):
        if type(config) is str:
            # if the configuration is a string it is either a path to a yaml file or a huggingface model id
            if os.path.exists(config):
                # we treat it as a file path
                with open(config, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # let suppose it is a huggingface model id
                # we want to download config.yaml from the model repository
                from .hf_utils import get_model_path_with_hf_fallback
                hf_repo_id = str(config).split('huggingface.co/')[-1] # if the full url is given we extract the repo id
                print(f"Loading model configuration from HuggingFace repo {hf_repo_id}")
                config_file = 'config.yaml'
                config_path = get_model_path_with_hf_fallback(config, hf_repo_id=hf_repo_id, filename=config_file)
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
        model = cls(
            projection_type=config.get('projection_type', 'coco'),
            decoder_weights=config.get('decap_weights', None),
            device=device,
            prefix_size=config['prefix_size'],
            linear_talk2dino=config.get('linear_talk2dino', False),
            support_memory_size=config['support_memory_size'],
            dino_model=config.get('dino_model', None),
            proxyclip_clipmodel=config.get('proxyclip_clipmodel', None),
            proxyclip_vfm=config.get('proxyclip_vfm', None),
            use_talk2dino_project=config.get('use_talk2dino_project', True),
            normalize=config.get('normalize', True),
            attention_type=config.get('attention_type', 'qkv'),
            talk2dino_config=config.get('talk2dino_config', None),
            talk2dino_weights=config.get('talk2dino_weights', None),
            resize_dim=config.get('resize_dim', 518),
            crop_dim=config.get('crop_dim', 518),
            talk2dino_attn_type=config.get('talk2dino_attn_type', 'qkv'),
            calculate_argmax_text=config.get('calculate_argmax_text', False),
            clip_model_name=config.get('clip_model_name', None),
            online_texts=online_texts,
            use_open_clip=config.get('use_open_clip', False),
            viecap_config=config.get('viecap', None),
            regionclip_config=config.get('regionclip_config', None),
            invite_config=config.get('invite_config', None),
            denseclip_config=config.get('denseclip_config', None),
            alphaclip_config=config.get('alphaclip_config', None),
            clipcap_config=config.get('clipcap', None),
            hf_repo_id=config.get('hf_repo_id', None),
        )
        model.to(device)
        return model


    def forward(self, imgs,
                get_cls_capt=True,
                get_avg_self_attn_capt=False,
                get_attn_heads_capt=False,
                get_patch_capts=False,
                get_register_capts=False,
                bboxes=None,
                traces=None,
                get_controllable_capts=False,
                bs_factor=4,
                gaussian_avg=False,
                gaussian_bbox_variance=0.5,
                get_avg_patch_capt=False,
                gaussian_img_variance=1,
                use_attn_map_for_bboxes=False,
                use_attention_tracing=False,
                double_DINO_for_bboxes=False,
                double_DINO_for_bboxes_return_type="avg",
                double_DINO_use_cls=False,
                cleaning_type=None,
                clean_after_projection=True,
                alpha=1.0,
                clean_from="cls",
                caption_bboxes_type : str = None,
                return_n_best_sims=None,
                compute_scores : bool = False
                ):
        """
        bboxes: [BS x N_BOX_MAX x 4]
        - double_DINO_for_bboxes_return_type : "cls" | "avg" | "gaussian_avg"
        - caption_bboxes_type = None | capt_type : str either 'avg_self_attn_capt' or 'cls_capt' if we want to compute the image caption of each bounding box as the caption of the cropped image
        - cleaning_type : None | "orthogonal_projection" | "contrastive_mask"
        - clean_after_projection : bool - if True, it first projects the patch embeddings and general token in textual space and then apply cleaning
        - alpha : between 0.0 and 1.0, used for "orthogonal_projection", weights the projection to subtract
        - clean_from : "cls" | "avg_self_attn"
        """
        assert clean_from in ["cls", "avg_self_attn"]
        assert cleaning_type in [None, "orthogonal_projection", "contrastive_mask"]

        outs = {}
        bs = imgs.shape[0]

        if self.dino is not None and bboxes is not None and double_DINO_for_bboxes:
            if self.backbone_type == 'AlphaClip':
                raise ValueError("double_DINO_for_bboxes is not supported with AlphaClip. AlphaClip processes regions differently.")
            self.dino.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
            self.dino.blocks[-1].register_forward_hook(get_layer_n_output)

        if self.dino is not None and hasattr(self.dino, 'visual') and hasattr(self.dino.visual, 'transformer'):
            # Attach hook to the visual transformer
            hook_handle = self.dino.visual.transformer.register_forward_hook(save_patch_embeddings)

        if caption_bboxes_type is not None:
            return self.caption_bboxes(imgs, bboxes, caption_bboxes_type, compute_scores=compute_scores)

        # Special handling for AlphaClip: process each bbox/trace separately
        if self.backbone_type == 'AlphaClip' and (bboxes is not None or traces is not None):
            return self.forward_alphaclip_with_regions(imgs, bboxes, traces, get_cls_capt, get_avg_self_attn_capt, 
                                                      get_attn_heads_capt, get_patch_capts, get_register_capts,
                                                      get_controllable_capts, get_avg_patch_capt, 
                                                      gaussian_avg, gaussian_bbox_variance, gaussian_img_variance,
                                                      compute_scores, return_n_best_sims)

        # Forward pass based on backbone type
        if 'DINO' in self.backbone_type:
            dino_outs = self.dino(imgs, is_training=True)
        elif self.backbone_type == 'CLIP' and self.model_name == 'proxyclip':
            dino_outs = self.proxyclip(imgs)
        elif self.backbone_type == 'CLIP' and 'openai' in self.model_name:
            # Using timm interface for OpenAI CLIP models
            output = self.dino.forward_features(imgs)
            # Projecting 768 -> 512
            output = self.dino.head(output)

            # Reporting output in DINOv2 format
            dino_outs = {
                'x_norm_clstoken': output[:, 0, :],
                'x_norm_patchtokens': output[:, 1:, :],
            }
        elif self.backbone_type == 'AlphaClip':
            # AlphaClip ViT case - standard forward for whole images
            # alphaclip always needs the alpha mask, so we pass it dummy masks made of ones, one per image
            # the shape is [BS, 1, H, W] where H and W are grid dimensions
            grid_size = self.crop_dim // self.patch_size
            alpha_mask = torch.ones((imgs.shape[0], 1, grid_size, grid_size), device=imgs.device)
            # upscale alpha_mask to match the input image size
            alpha_mask = F.interpolate(alpha_mask, size=(self.crop_dim, self.crop_dim),
                                       mode='nearest') # using nearest, so that the mask is made only of ones
            output = self.dino.visual(imgs, alpha=alpha_mask, return_patches=True)

            # Reporting output in DINOv2 format
            dino_outs = {
                'x_norm_clstoken': output[:, 0, :],      # CLS token
                'x_norm_patchtokens': output[:, 1:, :],  # Patch tokens
                'x_norm_regtokens': None,                # AlphaClip doesn't have register tokens
            }
        elif self.backbone_type == 'RegionCLIP':
            # RegionCLIP ResNet case
            use_attnpool_for_spatial_feats = self.regionclip_config.get('use_attnpool_for_spatial_feats', True)
            dino_outs = self.dino.visual.forward_return_spatial_feats(imgs, use_attnpool_for_spatial_feats=use_attnpool_for_spatial_feats)
        elif self.backbone_type == 'INViTE':
            # INViTE CLIP ViT case
            get_all_last = True
            output = self.dino.visual(imgs, get_all_last=get_all_last)

            # Reporting output in DINOv2 format
            dino_outs = {
                'x_norm_clstoken': output[:, 0, :],      # CLS token
                'x_norm_patchtokens': output[:, 1:, :],  # Patch tokens
            }
            
        elif self.backbone_type == 'DenseClip':
            # DenseClip ViT case
            # DenseClip model has encode_image method that returns features
            output = self.dino.visual.forward(imgs, get_patches=True)
            
            # DenseClip returns features in format compatible with CLIP
            # We need to extract the visual features and structure them properly
            if hasattr(output, 'shape') and len(output.shape) == 3:
                # Output is [batch_size, num_tokens, embed_dim]
                # First token is CLS, rest are patch tokens
                dino_outs = {
                    'x_norm_clstoken': output[:, 0, :],      # CLS token
                    'x_norm_patchtokens': output[:, 1:, :],  # Patch tokens
                }
            else:
                # If output format is different, handle accordingly
                # This might need adjustment based on actual DenseClip output format
                raise ValueError(f"Unexpected DenseClip output format: {output.shape if hasattr(output, 'shape') else type(output)}")
            
        elif self.backbone_type == 'OpenCLIP':
            # Using open_clip interface
            output = self.dino.visual(imgs)
            output = patch_embeddings['full']

            output = output @ self.dino.visual.proj  # shape (B, N_patches, output_dim)

            # Reporting output in DINOv2 format
            dino_outs = {
                'x_norm_clstoken': output[:, 0, :],
                'x_norm_patchtokens': output[:, 1:, :],
            }
        else:
            raise ValueError(f"Unsupported backbone type: {self.backbone_type}")
        
        # Handle self-attention processing (only for models that have attention mechanisms)
        has_attention = (('DINO' in self.backbone_type or self.backbone_type == 'DenseClip') and 
                        'self_attn' in feats)
        
        if has_attention:
            self_attn, self_attn_maps = process_self_attention(feats['self_attn'], imgs.shape[0], self.num_tokens, self.num_attn_heads, self.embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
            avg_self_attn_token = (self_attn.unsqueeze(-1) * dino_outs['x_norm_patchtokens']).mean(dim=1)

            self_attn_maps = self_attn_maps.softmax(dim=-1)
            disentangled_self_attn = (dino_outs['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)
        #else:
        #    # For models without accessible self-attention (RegionCLIP ResNet, INViTE CLIP), create fallback
        #    avg_self_attn_token = dino_outs['x_norm_patchtokens'].mean(dim=1)
        #    # Create dummy attention heads (just repeat the average)
        #    disentangled_self_attn = avg_self_attn_token.unsqueeze(1).repeat(1, self.num_attn_heads, 1)

        if cleaning_type is not None:
            batch_patchtokens = dino_outs['x_norm_patchtokens']
            if clean_from == "cls":
                batch_clean_from_token = dino_outs['x_norm_clstoken']
            else:  # clean_from == "avg_self_attn"
                if has_attention:
                    batch_clean_from_token = avg_self_attn_token
                else:
                    # Fallback to cls token if self-attention not available
                    batch_clean_from_token = dino_outs['x_norm_clstoken']
            
            dino_outs['x_norm_patchtokens'] = None

            # Loop over the batch size and apply ctx_cleaner per element
            for i in range(bs):
                # Extract the patch tokens and class token for the current batch element
                patchtokens_i = batch_patchtokens[i:i+1]  # Shape: [1, seq_len, embed_dim]
                clean_from_token_i = batch_clean_from_token[i:i+1]  # Shape: [1, embed_dim]

                # Apply ctx_cleaner to each batch element
                if clean_after_projection:
                    cleaned_patchtokens = self.ctx_cleaner(
                        self.im_proj.project(patchtokens_i, normalize=True),
                        self.im_proj.project(clean_from_token_i, normalize=True),
                        cleaning_type=cleaning_type,
                        alpha=alpha
                    )
                else:
                    cleaned_patchtokens = self.im_proj.project( \
                                            self.ctx_cleaner(
                                            patchtokens_i / patchtokens_i.norm(dim=-1,keepdim=True),
                                            clean_from_token_i / clean_from_token_i.norm(dim=-1,keepdim=True),
                                            cleaning_type=cleaning_type,
                                            alpha=alpha
                                        ), normalize=True
                                        )

                # Store the cleaned patch tokens in the output dictionary
                if 'x_norm_patchtokens' not in dino_outs or dino_outs['x_norm_patchtokens'] is None:
                    dino_outs['x_norm_patchtokens'] = cleaned_patchtokens
                else:
                    dino_outs['x_norm_patchtokens'] = torch.cat(
                        (dino_outs['x_norm_patchtokens'], cleaned_patchtokens), dim=0
                    )


        embed_dim = dino_outs['x_norm_patchtokens'].shape[-1]
        if get_cls_capt:
            ret = self.caption_tokens(dino_outs['x_norm_clstoken'], compute_scores=compute_scores)
            if compute_scores is True:
                outs['cls_capt'], outs['cls_capt_scores'] = ret
            else:
                outs['cls_capt'] = ret
        if get_avg_self_attn_capt:
            ret = self.caption_tokens(avg_self_attn_token, compute_scores=compute_scores)
            if compute_scores is True:
                outs['avg_self_attn_capt'], outs['avg_self_attn_capt_scores'] = ret
            else:
                outs['avg_self_attn_capt'] = ret
        if get_avg_patch_capt:
            ret = self.caption_tokens(compute_region_means(dino_outs['x_norm_patchtokens'], gaussian_img_variance), compute_scores=compute_scores)
            if compute_scores is True:
                outs['avg_patch_capt'], outs['avg_patch_capt_scores'] = ret
            else:
                outs['avg_patch_capt'] = ret
            
        
        if get_attn_heads_capt:
            
            ret = self.caption_tokens(disentangled_self_attn.view(-1, embed_dim), compute_scores=compute_scores)
            
            if compute_scores is True:
                attn_heads_capt_unrolled, attn_heads_scores_unrolled = ret
                outs['attn_heads_capts'] = [attn_heads_capt_unrolled[i * self.num_attn_heads:(i + 1) * self.num_attn_heads] for i in range(bs)]
                outs['attn_heads_scores'] = [attn_heads_scores_unrolled[i * self.num_attn_heads:(i + 1) * self.num_attn_heads] for i in range(bs)]
            else:
                attn_heads_capt_unrolled = ret
                outs['attn_heads_capts'] = [attn_heads_capt_unrolled[i * self.num_attn_heads:(i + 1) * self.num_attn_heads] for i in range(bs)]
        if get_patch_capts:
            n_patches = dino_outs['x_norm_patchtokens'].shape[1]
            
            ret = self.caption_tokens(dino_outs['x_norm_patchtokens'].reshape(-1, embed_dim), project=cleaning_type is None, compute_scores=compute_scores)
            
            if compute_scores is True:
                patch_tokens_capts_unrolled, patch_tokens_scores_unrolled = ret
                outs['patch_tokens_capts'] = [patch_tokens_capts_unrolled[i * n_patches:(i + 1) * n_patches] for i in range(bs)]
                outs['patch_tokens_scores'] = [patch_tokens_scores_unrolled[i * n_patches:(i + 1) * n_patches] for i in range(bs)]
            else:
                patch_tokens_capts_unrolled = ret
                outs['patch_tokens_capts'] = [patch_tokens_capts_unrolled[i * n_patches:(i + 1) * n_patches] for i in range(bs)]
        if get_register_capts:
            
            ret = self.caption_tokens(dino_outs['x_norm_regtokens'].view(-1, embed_dim), compute_scores=compute_scores)
            
            if compute_scores is True:
                register_capt_unrolled, register_scores_unrolled = ret
                outs['register_capts'] = [register_capt_unrolled[i * 4:(i + 1) * 4] for i in range(bs)]
                outs['register_scores'] = [register_scores_unrolled[i * 4:(i + 1) * 4] for i in range(bs)]
            else:
                register_capt_unrolled = ret
                outs['register_capts'] = [register_capt_unrolled[i * 4:(i + 1) * 4] for i in range(bs)]
        if bboxes is not None and not get_controllable_capts:
            bbox_bs = bs * bs_factor
            n_boxes = bboxes.shape[1]
            if double_DINO_for_bboxes:
                outs_layer_n = transform_to_standard_dino_out(feats['intermediate_output'], self.dino)
                if double_DINO_use_cls:
                    cls_layer_n = outs_layer_n['x_norm_clstoken']
                    registers_layer_n = outs_layer_n['x_norm_regtokens']
                else:
                    cls_layer_n = None
                    registers_layer_n = None
                patches_layer_n = outs_layer_n['x_norm_patchtokens']
                bbox_feats = extract_bboxes_feats_double_dino(self.dino, patches_layer_n, bboxes, cls_layer_n, registers_layer_n, self.patch_size, return_type=double_DINO_for_bboxes_return_type, gaussian_bbox_variance=gaussian_bbox_variance)#.view(-1, self.embed_dim)
            else:
                bbox_attn_maps = self_attn.cpu() if (use_attn_map_for_bboxes and has_attention) else None
                bbox_feats = extract_bboxes_feats(dino_outs['x_norm_patchtokens'], bboxes, gaussian_avg=gaussian_avg, 
                                                  gaussian_bbox_variance=gaussian_bbox_variance,
                                                  patch_size=self.patch_size, attention_map=bbox_attn_maps)#.view(-1, self.embed_dim)


            bbox_feats = bbox_feats.view(-1, embed_dim)
            n_batch = math.ceil(bbox_feats.shape[0] / bbox_bs)
            outs['bbox_capts'] = []
            if compute_scores is True:
                outs['bbox_scores'] = []
            if return_n_best_sims is not None:
                outs['bbox_sims'] = []
            #print(f"{n_batch = }, {bs = }, {bbox_bs = }")
            for i in range(n_batch):
                start = i * bbox_bs
                end = start + bbox_bs if i < n_batch - 1 else bbox_feats.shape[0]
                #cur_bbox_feats = bbox_feats[start:end]
                if return_n_best_sims is None:
                    
                    ret = self.caption_tokens(bbox_feats[start:end], project=(cleaning_type is None), compute_scores=compute_scores)
                    
                    if compute_scores is True:
                        bbox_capts, bbox_scores = ret
                        outs['bbox_capts'].extend(bbox_capts)
                        outs['bbox_scores'].extend(bbox_scores)
                    else:
                        bbox_capts = ret
                        outs['bbox_capts'].extend(bbox_capts)
                else:
                    
                    ret = self.caption_tokens(bbox_feats[start:end], project=(cleaning_type is None), return_n_best_sims=return_n_best_sims, compute_scores=compute_scores)
                    
                    if compute_scores is True:
                        (bbox_capts, bbox_sims), bbox_scores = ret
                        outs['bbox_capts'].extend(bbox_capts)
                        outs['bbox_sims'].extend(bbox_sims)
                        outs['bbox_scores'].extend(bbox_scores)
                    else:
                        bbox_capts, bbox_sims = ret
                        outs['bbox_capts'].extend(bbox_capts)
                        outs['bbox_sims'].extend(bbox_sims)
                    
            outs['bbox_capts'] = [outs['bbox_capts'][i * n_boxes:(i + 1) * n_boxes] for i in range(bs)]
            if compute_scores is True:
                outs['bbox_scores'] = [outs['bbox_scores'][i * n_boxes:(i + 1) * n_boxes] for i in range(bs)]
            if return_n_best_sims is not None:
                outs['bbox_sims'] = [outs['bbox_sims'][i * n_boxes:(i + 1) * n_boxes] for i in range(bs)]
        elif bboxes is not None and get_controllable_capts and self.backbone_type != 'AlphaClip':
            bbox_attn_maps = self_attn.cpu() if (use_attn_map_for_bboxes and has_attention) else None
            n_boxes = bboxes.shape[1]
            bbox_feats = extract_bboxes_feats(dino_outs['x_norm_patchtokens'], bboxes, gaussian_avg=gaussian_avg, gaussian_bbox_variance=gaussian_bbox_variance, get_single_embedding_per_image=True, patch_size=self.patch_size, attention_map=bbox_attn_maps)
            
            outs['set_controllable_capts'] = self.caption_tokens(bbox_feats)
        
        if traces is not None and self.backbone_type != 'AlphaClip':
            n_patches = int(dino_outs['x_norm_patchtokens'].shape[1] ** 0.5)
            relevant_patches = torch.stack([map_traces_to_grid(trace, n_patches) for trace in traces], dim=0).to(next(self.parameters()).device)
            if use_attention_tracing and has_attention:
                relevant_patches = (self_attn.view(relevant_patches.shape) * relevant_patches)
            trace_embeds = (relevant_patches.unsqueeze(-1) * dino_outs['x_norm_patchtokens'].view(bs, n_patches, n_patches, embed_dim)).mean(dim=(1,2))
            
            outs['trace_capts'] = self.caption_tokens(trace_embeds)

        return outs

    def forward_alphaclip_with_regions(self, imgs, bboxes=None, traces=None, get_cls_capt=True, 
                                     get_avg_self_attn_capt=False, get_attn_heads_capt=False, 
                                     get_patch_capts=False, get_register_capts=False,
                                     get_controllable_capts=False, get_avg_patch_capt=False,
                                     gaussian_avg=False, gaussian_bbox_variance=0.5, gaussian_img_variance=1,
                                     compute_scores=False, return_n_best_sims=None):
        """
        Special forward method for AlphaClip that processes each bbox/trace separately.
        This is required because AlphaClip processes regions differently than other backbones.
        
        AlphaClip's visual forward accepts an alpha parameter which is a binary mask
        indicating which patches should be attended to.
        """
        from src.alphaclip.alpha_mask_utils import (
            bbox_to_alpha_mask, bboxes_to_alpha_mask, 
            trace_to_alpha_mask, traces_to_alpha_mask
        )
        
        outs = {}
        bs = imgs.shape[0]
        device = next(self.parameters()).device
        
        # Calculate grid size for alpha masks
        crop_dim = self.crop_dim
        patch_size = 1
        grid_size = crop_dim // patch_size

        effective_grid_size = self.crop_dim // self.patch_size
        
        # Check if we should use CLS token or aggregate patches
        use_cls_for_localized = self.alphaclip_config.get('use_cls_for_localized_captions', True)
        
        def extract_alphaclip_features(output, alpha_mask=None):
            """
            Extract features from AlphaClip output based on configuration.
            
            Args:
                output: AlphaClip visual output [batch, num_tokens, embed_dim]
                alpha_mask: Optional alpha mask used for processing [batch, grid_size, grid_size]
            
            Returns:
                features: Extracted features [batch, embed_dim]
            """
            if use_cls_for_localized:
                # Use CLS token (original implementation)
                return output[:, 0, :]  # [batch, embed_dim]
            else:
                # Aggregate patches like standard forward method
                patch_tokens = output[:, 1:, :]  # [batch, num_patches, embed_dim]
                
                if alpha_mask is not None:
                    # Weight patches by alpha mask
                    # Flatten alpha mask to match patch dimensions
                    alpha_flat = alpha_mask.view(alpha_mask.shape[0], -1)  # [batch, num_patches]
                    
                    # Apply mask weights to patches
                    weighted_patches = patch_tokens * alpha_flat.unsqueeze(-1)  # [batch, num_patches, embed_dim]
                    
                    # Normalize by sum of weights to get average
                    mask_sum = alpha_flat.sum(dim=1, keepdim=True) + 1e-8  # Avoid division by zero
                    aggregated_features = weighted_patches.sum(dim=1) / mask_sum.unsqueeze(-1)  # [batch, embed_dim]
                else:
                    # Simple average of all patches
                    aggregated_features = patch_tokens.mean(dim=1)  # [batch, embed_dim]
                
                return aggregated_features
        
        # Handle controllable captions case (OR all regions into single mask per image)
        if get_controllable_capts and (bboxes is not None or traces is not None):
            controllable_capts = []
            
            for img_idx in range(bs):
                img = imgs[img_idx:img_idx+1]  # [1, C, H, W]
                
                # Create combined alpha mask for this image
                alpha_mask = torch.zeros((grid_size, grid_size))

                alpha_mask_patches = torch.zeros((effective_grid_size, effective_grid_size), device=device)

                if bboxes is not None:
                    img_bboxes = bboxes[img_idx]  # [n_boxes, 4]
                    alpha_mask = bboxes_to_alpha_mask(img_bboxes, grid_size, patch_size, crop_dim)
                    alpha_mask_patches = bboxes_to_alpha_mask(img_bboxes, effective_grid_size, self.patch_size, self.crop_dim)
                
                if traces is not None:
                    img_traces = traces[img_idx]  # List of traces
                    trace_mask = traces_to_alpha_mask(img_traces, grid_size)
                    alpha_mask = torch.logical_or(alpha_mask, trace_mask).float()
                    alpha_mask_patches = torch.logical_or(alpha_mask_patches, traces_to_alpha_mask(img_traces, effective_grid_size)).float()
                
                # Add batch dimension and move to device: [1, grid_size, grid_size]
                alpha_mask = alpha_mask.unsqueeze(0).to(device)

                alpha_mask_patches = alpha_mask_patches.unsqueeze(0).to(device)  # [1, grid_size, grid_size]
                
                # Process with AlphaClip using the combined mask
                output = self.dino.visual(img, alpha=alpha_mask, return_patches=True)
                
                # Extract features based on configuration
                features = extract_alphaclip_features(output, alpha_mask_patches)
                
                # Caption the extracted features
                ret = self.caption_tokens(features, compute_scores=compute_scores)
                if compute_scores:
                    capt, score = ret
                    controllable_capts.extend(capt)
                else:
                    controllable_capts.extend(ret)
            
            outs['set_controllable_capts'] = controllable_capts
            return outs
        
        # Handle standard bboxes case (separate caption for each bbox)
        if bboxes is not None:
            n_boxes = bboxes.shape[1]
            
            # Process each image and each bbox separately
            all_bbox_capts = []
            all_bbox_scores = [] if compute_scores else None
            all_bbox_sims = [] if return_n_best_sims is not None else None
            
            for img_idx in range(bs):
                img = imgs[img_idx:img_idx+1]  # [1, C, H, W]
                img_bboxes = bboxes[img_idx]  # [n_boxes, 4]
                
                bbox_capts_for_img = []
                bbox_scores_for_img = [] if compute_scores else None
                bbox_sims_for_img = [] if return_n_best_sims is not None else None
                
                for box_idx in range(n_boxes):
                    bbox = img_bboxes[box_idx]  # [4]
                    
                    # Skip dummy boxes (negative values)
                    if bbox.sum().item() < 0:
                        bbox_capts_for_img.append("")  # Empty caption for dummy box
                        if compute_scores:
                            bbox_scores_for_img.append(0.0)
                        if return_n_best_sims is not None:
                            bbox_sims_for_img.append([])
                        continue
                    
                    # Create alpha mask for this bbox
                    alpha_mask = bbox_to_alpha_mask(bbox, grid_size, patch_size, self.crop_dim)
                    alpha_mask = alpha_mask.unsqueeze(0).to(device)  # [1, crop_dim, crop_dim]

                    alpha_mask_patches = bbox_to_alpha_mask(bbox, effective_grid_size, self.patch_size, self.crop_dim)
                    alpha_mask_patches = alpha_mask_patches.unsqueeze(0).to(device)  # [1, effective_grid_size, effective_grid_size]
                    
                    # Process with AlphaClip using the bbox mask
                    output = self.dino.visual(img, alpha=alpha_mask, return_patches=True)
                    
                    # Extract features based on configuration
                    features = extract_alphaclip_features(output, alpha_mask_patches)
                    
                    # Caption the extracted features
                    if return_n_best_sims is None:
                        ret = self.caption_tokens(features, compute_scores=compute_scores)
                        if compute_scores:
                            capt, score = ret
                            bbox_capts_for_img.extend(capt)
                            bbox_scores_for_img.extend(score)
                        else:
                            bbox_capts_for_img.extend(ret)
                    else:
                        ret = self.caption_tokens(features, return_n_best_sims=return_n_best_sims, compute_scores=compute_scores)
                        if compute_scores:
                            (capt, sim), score = ret
                            bbox_capts_for_img.extend(capt)
                            bbox_sims_for_img.extend(sim)
                            bbox_scores_for_img.extend(score)
                        else:
                            capt, sim = ret
                            bbox_capts_for_img.extend(capt)
                            bbox_sims_for_img.extend(sim)
                
                all_bbox_capts.append(bbox_capts_for_img)
                if compute_scores:
                    all_bbox_scores.append(bbox_scores_for_img)
                if return_n_best_sims is not None:
                    all_bbox_sims.append(bbox_sims_for_img)
            
            outs['bbox_capts'] = all_bbox_capts
            if compute_scores:
                outs['bbox_scores'] = all_bbox_scores
            if return_n_best_sims is not None:
                outs['bbox_sims'] = all_bbox_sims
        
        # Handle traces case (separate caption for each trace)
        if traces is not None:
            trace_capts = []
            trace_scores = [] if compute_scores else None
            
            for img_idx in range(bs):
                img = imgs[img_idx:img_idx+1]  # [1, C, H, W]
                trace = traces[img_idx]  # List of traces for this image
                
                # Create alpha mask for this trace
                alpha_mask = trace_to_alpha_mask(trace, grid_size)
                alpha_mask = alpha_mask.unsqueeze(0).to(device)  # [1, crop_size, crop_size]

                alpha_mask_patches = trace_to_alpha_mask(trace, effective_grid_size)
                alpha_mask_patches = alpha_mask_patches.unsqueeze(0).to(device)  # [1, effective_grid_size, effective_grid_size]
                
                # Process with AlphaClip using the trace mask
                output = self.dino.visual(img, alpha=alpha_mask, return_patches=True)
                
                # Extract features based on configuration
                features = extract_alphaclip_features(output, alpha_mask_patches)
                
                # Caption the extracted features
                ret = self.caption_tokens(features, compute_scores=compute_scores)
                if compute_scores:
                    capt, score = ret
                else:
                    capt = ret
                
                trace_capts.extend(capt)
                if compute_scores:
                    trace_scores.extend(score)

            outs['trace_capts'] = trace_capts
            if compute_scores:
                outs['trace_scores'] = trace_scores
        
        # If no bboxes or traces, do standard processing for other caption types
        if bboxes is None and traces is None:
            # Standard AlphaClip processing without alpha mask (whole image)
            output = self.dino.visual(imgs, return_patches=True)  # No alpha parameter = whole image attention

            if get_cls_capt:
                # For CLS captions, always use CLS token regardless of config
                cls_token = output[:, 0, :]
                ret = self.caption_tokens(cls_token, compute_scores=compute_scores)
                if compute_scores:
                    outs['cls_capt'], outs['cls_capt_scores'] = ret
                else:
                    outs['cls_capt'] = ret
            
            if get_avg_patch_capt:
                patch_tokens = output[:, 1:, :]
                avg_patch = compute_region_means(patch_tokens, gaussian_img_variance)
                ret = self.caption_tokens(avg_patch, compute_scores=compute_scores)
                if compute_scores:
                    outs['avg_patch_capt'], outs['avg_patch_capt_scores'] = ret
                else:
                    outs['avg_patch_capt'] = ret
            
            # For get_avg_self_attn_capt, get_attn_heads_capt, get_patch_capts, get_register_capts
            # AlphaClip doesn't provide self-attention access, so we'll use fallback behavior
            if get_avg_self_attn_capt:
                # Use average of patch tokens as fallback
                patch_tokens = output[:, 1:, :]
                avg_self_attn_token = patch_tokens.mean(dim=1)
                ret = self.caption_tokens(avg_self_attn_token, compute_scores=compute_scores)
                if compute_scores:
                    outs['avg_self_attn_capt'], outs['avg_self_attn_capt_scores'] = ret
                else:
                    outs['avg_self_attn_capt'] = ret
            
            if get_attn_heads_capt:
                # Use repeated average patch token as fallback for attention heads
                patch_tokens = output[:, 1:, :]
                avg_patch_token = patch_tokens.mean(dim=1)  # [bs, embed_dim]
                # Repeat for each attention head
                repeated_tokens = avg_patch_token.unsqueeze(1).repeat(1, self.num_attn_heads, 1)  # [bs, num_heads, embed_dim]
                
                ret = self.caption_tokens(repeated_tokens.view(-1, self.embed_dim), compute_scores=compute_scores)
                if compute_scores:
                    attn_heads_capt_unrolled, attn_heads_scores_unrolled = ret
                    outs['attn_heads_capts'] = [attn_heads_capt_unrolled[i * self.num_attn_heads:(i + 1) * self.num_attn_heads] for i in range(bs)]
                    outs['attn_heads_scores'] = [attn_heads_scores_unrolled[i * self.num_attn_heads:(i + 1) * self.num_attn_heads] for i in range(bs)]
                else:
                    attn_heads_capt_unrolled = ret
                    outs['attn_heads_capts'] = [attn_heads_capt_unrolled[i * self.num_attn_heads:(i + 1) * self.num_attn_heads] for i in range(bs)]
            
            if get_patch_capts:
                patch_tokens = output[:, 1:, :]  # [bs, num_patches, embed_dim]
                n_patches = patch_tokens.shape[1]
                
                ret = self.caption_tokens(patch_tokens.reshape(-1, self.embed_dim), compute_scores=compute_scores)
                if compute_scores:
                    patch_tokens_capts_unrolled, patch_tokens_scores_unrolled = ret
                    outs['patch_tokens_capts'] = [patch_tokens_capts_unrolled[i * n_patches:(i + 1) * n_patches] for i in range(bs)]
                    outs['patch_tokens_scores'] = [patch_tokens_scores_unrolled[i * n_patches:(i + 1) * n_patches] for i in range(bs)]
                else:
                    patch_tokens_capts_unrolled = ret
                    outs['patch_tokens_capts'] = [patch_tokens_capts_unrolled[i * n_patches:(i + 1) * n_patches] for i in range(bs)]
            
            if get_register_capts:
                # AlphaClip doesn't have register tokens, return empty lists
                outs['register_capts'] = [[] for _ in range(bs)]
                if compute_scores:
                    outs['register_scores'] = [[] for _ in range(bs)]
        
        return outs

    def caption_bboxes(self, imgs, bboxes, capt_type='cls_capt', crop_boxes=False, compute_scores=False):
        """
        - capt_type : str either 'avg_self_attn_capt' or 'cls_capt'
        """
        device = next(self.parameters()).device
        bs = len(imgs)
        n_bboxes = bboxes.shape[1]
        if not crop_boxes:
            crops = process_bboxes(imgs, bboxes, self.image_transforms_no_crop).to(device)
        else:
            crops = process_bboxes(imgs, bboxes, self.image_transforms).to(device)
            
        n_batch = n_bboxes
        capts = []
        scores = []
        # batching the inference of crops
        for i in range(n_batch):
            start = i * bs
            end = start + bs if i < n_batch - 1 else crops.shape[0]
            forward_out = self.forward(crops[start:end],
                                  get_cls_capt=capt_type == 'cls_capt',
                                  get_avg_self_attn_capt=capt_type == 'avg_self_attn_capt')
            capts += forward_out[capt_type]
            if compute_scores:
                scores += forward_out[f"{capt_type}_scores"]

        # rearranging the captions ensuring shape BS x N_BBOXES
        capts = [capts[i * n_bboxes:(i + 1) * n_bboxes] for i in range(bs)]
        
        ret = {'bbox_capts' : capts}
        
        if compute_scores:
            scores = [scores[i * n_bboxes:(i + 1) * n_bboxes] for i in range(bs)]
            ret['bbox_scores'] = scores
        return ret

    def caption_tokens(self, dino_tokens, project=True, return_n_best_sims=None, compute_scores : bool = False):
        
        if self.viecap is not None:
            if return_n_best_sims:
                raise Exception("return_n_best_sims is not supported with viecap")
            outs = self.viecap.forward(dino_tokens, compute_scores=compute_scores)
            return outs
        
        if self.clipcap is not None:
            if return_n_best_sims:
                raise Exception("return_n_best_sims is not supported with clipcap")
            outs = self.clipcap.forward(dino_tokens, compute_scores=compute_scores)
            return outs
        
        if self.im_proj is None:
            project = False
        if self.calculate_argmax_text:
            # if calculate_argmax_text we return the argmax of the similarities between tokens and memory without using the decoder
            captions = self.im_proj.project(dino_tokens, normalize=self.normalize, return_argmax_text=True, return_n_best_sims=return_n_best_sims)
            return captions if compute_scores is False else (captions, [1.0] * len(captions)) # we return a list of 1.0s as scores
        if not self.embed_inversion:
            # classical decoder forward
            if project:
                projected_outs = self.im_proj.project(dino_tokens, normalize=self.normalize)
            else:
                projected_outs = dino_tokens
            outs = decoding_batched(self.decoder, projected_outs, compute_scores=compute_scores, decoding_method=self.decoding_method)
        else:
            # DINOv2 embedding inversion
            clip_tokens = revert_transformation(self.im_proj.project(dino_tokens, normalize=self.normalize), A_pinv=self.talk2dino_A_pinv, b=self.talk2dino_b)
            outs = decoding_batched(self.decoder, clip_tokens, compute_scores=compute_scores, decoding_method=self.decoding_method)
        return outs

    def ctx_cleaner(self, dirty_embeds : torch.Tensor, ctx_embed : torch.Tensor, cleaning_type='orthogonal_projection', alpha=1.0, epsilon=1e-6):
        if cleaning_type == 'orthogonal_projection':
            #return dirty_embeds - (alpha * (dirty_embeds @ ctx_embed.t() / (torch.norm(ctx_embed, p=2) ** 2))) * ctx_embed
            ctx_embed = ctx_embed.unsqueeze(1)  # [batch_size, 1, embed_dim]
            projection = (dirty_embeds @ ctx_embed.transpose(-1, -2)) / (torch.norm(ctx_embed, dim=-1, keepdim=True) ** 2)
            return dirty_embeds - alpha * projection * ctx_embed
        if cleaning_type == "contrastive_mask":
            ctx_embed = ctx_embed.unsqueeze(1)  # [batch_size, 1, embed_dim]
            ctx_embed_norm = torch.norm(ctx_embed, p=2, dim=2, keepdim=True) + epsilon
            mask = 1 - (ctx_embed / ctx_embed_norm)
            specific_embedding = dirty_embeds * mask
            return specific_embedding

    def analyze_feature_compatibility(self, imgs, analyze_layers=True):
        """
        Analyze compatibility between different layer features and textual embeddings.
        
        Args:
            imgs: Input images tensor
            analyze_layers: Whether to compare layer3 vs layer4 features
            
        Returns:
            Dictionary with compatibility metrics
        """
        device = imgs.device
        results = {}
        
        if self.dino is not None and not (hasattr(self.dino, 'visual') and hasattr(self.dino.visual, 'attnpool')):
            print("Feature compatibility analysis only available for RegionCLIP ResNet models")
            return results
        
        original_patch_size = self.patch_size
        
        with torch.no_grad():
            # Test both layer3 and layer4 if requested
            layer_configs = []
            if analyze_layers:
                layer_configs = [
                    {'patch_size': 16, 'use_layer3': True, 'name': 'layer3'},
                    {'patch_size': 32, 'use_layer3': False, 'name': 'layer4'}
                ]
            else:
                # Use current configuration
                use_layer3 = (self.patch_size == 16)
                layer_configs = [{'patch_size': self.patch_size, 'use_layer3': use_layer3, 'name': f'layer{"3" if use_layer3 else "4"}'}]
            
            for config in layer_configs:
                self.patch_size = config['patch_size']
                
                # Get features from specified layer
                dino_outs = self.dino.visual.forward_return_spatial_feats(imgs, use_layer3=config['use_layer3'])
                features = dino_outs['x_norm_patchtokens']  # [B, num_patches, embed_dim]
                cls_features = dino_outs['x_norm_clstoken']  # [B, embed_dim]
                
                layer_results = {
                    'spatial_resolution': f"{int(features.shape[1]**0.5)}x{int(features.shape[1]**0.5)}",
                    'embed_dim': features.shape[-1],
                    'num_patches': features.shape[1]
                }
                
                if self.im_proj is not None:
                    # Analyze patch features
                    patch_mean = features.mean(dim=1)  # [B, embed_dim] - average across patches
                    projected_patches = self.im_proj.project(patch_mean, normalize=True)
                    
                    # Analyze CLS features
                    if cls_features is not None:
                        projected_cls = self.im_proj.project(cls_features, normalize=True)
                        
                        # Measure similarity to memory bank
                        cls_sims = torch.mm(projected_cls, self.im_proj.embs_dataset.T)
                        patch_sims = torch.mm(projected_patches, self.im_proj.embs_dataset.T)
                        
                        layer_results.update({
                            'cls_max_similarity': torch.max(cls_sims, dim=1)[0].mean().item(),
                            'cls_mean_similarity': torch.mean(cls_sims).item(),
                            'patch_max_similarity': torch.max(patch_sims, dim=1)[0].mean().item(),
                            'patch_mean_similarity': torch.mean(patch_sims).item(),
                            'cls_feature_norm': torch.norm(cls_features, dim=1).mean().item(),
                            'patch_feature_norm': torch.norm(patch_mean, dim=1).mean().item(),
                            'cls_projected_norm': torch.norm(projected_cls, dim=1).mean().item(),
                            'patch_projected_norm': torch.norm(projected_patches, dim=1).mean().item()
                        })
                    
                    # Feature distribution analysis
                    feature_std = torch.std(features.reshape(-1, features.shape[-1]), dim=0).mean().item()
                    projection_std = torch.std(projected_patches, dim=0).mean().item()
                    
                    layer_results.update({
                        'feature_variability': feature_std,
                        'projection_variability': projection_std,
                        'projection_efficiency': projection_std / (feature_std + 1e-8)  # How well projection preserves variability
                    })
                
                results[config['name']] = layer_results
        
        # Restore original configuration
        self.patch_size = original_patch_size
        
        return results
    
    def print_compatibility_analysis(self, analysis_results):
        """Print formatted compatibility analysis results."""
        print("\n" + "="*60)
        print("REGIONCLIP LAYER COMPATIBILITY ANALYSIS")
        print("="*60)
        
        for layer_name, metrics in analysis_results.items():
            print(f"\n{layer_name.upper()} FEATURES:")
            print("-" * 30)
            
            # Basic info
            print(f"Spatial Resolution: {metrics['spatial_resolution']}")
            print(f"Embedding Dimension: {metrics['embed_dim']}")
            print(f"Number of Patches: {metrics['num_patches']}")
            
            if 'cls_max_similarity' in metrics:
                print(f"\nSimilarity to Text Memory Bank:")
                print(f"  CLS Token - Max: {metrics['cls_max_similarity']:.4f}, Mean: {metrics['cls_mean_similarity']:.4f}")
                print(f"  Patch Avg - Max: {metrics['patch_max_similarity']:.4f}, Mean: {metrics['patch_mean_similarity']:.4f}")
                
                print(f"\nFeature Norms:")
                print(f"  CLS Features: {metrics['cls_feature_norm']:.4f}")
                print(f"  Patch Features: {metrics['patch_feature_norm']:.4f}")
                print(f"  CLS Projected: {metrics['cls_projected_norm']:.4f}")
                print(f"  Patch Projected: {metrics['patch_projected_norm']:.4f}")
                
                print(f"\nProjection Quality:")
                print(f"  Feature Variability: {metrics['feature_variability']:.4f}")
                print(f"  Projection Variability: {metrics['projection_variability']:.4f}")
                print(f"  Projection Efficiency: {metrics['projection_efficiency']:.4f}")
        
        if len(analysis_results) == 2:
            layer3_metrics = analysis_results.get('layer3', {})
            layer4_metrics = analysis_results.get('layer4', {})
            
            if 'cls_max_similarity' in layer3_metrics and 'cls_max_similarity' in layer4_metrics:
                print(f"\n{'COMPARISON (Layer3 vs Layer4)':^60}")
                print("-" * 60)
                
                # Similarity comparison
                l3_sim = layer3_metrics['patch_max_similarity']
                l4_sim = layer4_metrics['patch_max_similarity']
                better_sim = "Layer3" if l3_sim > l4_sim else "Layer4"
                print(f"Better Text Similarity: {better_sim} ({max(l3_sim, l4_sim):.4f} vs {min(l3_sim, l4_sim):.4f})")
                
                # Projection efficiency comparison
                l3_eff = layer3_metrics['projection_efficiency']
                l4_eff = layer4_metrics['projection_efficiency']
                better_eff = "Layer3" if l3_eff > l4_eff else "Layer4"
                print(f"Better Projection Efficiency: {better_eff} ({max(l3_eff, l4_eff):.4f} vs {min(l3_eff, l4_eff):.4f})")
                
                # Spatial resolution comparison
                print(f"Spatial Resolution: Layer3 ({layer3_metrics['spatial_resolution']}) vs Layer4 ({layer4_metrics['spatial_resolution']})")


    def __len__(self):
        return sum(p.numel() for p in self.parameters())