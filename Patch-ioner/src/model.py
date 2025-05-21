import timm
import torch
import torch.nn as nn
import math
import yaml
import os
import pickle
import random
import torchvision.transforms as T


from decap import decoding_batched
from decap import get_decap_model
from dino_extraction import get_self_attention, process_self_attention, transform_to_standard_dino_out, get_layer_n_output, feats
from im2txtprojection.im2txtprojection import Im2TxtProjector, ProjectionType
from src.bbox_utils import extract_bboxes_feats, extract_bboxes_feats_double_dino, process_bboxes, map_traces_to_grid
from transformers import GPT2LMHeadModel
from src.embedding_utils import get_pseudo_inverse, revert_transformation
from typing import Tuple


import math
from tqdm import tqdm

import torch.nn.functional as F
from open_clip_proxy import create_model, tokenizer

DECODER_CONFIG_PATH = os.path.join("./decoder_config.pkl")


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



class MLP(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

class DeCap(nn.Module):

    def __init__(self,prefix_size: int = 768):
        super(DeCap, self).__init__()

        # decoder: 4 layers transformer with 4 attention heads
        # the decoder is not pretrained
        with open(DECODER_CONFIG_PATH,'rb') as f:
            config = pickle.load(f)
        self.decoder = GPT2LMHeadModel(config)
        self.embedding_size = self.decoder.transformer.wte.weight.shape[1]
        self.clip_project = MLP((prefix_size,self.embedding_size))
        
    def forward(self, clip_features,gpt_tokens):
        embedding_text = self.decoder.transformer.wte(gpt_tokens)
        embedding_clip = self.clip_project(clip_features)
        embedding_clip = embedding_clip.reshape(-1,1,self.embedding_size)
        embedding_cat = torch.cat([embedding_clip,embedding_text],dim=1)
        out = self.decoder(inputs_embeds=embedding_cat)
        return out


class ProjectionLayer(nn.Module):
    """
    Creates a projection layer on top of the CLIP-text encoder.
    The forward method calculate the similarity between the DINO CLS token and the projected CLIP textual CLS token. 
    """
    def __init__(self, act=nn.Tanh(), hidden_layer=False, cosine=True, dino_embed_dim=1024, clip_embed_dim=512, num_attn_head=16, weight_attn_heads=None,
                 alignment_strategy='max_score', alpha=0.6, keep_cls=False, keep_end_seq=False):
        # mlp_dims list of mlp dimensions
        super().__init__()
        self.num_attn_head = num_attn_head      
        
        self.linear_layer = nn.Linear(clip_embed_dim, dino_embed_dim)
        if hidden_layer:
            hidden_layer = 1 if hidden_layer is True else hidden_layer # ensuring compatibility with old code
            # self.linear_layer2 = nn.Linear(dino_embed_dim, dino_embed_dim) 
            self.hidden_layers = nn.ModuleList([nn.Linear(dino_embed_dim, dino_embed_dim) for _ in range(hidden_layer)])
        self.act = act
        self.cosine = cosine
        
        self.weight_attn_heads = weight_attn_heads
        if weight_attn_heads == 'static':
            self.attn_weights = nn.Parameter(torch.rand(self.num_attn_head))
        elif weight_attn_heads == 'conditioned':
            self.weight_layer1 = nn.Linear(dino_embed_dim, dino_embed_dim)
            self.weight_layer2 = nn.Linear(dino_embed_dim, self.num_attn_head)
            
        self.alignment_strategy = alignment_strategy # relevant only if we use disentangled_self_attn
        self.keep_cls = keep_cls # relevant only if we use clip_txt_tokens_out
        self.keep_end_seq = keep_end_seq # relevant only if we use clip_txt_tokens_out
        self.alpha = alpha
    
    @classmethod
    def from_config(cls, config):
        if type(config) is str:
            # if the configuration is a string, we treat it as a file path
            with open(config, 'r') as f:
                config = yaml.safe_load(f)['model']
        
        # loading the activation function
        act = config.get('act', None)
        if act == 'tanh':
            act = nn.Tanh()
        elif act == 'relu':
            act = nn.ReLU()
        elif act == 'sigmoid':
            act = nn.Sigmoid()
        elif act is not None:
            raise Exception("Unknown activation function")
        
        model = cls(
            act=act,
            hidden_layer=config.get('hidden_layer', False),
            cosine=config.get('cosine', True),
            dino_embed_dim=config.get('dino_embed_dim', 1024),
            num_attn_head=config.get('num_attn_head', 16),
            clip_embed_dim=config.get('clip_embed_dim', 512),
            weight_attn_heads=config.get('weight_attn_heads', None),
            alignment_strategy=config.get('alignment_strategy', 'max_score'),
            alpha=config.get('alpha', 0.6),
            keep_cls=config.get('keep_cls', None),
            keep_end_seq=config.get('keep_end_seq', None),
        )
        if config.get('starting_checkpoint', None) is not None:
            model.load_state_dict(torch.load(config['starting_checkpoint'], 'cpu'))
        
        return model
    
    def project_clip_txt(self, textual_embedding):
        textual_embedding = textual_embedding.float()
        x = self.linear_layer(textual_embedding)
        
        if hasattr(self, 'hidden_layers'):
            for hidden_layer in self.hidden_layers:
                if self.act:
                    x = self.act(x)
                x = hidden_layer(x)
            
        return x
    def load_state_dict(self, state_dict, strict=True):
        # compatibility with old code
        if 'linear_layer2.weight' in state_dict:
            state_dict['hidden_layers.0.weight'] = state_dict.pop('linear_layer2.weight')
            state_dict['hidden_layers.0.bias'] = state_dict.pop('linear_layer2.bias')
        # Call the parent class's load_state_dict with the modified state_dict
        super(ProjectionLayer, self).load_state_dict(state_dict, strict)
    
    def set_alignment_strategy(self, alignment_strategy):
        self.alignment_strategy = alignment_strategy
        return
    
    def __len__(self):
        return sum(p.numel() for p in self.parameters())  

class ProxyCLIP(nn.Module):
    def __init__(self, clip_type, model_type, vfm_model, device=torch.device('cuda'), beta=1.2, gamma=3.0, slide_crop=336):

        super().__init__()

        self.clip = create_model(model_type, pretrained=clip_type, precision='fp16')
        self.clip.eval().to(device)
        self.tokenizer = tokenizer.tokenize

        self.vfm_model = vfm_model

        if vfm_model == 'dino':
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')
            # self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
            self.vfm = torch.hub.load('facebookresearch/dino:main', 'dino_vitb8')

        elif vfm_model == 'dinov2':
            # self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
            self.vfm = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')

        self.vfm = self.vfm.half()
        for p in self.vfm.parameters():
            p.requires_grad = False
        self.vfm.eval().to(device)

        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        self.slide_crop = slide_crop
        self.beta = beta
        self.gamma = gamma

    @torch.no_grad()
    def forward(self, img):
        if type(img) == list:
            img = img[0]

        clip_token_size = img.shape[-2] // self.clip.visual.patch_size[0], img.shape[-1] // self.clip.visual.patch_size[1]

        # imgs_norm = [self.norm(self.unnorm(img[i])) for i in range(len(img))]
        # imgs_norm = torch.stack(imgs_norm, dim=0)
        imgs_norm = img

        imgs_norm = imgs_norm.half()
        if self.vfm_model == 'dino':
            feat_out = {}
            def hook_fn_forward_qkv(module, input, output):
                feat_out["qkv"] = output
            self.vfm._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(
                hook_fn_forward_qkv)

            # Forward pass in the model
            feat = self.vfm.get_intermediate_layers(imgs_norm)[0]

            nb_im = feat.shape[0]  # Batch size

            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm[0].shape[-2] // patch_size, imgs_norm[0].shape[-2] // patch_size
            ex_feats = feat[:, 1:, :].reshape(nb_im, I, J, -1).permute(0, 3, 1, 2)

        elif self.vfm_model == 'dinov2':
            patch_size = self.vfm.patch_embed.patch_size
            I, J = imgs_norm.shape[-2] // patch_size[0], imgs_norm.shape[-2] // patch_size[1]
            ex_feats = self.vfm.get_intermediate_layers(imgs_norm, reshape=True)[0]

        else:
            I, J = clip_token_size
            ex_feats = None

        image_features = self.clip.encode_image(img.half(),
                                               external_feats=ex_feats,
                                               beta=self.beta,
                                               gamma=self.gamma)

        image_features /= image_features.norm(dim=-1, keepdim=True)

        
        
        return {
            'x_norm_patchtokens': image_features.float()
        }



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
    def __init__(self, projection_type, decoder_weights, device, prefix_size, linear_talk2dino, support_memory_size,
                 dino_model=None, proxyclip_clipmodel=None, proxyclip_vfm=None, use_talk2dino_project=True, normalize=True, attention_type='qkv', talk2dino_config=None, 
                 talk2dino_weights=None, resize_dim=518, crop_dim=518, talk2dino_attn_type='qkv', calculate_argmax_text=False,
                 online_texts=None, clip_model_name=None, use_open_clip=False, viecap_config=None):
        super().__init__()


        self.decoding_method = None

        if viecap_config is not None:
            if viecap_config.get('meacap', False):
                from src.meacap.entrypoint import MeaCap
                self.viecap = MeaCap(viecap_config, device, clip_model_name)
            else:
                from src.viecap.entrypoint import VieCap
                self.viecap = VieCap(viecap_config, device, clip_model_name)
        else:
            self.viecap = None

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
        elif os.path.exists(projection_type):
            print(f"Loading memory bank from {projection_type}")
            projection_type_enum = projection_type
        else:
            raise Exception("The projection_type field must be 'coco', 'msmarco', 'blip' or 'vg'")

        self.calculate_argmax_text = calculate_argmax_text
        if not self.calculate_argmax_text:
            self.decoder = get_decap_model(device, decoder_weights, prefix_size)
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
                use_open_clip=use_open_clip
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
            else:
                raise Exception("Unknown patch size")

            self.num_patch_tokens = crop_dim // patch_size * crop_dim // patch_size
            self.num_tokens = self.num_global_tokens + self.num_patch_tokens
            
            if 'vitl' in dino_model or 'vit_large' in dino_model or 'ViT-L' in dino_model or 'ViT-H' in dino_model:
                self.embed_dim = 1024
            elif 'vitb' in dino_model or 'vit_base' in dino_model or 'ViT-B' in dino_model:
                self.embed_dim = 768
            elif 'vits' in dino_model or 'vit_small' in dino_model:
                self.embed_dim = 384
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
                # we use this case to test DeCap original architecture (with CLIP instead of DINOv2)

                # timm uses GELU while original OpenAI model uses QuickGELU
                # https://github.com/huggingface/pytorch-image-models/issues/1754
                # we fix the activation function because DeCap is trained using OpenAI interface
                class QuickGELU(torch.nn.Module):
                    def forward(self, x: torch.Tensor):
                        return x * torch.sigmoid(1.702 * x)
                
                self.dino = timm.create_model(dino_model, pretrained=True, act_layer=QuickGELU, img_size=resize_dim)
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
            self.dino.eval()
            if hasattr(self.dino, 'blocks'):
                self.dino.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)

            # need patch_size
            if 'dino' in dino_model:
                self.patch_size = self.dino.patch_size
            elif 'openai' in dino_model:
                # in the case self.dino is a timm model, we need to get patch_size from 
                # the model's configuration
                # should get patch size from dino_model, which is a string with the following format:
                # 'vit_base_patch32_clip_224.openai'
                self.patch_size = int(dino_model.split('_')[2].replace('patch', ''))         

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



    @classmethod
    def from_config(cls, config, device='cpu', online_texts=None):
        if type(config) is str:
            # if the configuration is a string, we treat it as a file path
            with open(config, 'r') as f:
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
            viecap_config=config.get('viecap', None)
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

        if bboxes is not None and double_DINO_for_bboxes:
            self.dino.blocks[-1].attn.qkv.register_forward_hook(get_self_attention)
            self.dino.blocks[-1].register_forward_hook(get_layer_n_output)

        if hasattr(self.dino, 'visual') and hasattr(self.dino.visual, 'transformer'):
            # Attach hook to the visual transformer
            hook_handle = self.dino.visual.transformer.register_forward_hook(save_patch_embeddings)

        if caption_bboxes_type is not None:
            return self.caption_bboxes(imgs, bboxes, caption_bboxes_type, compute_scores=compute_scores)

        if 'dino' in self.model_name:
            dino_outs = self.dino(imgs, is_training=True)
        elif self.model_name == 'proxyclip':
            dino_outs = self.proxyclip(imgs)
        else:
            if hasattr(self.dino, 'blocks'):
                # using timm interface
                output = self.dino.forward_features(imgs)
                # projecting 768 -> 512
                output = self.dino.head(output)
            else:
                # using open_clip interface
                output = self.dino.visual(imgs)
                output = patch_embeddings['full']

                output = output @ self.dino.visual.proj  # shape (B, N_patches, output_dim)
                

            # reporting output in DINOv2 format
            dino_outs = {
                'x_norm_clstoken': output[:, 0, :],
                'x_norm_patchtokens': output[:, 1:, :],
            }
        
        if self.model_name != 'proxyclip' and 'self_attn' in feats:
            self_attn, self_attn_maps = process_self_attention(feats['self_attn'], imgs.shape[0], self.num_tokens, self.num_attn_heads, self.embed_dim, self.scale, self.num_global_tokens, ret_self_attn_maps=True)
            avg_self_attn_token = (self_attn.unsqueeze(-1) * dino_outs['x_norm_patchtokens']).mean(dim=1)


            self_attn_maps = self_attn_maps.softmax(dim=-1)
            disentangled_self_attn = (dino_outs['x_norm_patchtokens'].unsqueeze(1) * self_attn_maps.unsqueeze(-1)).mean(dim=2)

        if cleaning_type is not None:
            batch_patchtokens = dino_outs['x_norm_patchtokens']
            batch_clean_from_token = dino_outs['x_norm_clstoken'] if clean_from == "cls" else avg_self_attn_token
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
                bbox_attn_maps = self_attn.cpu() if use_attn_map_for_bboxes else None
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
        elif bboxes is not None and get_controllable_capts:
            bbox_attn_maps = self_attn.cpu() if use_attn_map_for_bboxes else None
            n_boxes = bboxes.shape[1]
            bbox_feats = extract_bboxes_feats(dino_outs['x_norm_patchtokens'], bboxes, gaussian_avg=gaussian_avg, gaussian_bbox_variance=gaussian_bbox_variance, get_single_embedding_per_image=True, patch_size=self.patch_size, attention_map=bbox_attn_maps)
            
            outs['set_controllable_capts'] = self.caption_tokens(bbox_feats)
        
        if traces is not None:
            n_patches = int(dino_outs['x_norm_patchtokens'].shape[1] ** 0.5)
            relevant_patches = torch.stack([map_traces_to_grid(trace, n_patches) for trace in traces], dim=0).to(next(self.parameters()).device)
            if use_attention_tracing:
                relevant_patches = (self_attn.view(relevant_patches.shape) * relevant_patches)
            trace_embeds = (relevant_patches.unsqueeze(-1) * dino_outs['x_norm_patchtokens'].view(bs, n_patches, n_patches, embed_dim)).mean(dim=(1,2))
            
            outs['trace_capts'] = self.caption_tokens(trace_embeds)

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


    def __len__(self):
        return sum(p.numel() for p in self.parameters())