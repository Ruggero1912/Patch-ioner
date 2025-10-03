"""
Standalone model components for DenseCLIP to CLIP loader.
Contains all necessary classes without external dependencies.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, Union


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model, n_head, attn_mask=None):
        super().__init__()
        
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x):
        # Create attention mask for current sequence length
        seq_len = x.shape[0]
        if self.attn_mask is not None:
            if seq_len <= self.attn_mask.shape[0]:
                attn_mask = self.attn_mask[:seq_len, :seq_len].to(dtype=x.dtype, device=x.device)
            else:
                # Extend mask for longer sequences
                attn_mask = torch.empty(seq_len, seq_len, device=x.device, dtype=x.dtype)
                attn_mask.fill_(float("-inf"))
                attn_mask.triu_(1)
        else:
            attn_mask = None
        
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPTextEncoder(nn.Module):
    """
    Standard CLIP text encoder implementation for comparison.
    This matches the original CLIP text encoder architecture.
    """
    
    def __init__(self, context_length=77, vocab_size=49408, transformer_width=512,
                 transformer_heads=8, transformer_layers=12, embed_dim=1024, **kwargs):
        super().__init__()
        
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.transformer_width = transformer_width
        self.embed_dim = embed_dim
        
        # Build the transformer layers with proper naming
        self.transformer = self._build_transformer(
            transformer_width, transformer_layers, transformer_heads, context_length
        )
        
        # Text processing components
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)
        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        
        self._initialize_parameters()
    
    def _build_transformer(self, width, layers, heads, context_length):
        """Build transformer layers with causal attention mask"""
        # Create causal attention mask
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        
        # Build transformer blocks with proper naming (resblocks)
        resblocks = nn.Sequential(*[
            ResidualAttentionBlock(width, heads, mask) for _ in range(layers)
        ])
        
        # Create a module that matches CLIP's naming convention
        transformer = nn.Module()
        transformer.resblocks = resblocks
        return transformer
    
    def _initialize_parameters(self):
        """Initialize parameters following CLIP initialization"""
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.transformer_width ** -0.5)
    
    def forward(self, text):
        """
        Forward pass for text encoding
        
        Args:
            text: Tokenized text tensor of shape [batch_size, context_length]
            
        Returns:
            Text features tensor of shape [batch_size, embed_dim]
        """
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)
        
        # Take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        
        return x


class CLIPVisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = nn.Module()
        self.transformer.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads) for _ in range(layers)])

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor, get_patches : bool = False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer.resblocks(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        if not get_patches:
            # returns only the CLS token
            x = self.ln_post(x[:, 0, :])
        else:
            # returns all patch tokens AND the CLS token
            x = self.ln_post(x[:, :, :])
        
        if self.proj is not None:
            x = x @ self.proj

        return x
