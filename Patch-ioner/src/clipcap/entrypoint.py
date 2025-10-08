import torch
from torch import nn
import json
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from typing import List, Optional, Tuple, Union
from argparse import Namespace
from enum import Enum

import torch.nn.functional as nnf

# Import HuggingFace Hub utilities for model loading
from ..hf_utils import load_model_with_hf_fallback

class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


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


class MlpTransformer(nn.Module):
    def __init__(self, in_dim, h_dim, out_d: Optional[int] = None, act=nnf.relu, dropout=0.):
        super().__init__()
        out_d = out_d if out_d is not None else in_dim
        self.fc1 = nn.Linear(in_dim, h_dim)
        self.act = act
        self.fc2 = nn.Linear(h_dim, out_d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):

    def __init__(self, dim_self, dim_ref, num_heads, bias=True, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim_self // num_heads
        self.scale = head_dim ** -0.5
        self.to_queries = nn.Linear(dim_self, dim_self, bias=bias)
        self.to_keys_values = nn.Linear(dim_ref, dim_self * 2, bias=bias)
        self.project = nn.Linear(dim_self, dim_self)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y=None, mask=None):
        y = y if y is not None else x
        b, n, c = x.shape
        _, m, d = y.shape
        # b n h dh
        queries = self.to_queries(x).reshape(b, n, self.num_heads, c // self.num_heads)
        # b m 2 h dh
        keys_values = self.to_keys_values(y).reshape(b, m, 2, self.num_heads, c // self.num_heads)
        keys, values = keys_values[:, :, 0], keys_values[:, :, 1]
        attention = torch.einsum('bnhd,bmhd->bnmh', queries, keys) * self.scale
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            attention = attention.masked_fill(mask.unsqueeze(3), float("-inf"))
        attention = attention.softmax(dim=2)
        out = torch.einsum('bnmh,bmhd->bnhd', attention, values).reshape(b, n, c)
        out = self.project(out)
        return out, attention


class TransformerLayer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        x_, attention = self.attn(self.norm1(x), y, mask)
        x = x + x_
        x = x + self.mlp(self.norm2(x))
        return x, attention

    def forward(self, x, y=None, mask=None):
        x = x + self.attn(self.norm1(x), y, mask)[0]
        x = x + self.mlp(self.norm2(x))
        return x

    def __init__(self, dim_self, dim_ref, num_heads, mlp_ratio=4., bias=False, dropout=0., act=nnf.relu,
                 norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim_self)
        self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)
        self.norm2 = norm_layer(dim_self)
        self.mlp = MlpTransformer(dim_self, int(dim_self * mlp_ratio), act=act, dropout=dropout)


class Transformer(nn.Module):

    def forward_with_attention(self, x, y=None, mask=None):
        attentions = []
        for layer in self.layers:
            x, att = layer.forward_with_attention(x, y, mask)
            attentions.append(att)
        return x, attentions

    def forward(self, x, y=None, mask=None):
        for i, layer in enumerate(self.layers):
            if i % 2 == 0 and self.enc_dec: # cross
                x = layer(x, y)
            elif self.enc_dec:  # self
                x = layer(x, x, mask)
            else:  # self or cross
                x = layer(x, y, mask)
        return x

    def __init__(self, dim_self: int, num_heads: int, num_layers: int, dim_ref: Optional[int] = None,
                 mlp_ratio: float = 2., act=nnf.relu, norm_layer: nn.Module = nn.LayerNorm, enc_dec: bool = False):
        super(Transformer, self).__init__()
        dim_ref = dim_ref if dim_ref is not None else dim_self
        self.enc_dec = enc_dec
        if enc_dec:
            num_layers = num_layers * 2
        layers = []
        for i in range(num_layers):
            if i % 2 == 0 and enc_dec:  # cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            elif enc_dec:  # self
                layers.append(TransformerLayer(dim_self, dim_self, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
            else:  # self or cross
                layers.append(TransformerLayer(dim_self, dim_ref, num_heads, mlp_ratio, act=act, norm_layer=norm_layer))
        self.layers = nn.ModuleList(layers)

class TransformerMapper(nn.Module):
    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        self.transformer = Transformer(dim_embedding, 8, num_layers) #nn.Transformer(d_model=dim_embedding, nhead=8, num_encoder_layers=num_layers)
        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)


class ClipCaptionModel(nn.Module):

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, tokens: torch.Tensor, prefix: torch.Tensor, mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if mapping_type == MappingType.MLP:
            self.clip_project = MLP((prefix_size, (self.gpt_embedding_size * prefix_length) // 2,
                                     self.gpt_embedding_size * prefix_length))
        else:
            self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                  clip_length, num_layers)


class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def generate_batched(
        model,
        tokenizer,
        prefix_embeds,
        entry_length=67,
        top_p=0.8,
        temperature=1.0,
        stop_token: str = '.',
):
    """
    Batched text generation for ClipCap models.
    
    Args:
        model: ClipCap model
        tokenizer: GPT2 tokenizer
        prefix_embeds: (batch_size, prefix_length, embedding_dim) - prefix embeddings
        entry_length: Maximum sequence length to generate
        top_p: Nucleus sampling parameter
        temperature: Sampling temperature
        stop_token: Token to stop generation
    
    Returns:
        List[str]: Generated captions for each item in batch
    """
    model.eval()
    device = next(model.parameters()).device
    batch_size = prefix_embeds.shape[0]
    
    # Initialize
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    
    # Track which sequences are still generating
    active_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)
    
    # Initialize token sequences - start with None
    tokens = None
    generated_embeds = prefix_embeds  # Start with prefix embeddings
    
    with torch.no_grad():
        for step in range(entry_length):
            # Forward pass for all active sequences
            outputs = model.gpt(inputs_embeds=generated_embeds)
            logits = outputs.logits[:, -1, :]  # Get logits for last token: (batch_size, vocab_size)
            
            # Apply temperature
            logits = logits / (temperature if temperature > 0 else 1.0)
            
            # Apply nucleus sampling for each sequence in batch
            for i in range(batch_size):
                if not active_sequences[i]:
                    continue
                    
                # Sort logits for this sequence
                sorted_logits, sorted_indices = torch.sort(logits[i], descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Find indices to remove (above top_p threshold)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                # Set logits to -inf for tokens to remove
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[i, indices_to_remove] = filter_value
            
            # Clamp logits to avoid extreme values
            logits = torch.clamp(logits, min=-1e9, max=1e9)  # keep values bounded
            # Sample next tokens for all sequences
            probs = torch.softmax(logits, dim=-1)

            # if some sequences probs tensor contains NaNs (e.g. all logits were -inf), set stop_token_index prob to 1
            for i in range(batch_size):
                if torch.isnan(probs[i]).all(): #if not torch.isfinite(probs[i]).any() or probs[i].sum() == 0:
                    probs[i] = torch.zeros_like(probs[i])
                    probs[i, stop_token_index] = 1.0

            next_tokens = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)
            
            # Get embeddings for next tokens
            next_token_embeds = model.gpt.transformer.wte(next_tokens)  # (batch_size, 1, embed_dim)
            
            # Update token sequences
            if tokens is None:
                tokens = next_tokens
            else:
                tokens = torch.cat((tokens, next_tokens), dim=1)
            
            # Update generated embeddings
            generated_embeds = torch.cat((generated_embeds, next_token_embeds), dim=1)
            
            # Check for stop tokens and update active sequences
            for i in range(batch_size):
                if active_sequences[i] and next_tokens[i].item() == stop_token_index:
                    active_sequences[i] = False
            
            # If all sequences have stopped, break early
            if not active_sequences.any():
                break
    
    # Decode all sequences
    captions = []
    for i in range(batch_size):
        if tokens is not None:
            token_list = tokens[i].cpu().numpy().tolist()
            # Remove padding and decode
            caption = tokenizer.decode(token_list)
            # Clean up the caption
            caption = caption.split(stop_token)[0] + stop_token
            captions.append(caption)
        else:
            captions.append("")
    
    return captions


def generate2(
        model,
        tokenizer,
        tokens=None,
        prompt=None,
        embed=None,
        entry_count=1,
        entry_length=67,  # maximum number of words
        top_p=0.8,
        temperature=1.,
        stop_token: str = '.',
):
    """
    Legacy single-sequence generation function.
    For new code, use generate_batched instead.
    """
    model.eval()
    generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():

        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):

                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                                                    ..., :-1
                                                    ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


class ClipCapModel(torch.nn.Module):
    """
    ClipCap integration for the Patchioner class.
    """

    def __init__(self, args, device, dino_feature_dim=768):
        super(ClipCapModel, self).__init__()
        args_dict = args.copy()
        self.args = args = self.load_config(args)
        self.device = device
        self.dino_feature_dim = dino_feature_dim
        
        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(args.language_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Determine mapping type
        mapping_type = MappingType.MLP if args.mapping_type.lower() == 'mlp' else MappingType.Transformer
        
        # Initialize model with DINO feature dimensions
        if args.only_prefix:
            self.model = ClipCaptionPrefix(
                prefix_length=args.prefix_length,
                clip_length=args.clip_length,
                prefix_size=dino_feature_dim,
                num_layers=args.num_layers,
                mapping_type=mapping_type
            )
        else:
            self.model = ClipCaptionModel(
                prefix_length=args.prefix_length,
                clip_length=args.clip_length,
                prefix_size=dino_feature_dim,
                num_layers=args.num_layers,
                mapping_type=mapping_type
            )
        
        # Load trained weights with HuggingFace Hub fallback
        print(f"Loading ClipCap weights from: {args.weight_path}")
        try:
            hf_repo_id = getattr(args, 'hf_repo_id', None) or getattr(args, 'weight_path_hf_repo_id', None)
            checkpoint = load_model_with_hf_fallback(
                local_path=args.weight_path,
                hf_repo_id=hf_repo_id,
                map_location=device
            )
            self.model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(f"Warning: Failed to load with HF fallback: {e}")
            # Fallback to original loading method
            checkpoint = torch.load(args.weight_path, map_location=device)
            self.model.load_state_dict(checkpoint, strict=False)
        
        self.model.to(device)
        self.model.eval()

    defaults = {
        "language_model": "gpt2",
        "prefix_length": 10,
        "clip_length": 10,
        "num_layers": 8,
        "mapping_type": "mlp",
        "only_prefix": True,
        "temperature": 1.0,
        "top_p": 0.8,
        "entry_length": 67,
        "stop_token": ".",
        "use_batched_generation": True,  # Use batched generation by default
        "normalize_prefix": False,  # Whether to L2 normalize the input features
        "weight_path": "/raid/datasets/models_weights/clipcap/training-features/clipcap_dino_vitb14_len10_mlp.pt"
    }

    def load_config(self, args_dict: dict) -> Namespace:
        def dict_to_namespace(d):
            if isinstance(d, dict):
                return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
            return d

        # Apply defaults
        for key, value in self.defaults.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    args_dict.setdefault(key, {}).setdefault(sub_key, sub_value)
            else:
                args_dict.setdefault(key, value)

        args = dict_to_namespace(args_dict)
        return args

    def forward(self, dino_features, compute_scores: bool = False) -> List[str]:
        """
        DINO Features: (batch_size, dino_feature_dim)
        - returns: List[str] of generated captions
        """
        if self.args.use_batched_generation:
            return self.forward_batched(dino_features, compute_scores)
        else:
            return self.forward_sequential(dino_features, compute_scores)
    
    def forward_batched(self, dino_features, compute_scores: bool = False) -> List[str]:
        """
        Efficient batched generation for multiple sequences.
        """
        batch_size = dino_features.shape[0]
        
        # Apply normalization if specified (to match training)
        if self.args.normalize_prefix:
            dino_features = dino_features / dino_features.norm(dim=-1, keepdim=True)
        
        # Generate prefix embeddings for entire batch
        with torch.no_grad():
            prefix_embeds = self.model.clip_project(dino_features).view(
                batch_size, self.args.prefix_length, -1
            )
            
            # Generate captions for entire batch
            captions = generate_batched(
                model=self.model,
                tokenizer=self.tokenizer,
                prefix_embeds=prefix_embeds,
                entry_length=self.args.entry_length,
                temperature=self.args.temperature,
                top_p=self.args.top_p,
                stop_token=self.args.stop_token
            )
            
            if compute_scores:
                # Compute perplexity scores for generated captions
                scores = self.compute_perplexity_scores(captions)
                return captions, scores
            else:
                return captions
    
    def forward_sequential(self, dino_features, compute_scores: bool = False) -> List[str]:
        """
        Sequential generation for backward compatibility or debugging.
        """
        batch_size = dino_features.shape[0]
        captions = []
        scores = []

        # Process each feature in the batch sequentially
        for i in range(batch_size):
            feature = dino_features[i:i+1]  # Keep batch dimension
            
            # Apply normalization if enabled
            if self.args.normalize_prefix:
                feature = feature / feature.norm(dim=-1, keepdim=True)
            
            # Generate prefix embeddings
            with torch.no_grad():
                prefix_embed = self.model.clip_project(feature).view(1, self.args.prefix_length, -1)
                
                # Generate caption using legacy function
                caption = generate2(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    embed=prefix_embed,
                    entry_length=self.args.entry_length,
                    temperature=self.args.temperature,
                    top_p=self.args.top_p,
                    stop_token=self.args.stop_token
                )
                
                captions.append(caption)
                if compute_scores:
                    # Compute perplexity for this caption
                    score = self.compute_perplexity_scores([caption])[0]
                    scores.append(score)

        return captions if not compute_scores else (captions, scores)
    
    def compute_perplexity_scores(self, captions: List[str]) -> List[float]:
        """
        Compute perplexity scores for generated captions.
        """
        scores = []
        self.model.eval()
        
        with torch.no_grad():
            for caption in captions:
                try:
                    # Tokenize caption
                    tokens = self.tokenizer.encode(caption, return_tensors='pt').to(self.device)
                    
                    # Compute loss (negative log-likelihood)
                    outputs = self.model.gpt(input_ids=tokens, labels=tokens)
                    loss = outputs.loss
                    
                    # Convert to perplexity (lower is better, but we'll use 1/perplexity as score)
                    perplexity = torch.exp(loss).item()
                    score = 1.0 / perplexity if perplexity > 0 else 1.0
                    scores.append(score)
                except:
                    # Fallback score if computation fails
                    scores.append(1.0)
        
        return scores