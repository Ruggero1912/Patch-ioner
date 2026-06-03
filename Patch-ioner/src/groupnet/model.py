"""
GroupNet: Neural network models for aggregating sets of patch embeddings into a single feature vector.

This module provides three types of aggregation models:
1. AttentionLayer: Single-layer attention-based aggregation
2. TransformerAggregator: Multi-layer transformer encoder with learned aggregation
3. SetTransformer: Set-based aggregation using inducing points and pooling by multihead attention

All models handle variable-length sequences and flexible embedding dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any


class AttentionLayer(nn.Module):
    """
    Single-layer attention-based aggregation using a learnable query token.
    
    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_cls_token: If True, uses a learnable CLS token as query. Otherwise uses self-attention.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_cls_token = use_cls_token
        
        # Learnable CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
            # Cross-attention: CLS queries the patch embeddings
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            # Self-attention followed by mean pooling
            self.self_attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input patch embeddings of shape (batch_size, num_patches, embed_dim)
            mask: Optional attention mask of shape (batch_size, num_patches)
                  True values indicate positions to mask out
        
        Returns:
            Aggregated embedding of shape (batch_size, embed_dim)
        """
        batch_size = x.shape[0]
        
        if self.use_cls_token:
            # Expand CLS token for the batch
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            
            # Cross-attention: CLS attends to patches
            attn_output, _ = self.cross_attn(
                query=cls_tokens,
                key=x,
                value=x,
                key_padding_mask=mask
            )
            
            # Apply normalization and dropout
            output = self.norm(cls_tokens + self.dropout(attn_output))
            
            # Return the aggregated token
            return output.squeeze(1)  # (batch_size, embed_dim)
        else:
            # Self-attention on patches
            attn_output, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=mask
            )
            
            # Apply normalization and dropout
            output = self.norm(x + self.dropout(attn_output))
            
            # Mean pooling over patches
            if mask is not None:
                # Masked mean pooling
                mask_expanded = (~mask).float().unsqueeze(-1)
                output = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                output = output.mean(dim=1)
            
            return output  # (batch_size, embed_dim)


class TransformerAggregator(nn.Module):
    """
    Multi-layer transformer encoder with learned CLS token for aggregation.
    
    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        use_cls_token: If True, uses a learnable CLS token. Otherwise uses mean pooling after transformer.
        positional_encoding: Type of positional encoding ('none', 'learned', 'sinusoidal')
        max_seq_len: Maximum sequence length for learned positional encoding
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        positional_encoding: str = 'none',
        max_seq_len: int = 512,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_cls_token = use_cls_token
        self.positional_encoding_type = positional_encoding
        
        # Learnable CLS token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # Positional encoding
        # TODO: Implement spatial positional encoding that preserves patch positions within bboxes
        if positional_encoding == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len + (1 if use_cls_token else 0), embed_dim))
        elif positional_encoding == 'sinusoidal':
            self.register_buffer('pos_embedding', self._get_sinusoidal_encoding(max_seq_len, embed_dim))
        else:
            self.pos_embedding = None
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def _get_sinusoidal_encoding(self, max_len: int, embed_dim: int) -> torch.Tensor:
        """Generate sinusoidal positional encoding."""
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(1, max_len, embed_dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input patch embeddings of shape (batch_size, num_patches, embed_dim)
            mask: Optional attention mask of shape (batch_size, num_patches)
                  True values indicate positions to mask out
        
        Returns:
            Aggregated embedding of shape (batch_size, embed_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Adjust mask for CLS token
            if mask is not None:
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Add positional encoding
        if self.pos_embedding is not None:
            if self.positional_encoding_type == 'learned':
                x = x + self.pos_embedding[:, :x.shape[1], :]
            elif self.positional_encoding_type == 'sinusoidal':
                x = x + self.pos_embedding[:, :x.shape[1], :].to(x.device)
        
        # Apply transformer encoder
        # PyTorch's TransformerEncoder expects src_key_padding_mask
        output = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Extract aggregated representation
        if self.use_cls_token:
            # Return CLS token
            return self.norm(output[:, 0, :])  # (batch_size, embed_dim)
        else:
            # Mean pooling over all tokens
            if mask is not None:
                mask_expanded = (~mask).float().unsqueeze(-1)
                output = (output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                output = output.mean(dim=1)
            return self.norm(output)


class SetTransformer(nn.Module):
    """
    Set Transformer for aggregating sets of embeddings using Inducing Points and
    Pooling by Multihead Attention (PMA).
    
    Based on "Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks"
    (Lee et al., ICML 2019)
    
    Args:
        embed_dim: Dimension of input embeddings
        num_heads: Number of attention heads
        num_layers: Number of Set Attention Blocks (SAB)
        num_inducing_points: Number of inducing points for ISAB
        num_seed_vectors: Number of seed vectors for PMA (output aggregation)
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
        use_isab: If True, uses Induced Set Attention Block. Otherwise uses standard SAB.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        num_layers: int = 3,
        num_inducing_points: int = 32,
        num_seed_vectors: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_isab: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_inducing_points = num_inducing_points
        self.num_seed_vectors = num_seed_vectors
        self.use_isab = use_isab
        
        # Build encoder: stack of SAB or ISAB layers
        self.encoder = nn.ModuleList()
        for _ in range(num_layers):
            if use_isab:
                self.encoder.append(
                    ISAB(embed_dim, num_heads, num_inducing_points, dim_feedforward, dropout)
                )
            else:
                self.encoder.append(
                    SAB(embed_dim, num_heads, dim_feedforward, dropout)
                )
        
        # Pooling by Multihead Attention
        self.pma = PMA(embed_dim, num_heads, num_seed_vectors, dim_feedforward, dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input patch embeddings of shape (batch_size, num_patches, embed_dim)
            mask: Optional attention mask of shape (batch_size, num_patches)
                  True values indicate positions to mask out
        
        Returns:
            Aggregated embedding of shape (batch_size, embed_dim) if num_seed_vectors=1,
            otherwise (batch_size, num_seed_vectors, embed_dim)
        """
        # Apply encoder layers
        for layer in self.encoder:
            x = layer(x, mask=mask)
        
        # Apply PMA pooling
        output = self.pma(x, mask=mask)
        
        # If single seed vector, squeeze the dimension
        if self.num_seed_vectors == 1:
            return output.squeeze(1)  # (batch_size, embed_dim)
        else:
            return output  # (batch_size, num_seed_vectors, embed_dim)


class SAB(nn.Module):
    """Set Attention Block: Multi-head self-attention with residual connection and feedforward."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.mab = MAB(embed_dim, embed_dim, embed_dim, num_heads, dim_feedforward, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.mab(x, x, mask=mask)


class ISAB(nn.Module):
    """
    Induced Set Attention Block: Uses inducing points to reduce computational complexity
    from O(n^2) to O(mn) where m is the number of inducing points and n is the set size.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_inducing_points: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing_points, embed_dim))
        self.mab1 = MAB(embed_dim, embed_dim, embed_dim, num_heads, dim_feedforward, dropout)
        self.mab2 = MAB(embed_dim, embed_dim, embed_dim, num_heads, dim_feedforward, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        inducing = self.inducing_points.expand(batch_size, -1, -1)
        
        # H = MAB(I, X): Inducing points attend to input
        h = self.mab1(inducing, x, mask=mask)
        
        # Output = MAB(X, H): Input attends to processed inducing points
        return self.mab2(x, h)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention: Uses seed vectors to aggregate set elements.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_seed_vectors: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.seed_vectors = nn.Parameter(torch.randn(1, num_seed_vectors, embed_dim))
        self.mab = MAB(embed_dim, embed_dim, embed_dim, num_heads, dim_feedforward, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.shape[0]
        seeds = self.seed_vectors.expand(batch_size, -1, -1)
        
        # Seeds attend to input set
        return self.mab(seeds, x, mask=mask)


class MAB(nn.Module):
    """
    Multihead Attention Block: Core building block for Set Transformer.
    Implements MAB(X, Y) = LayerNorm(H + rFF(H)) where H = LayerNorm(X + Multihead(X, Y, Y))
    """
    
    def __init__(
        self,
        dim_q: int,
        dim_k: int,
        dim_v: int,
        num_heads: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dim_v = dim_v
        self.num_heads = num_heads
        
        self.fc_q = nn.Linear(dim_q, dim_v)
        self.fc_k = nn.Linear(dim_k, dim_v)
        self.fc_v = nn.Linear(dim_k, dim_v)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=dim_v,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.fc_o = nn.Linear(dim_v, dim_v)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(dim_v, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, dim_v),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(dim_v)
        self.norm2 = nn.LayerNorm(dim_v)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Query input (batch_size, len_q, dim_q)
            y: Key/Value input (batch_size, len_k, dim_k)
            mask: Optional mask for y (batch_size, len_k)
        """
        # Project inputs
        q = self.fc_q(x)
        k = self.fc_k(y)
        v = self.fc_v(y)
        
        # Multi-head attention
        attn_output, _ = self.attn(q, k, v, key_padding_mask=mask)
        
        # First residual connection with LayerNorm
        h = self.norm1(x + self.dropout(self.fc_o(attn_output)))
        
        # Feedforward with second residual connection
        output = self.norm2(h + self.ff(h))
        
        return output


class ResidualCentroidGroupNet(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        residual_model_type: str = 'attention',
        residual_config: Optional[Dict[str, Any]] = None,
        gate_init_logit: float = -2.0,   # sigmoid(-8) ~ 0.0003
        gate_type: str = "global",       # "global" | "per_sample" | "per_dim"
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.residual_model_type = residual_model_type
        self.gate_type = gate_type

        if residual_config is None:
            residual_config = {'embed_dim': embed_dim, 'num_heads': 8, 'dropout': 0.1}
        else:
            residual_config['embed_dim'] = embed_dim

        # --- build residual branch (same as yours) ---
        if residual_model_type == 'attention':
            self.residual_branch = AttentionLayer(
                embed_dim=embed_dim,
                num_heads=residual_config.get('num_heads', 8),
                dropout=residual_config.get('dropout', 0.1),
                use_cls_token=residual_config.get('use_cls_token', True),
            )
        elif residual_model_type == 'transformer':
            self.residual_branch = TransformerAggregator(
                embed_dim=embed_dim,
                num_heads=residual_config.get('num_heads', 8),
                num_layers=residual_config.get('num_layers', 3),
                dim_feedforward=residual_config.get('dim_feedforward', 2048),
                dropout=residual_config.get('dropout', 0.1),
                use_cls_token=residual_config.get('use_cls_token', True),
                positional_encoding=residual_config.get('positional_encoding', 'none'),
                max_seq_len=residual_config.get('max_seq_len', 512),
            )
        elif residual_model_type == 'set_transformer':
            self.residual_branch = SetTransformer(
                embed_dim=embed_dim,
                num_heads=residual_config.get('num_heads', 8),
                num_layers=residual_config.get('num_layers', 3),
                num_inducing_points=residual_config.get('num_inducing_points', 32),
                num_seed_vectors=residual_config.get('num_seed_vectors', 1),
                dim_feedforward=residual_config.get('dim_feedforward', 2048),
                dropout=residual_config.get('dropout', 0.1),
                use_isab=residual_config.get('use_isab', True),
            )
        else:
            raise ValueError(f"Unknown residual model type: {residual_model_type}")

        # --- gate params ---
        if gate_type == "global":
            self.gate_logit = nn.Parameter(torch.tensor(gate_init_logit))
        elif gate_type == "per_dim":
            self.gate_logit = nn.Parameter(torch.full((embed_dim,), gate_init_logit))
        elif gate_type == "per_sample":
            # make gate from centroid (one scalar per sample)
            self.gate_mlp = nn.Linear(embed_dim, 1)
            nn.init.zeros_(self.gate_mlp.weight)
            nn.init.constant_(self.gate_mlp.bias, gate_init_logit)
        else:
            raise ValueError("gate_type must be: global | per_dim | per_sample")

    def compute_centroid(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            mask_expanded = (~mask).float().unsqueeze(-1)
            centroid = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            centroid = x.mean(dim=1)
        return centroid

    def _compute_gate(self, centroid: torch.Tensor) -> torch.Tensor:
        if self.gate_type == "global":
            g = torch.sigmoid(self.gate_logit)              # scalar
            return g
        if self.gate_type == "per_dim":
            g = torch.sigmoid(self.gate_logit)              # (D,)
            return g
        # per_sample
        g = torch.sigmoid(self.gate_mlp(centroid))          # (B,1)
        return g

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        centroid = self.compute_centroid(x, mask)           # (B,D)
        residual = self.residual_branch(x, mask)            # (B,D)

        g = self._compute_gate(centroid)

        # broadcast works for scalar, (D,), or (B,1)
        out = centroid + g * residual
        return out
    
    def _initialize_residual_to_zero(self):
        """
        Initialize the residual branch weights so that it outputs zero vectors
        at the start of training. This ensures the model initially predicts
        exactly the centroid.
        """
        # Zero out the final linear layer(s) that produce the output
        for name, module in self.residual_branch.named_modules():
            # For AttentionLayer: zero out the output projection of cross_attn
            if isinstance(module, nn.MultiheadAttention):
                if hasattr(module, 'out_proj'):
                    nn.init.zeros_(module.out_proj.weight)
                    if module.out_proj.bias is not None:
                        nn.init.zeros_(module.out_proj.bias)
            
            # For TransformerEncoder layers: zero out the final layer
            if isinstance(module, nn.TransformerEncoder):
                # Zero out the last encoder layer's output
                last_layer = module.layers[-1]
                # Zero out the feedforward output
                if hasattr(last_layer, 'linear2'):
                    nn.init.zeros_(last_layer.linear2.weight)
                    if last_layer.linear2.bias is not None:
                        nn.init.zeros_(last_layer.linear2.bias)
            
            # For SetTransformer: zero out PMA output
            if isinstance(module, nn.Linear) and 'pma' in name.lower():
                # This will catch the final linear layers in PMA
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    

class GroupNet(nn.Module):
    """
    Wrapper class that instantiates the appropriate aggregation model based on config.
    
    Args:
        config: Configuration dictionary or object
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        self.config = config
        model_type = config.get('model_type', 'attention').lower()
        embed_dim = config['embed_dim']
        
        if model_type == 'attention':
            self.model = AttentionLayer(
                embed_dim=embed_dim,
                num_heads=config.get('num_heads', 8),
                dropout=config.get('dropout', 0.1),
                use_cls_token=config.get('use_cls_token', True),
            )
        elif model_type == 'transformer':
            self.model = TransformerAggregator(
                embed_dim=embed_dim,
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 3),
                dim_feedforward=config.get('dim_feedforward', 2048),
                dropout=config.get('dropout', 0.1),
                use_cls_token=config.get('use_cls_token', True),
                positional_encoding=config.get('positional_encoding', 'none'),
                max_seq_len=config.get('max_seq_len', 512),
            )
        elif model_type == 'set_transformer':
            self.model = SetTransformer(
                embed_dim=embed_dim,
                num_heads=config.get('num_heads', 8),
                num_layers=config.get('num_layers', 3),
                num_inducing_points=config.get('num_inducing_points', 32),
                num_seed_vectors=config.get('num_seed_vectors', 1),
                dim_feedforward=config.get('dim_feedforward', 2048),
                dropout=config.get('dropout', 0.1),
                use_isab=config.get('use_isab', True),
            )
        elif model_type == 'residual_centroid':
            # Two-branch model: centroid + residual
            residual_model_type = config.get('residual_model_type', 'attention')
            residual_config = config.get('residual_config', {})
            self.model = ResidualCentroidGroupNet(
                embed_dim=embed_dim,
                residual_model_type=residual_model_type,
                residual_config=residual_config,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Choose from 'attention', 'transformer', 'set_transformer', or 'residual_centroid'.")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input patch embeddings of shape (batch_size, num_patches, embed_dim)
            mask: Optional attention mask of shape (batch_size, num_patches)
                  True values indicate positions to mask out
        
        Returns:
            Aggregated embedding of shape (batch_size, embed_dim)
        """
        return self.model(x, mask=mask)
