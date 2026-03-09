"""
ECG Transformer with Rotary Positional Encoding (RoPE) for Multi-Task Learning.

Features:
- RoPE for length-agnostic positional encoding
- Multi-task heads (superclass, subclass)
- Adaptive patching for variable-length inputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Encodes relative positions by rotating query and key vectors.
    Allows the model to handle variable-length sequences naturally.
    
    Reference: Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """
    
    def __init__(self, dim: int, max_seq_len: int = 5000, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)
    
    def _build_cache(self, seq_len: int):
        """Pre-compute rotation matrices."""
        # Inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        
        # Position indices
        t = torch.arange(seq_len).float()
        
        # Outer product: (seq_len, dim/2)
        freqs = torch.outer(t, inv_freq)
        
        # Duplicate for cos and sin: (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin embeddings
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)
        self.cache_len = seq_len
    
    def forward(self, x: torch.Tensor, seq_len: int):
        """
        Get cos and sin embeddings for sequence length.
        
        Args:
            x: Input tensor (for device placement)
            seq_len: Sequence length
            
        Returns:
            cos, sin: Rotation embeddings (1, 1, seq_len, dim)
        """
        if seq_len > self.cache_len:
            self._build_cache(max(seq_len, self.max_seq_len))
            self.cos_cached = self.cos_cached.to(x.device)
            self.sin_cached = self.sin_cached.to(x.device)
        
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )


def rotate_half(x):
    """Rotate half the hidden dims of the input."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to query and key."""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ECGTransformerBlock(nn.Module):
    """Single Transformer Block with RoPE support."""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Layer norms (pre-norm architecture)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """
        Forward pass with RoPE.
        
        Args:
            x: Input (batch, seq_len, d_model)
            cos: Cosine embeddings (1, 1, seq_len, d_model)
            sin: Sine embeddings (1, 1, seq_len, d_model)
            
        Returns:
            Output (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Pre-norm
        residual = x
        x = self.norm1(x)
        
        # Multi-head attention with RoPE
        q = self.q_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout.p if self.training else 0.0
        )
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output = self.out_proj(attn_output)
        
        # Residual connection
        x = residual + self.dropout(attn_output)
        
        # Feedforward with residual
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = residual + x
        
        return x, None


class ECGTransformerRoPE(nn.Module):
    """
    ECG Transformer with RoPE for Multi-Task Learning.
    
    Supports variable-length ECG signals through:
    - Adaptive patching
    - RoPE for relative position encoding
    - Multi-task heads (superclass, subclass)
    """
    
    def __init__(
        self,
        in_channels: int = 12,
        num_superclasses: int = 5,
        num_subclasses: int = 24,  # 23 PTB-XL subclasses + BRUG
        patch_size: int = 50,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # 1. CNN Embedding for local feature extraction
        self.cnn_embedding = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, d_model, kernel_size=7, padding=3),
            nn.BatchNorm1d(d_model),
            nn.ReLU(inplace=True),
        )
        
        # Patch projection
        self.patch_proj = nn.Linear(d_model, d_model)
        
        # 2. RoPE Positional Encoding
        self.rope = RotaryEmbedding(d_model // nhead, max_seq_len=5000)
        
        # 3. Transformer Encoder Layers
        self.layers = nn.ModuleList([
            ECGTransformerBlock(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm_final = nn.LayerNorm(d_model)
        
        # 4. Multi-Task Heads
        self.head_superclass = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_superclasses)
        )
        
        self.head_subclass = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_subclasses)
        )
        
        # REMOVED: self.head_brugada
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input (batch, 12_leads, signal_length)
            return_embeddings: If True, return feature embeddings instead of logits
            
        Returns:
            Dictionary with 'superclass' (B, 5) and 'subclass' (B, 24) logits
        """
        # 1. CNN Embedding
        x = self.cnn_embedding(x)  # (batch, d_model, signal_length)
        
        # 2. Adaptive Patching
        # Use average pooling with stride=patch_size to create patches
        x = F.avg_pool1d(x, kernel_size=self.patch_size, stride=self.patch_size)
        
        # Transpose to (batch, num_patches, d_model)
        x = x.transpose(1, 2)
        
        # Project patches
        x = self.patch_proj(x)
        
        # 3. Apply Transformer with RoPE
        seq_len = x.shape[1]
        cos, sin = self.rope(x, seq_len)
        
        for layer in self.layers:
            x, _ = layer(x, cos, sin)
        
        x = self.norm_final(x)
        
        # 4. Global Average Pooling
        global_embedding = x.mean(dim=1)  # (batch, d_model)
        
        if return_embeddings:
            return global_embedding
        
        # 5. Multi-Task Heads
        return {
            'superclass': self.head_superclass(global_embedding),  # (batch, 5)
            'subclass': self.head_subclass(global_embedding),       # (batch, 24)
            # REMOVED: 'brugada': self.head_brugada(global_embedding)
        }
    
    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_ecg_transformer_rope(model_size: str = 'small', **kwargs) -> ECGTransformerRoPE:
    """
    Factory function to create ECG Transformer models with RoPE.
    
    Args:
        model_size: 'small', 'medium', or 'large'
        **kwargs: Override default parameters (e.g. num_subclasses=24)
        
    Returns:
        ECGTransformerRoPE model
    """
    configs = {
        'small': {
            'patch_size': 50,
            'd_model': 128,
            'nhead': 4,
            'num_layers': 3,
            'dim_feedforward': 256
        },
        'medium': {
            'patch_size': 50,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 512
        },
        'large': {
            'patch_size': 25,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 8,
            'dim_feedforward': 1024
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    config.update(kwargs)  # Allow overrides
    
    return ECGTransformerRoPE(**config)
