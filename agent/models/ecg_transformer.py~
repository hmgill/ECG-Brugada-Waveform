"""Transformer architecture for ECG classification."""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    
    Adds information about the position of each token in the sequence.
    """
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ECGTransformer(nn.Module):
    """
    Transformer-based ECG classifier.
    
    Processes ECG signals by:
    1. Dividing signal into patches (like tokens in NLP)
    2. Embedding each patch
    3. Adding positional information
    4. Running through transformer encoder
    5. Aggregating and classifying
    
    Args:
        in_channels: Number of ECG leads (12 for standard ECG)
        num_classes: Number of output classes (1 for binary)
        patch_size: Number of samples per patch
        d_model: Dimension of transformer embeddings
        nhead: Number of attention heads
        num_layers: Number of transformer encoder layers
        dim_feedforward: Dimension of feedforward network
        dropout: Dropout probability
    """
    
    def __init__(
        self, 
        in_channels=12,
        num_classes=1,
        patch_size=50,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.3
    ):
        super().__init__()
        
        self.patch_size = patch_size
        self.d_model = d_model
        
        # Patch embedding: Convert each patch to a d_model dimensional vector
        # Each patch has (in_channels * patch_size) values
        self.patch_embedding = nn.Linear(in_channels * patch_size, d_model)
        
        # Optional: Use CNN for initial feature extraction before patching
        # This can help capture local patterns within each lead
        self.use_cnn_embedding = True
        if self.use_cnn_embedding:
            self.cnn_embedding = nn.Sequential(
                nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, d_model, kernel_size=7, padding=3),
                nn.BatchNorm1d(d_model),
                nn.ReLU(inplace=True),
            )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # Input shape: (batch, seq, feature)
            norm_first=True    # Pre-norm for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, 
            num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Classification token (like BERT's [CLS])
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights properly for better training."""
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


    def forward(self, x, return_attention=False):
        """
        Forward pass.
        
        Args:
        x: Input tensor of shape (batch_size, in_channels, signal_length)
        return_attention: If True, return attention weights from all layers
        
        Returns:
        If return_attention=False: logits (batch_size, num_classes)
        If return_attention=True: (logits, attention_weights)
            attention_weights is list of (batch, num_heads, seq_len, seq_len) tensors
        """
        batch_size, channels, length = x.shape
    
        if self.use_cnn_embedding:
            # Use CNN to extract features
            x = self.cnn_embedding(x)  # (batch, d_model, length)
        
            # Divide into patches by pooling
            num_patches = length // self.patch_size
            if num_patches == 0:
                num_patches = 1
                self.patch_size = length
        
            # Reshape to patches
            x = x[:, :, :num_patches * self.patch_size]
            x = x.reshape(batch_size, self.d_model, num_patches, self.patch_size)
            
            # Average pool within each patch
            x = x.mean(dim=3)
            
            # Transpose to (batch, num_patches, d_model)
            x = x.transpose(1, 2)
        
        else:
            # Original patching method
            num_patches = length // self.patch_size
            if num_patches == 0:
                num_patches = 1
                self.patch_size = length
        
            # Trim signal to fit patches evenly
            x = x[:, :, :num_patches * self.patch_size]
        
            # Reshape to patches
            x = x.reshape(batch_size, channels, num_patches, self.patch_size)
        
            # Rearrange: (batch, num_patches, channels, patch_size)
            x = x.permute(0, 2, 1, 3)
        
            # Flatten each patch
            x = x.reshape(batch_size, num_patches, -1)
        
            # Embed patches
            x = self.patch_embedding(x)
    
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
    
        # Add positional encoding
        x = self.pos_encoder(x)
    
        # Pass through transformer encoder
        if return_attention:
            attention_weights = []
            
            # Manually iterate through layers to extract attention
            for layer in self.transformer_encoder.layers:
                # Get attention weights from this layer
                x, attn_weights = self._get_attention_from_layer(layer, x)
                attention_weights.append(attn_weights)
        
            # Apply final layer norm if it exists
            if self.transformer_encoder.norm is not None:
                x = self.transformer_encoder.norm(x)
        else:
            # Standard forward pass
            x = self.transformer_encoder(x)
    
        # Use CLS token for classification
        cls_output = x[:, 0]
    
        # Classify
        output = self.classifier(cls_output)
    
        if return_attention:
            return output, attention_weights
        else:
            return output


    def _get_attention_from_layer(self, layer, x):
        """
        Forward through a single transformer layer and extract attention weights.
        
        Args:
        layer: TransformerEncoderLayer
        x: Input tensor (batch, seq_len, d_model)
        
        Returns:
        output: (batch, seq_len, d_model)
        attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        # Save original forward behavior
        # We need to replicate the layer's forward pass but extract attention
        
        # Get layer components
        self_attn = layer.self_attn
        linear1 = layer.linear1
        dropout = layer.dropout
        linear2 = layer.linear2
        norm1 = layer.norm1
        norm2 = layer.norm2
        dropout1 = layer.dropout1
        dropout2 = layer.dropout2
        activation = layer.activation
    
        # Check if using pre-norm or post-norm
        if hasattr(layer, 'norm_first') and layer.norm_first:
            # Pre-norm architecture
            x_norm = norm1(x)
            
            # Self-attention with attention weights
            attn_output, attn_weights = self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False  # Keep per-head weights
            )
        
            x = x + dropout1(attn_output)
        
            # FFN
            x = x + dropout2(linear2(dropout(activation(linear1(norm2(x))))))
        else:
            # Post-norm architecture
            attn_output, attn_weights = self_attn(
                x, x, x,
                need_weights=True,
                average_attn_weights=False
            )
        
            x = norm1(x + dropout1(attn_output))
            x = norm2(x + dropout2(linear2(dropout(activation(linear1(x))))))
    
        return x, attn_weights


    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_num_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    

def create_ecg_transformer(
    model_size='small',
    in_channels=12,
    num_classes=1,
    dropout=0.3
):
    """
    Factory function to create ECG Transformer models.
    
    Args:
        model_size: 'small', 'medium', or 'large'
        in_channels: Number of input channels (12 for ECG)
        num_classes: Number of output classes
        dropout: Dropout probability
        
    Returns:
        ECGTransformer model
    """
    configs = {
        'small': {
            'patch_size': 100,
            'd_model': 128,
            'nhead': 4,
            'num_layers': 3,
            'dim_feedforward': 256
        },
        'medium': {
            'patch_size': 50,
            'd_model': 256,
            'nhead': 8,
            'num_layers': 4,
            'dim_feedforward': 512
        },
        'large': {
            'patch_size': 50,
            'd_model': 512,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    return ECGTransformer(
        in_channels=in_channels,
        num_classes=num_classes,
        patch_size=config['patch_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=dropout
    )
