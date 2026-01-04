"""
Model Components for MER2LATEX
================================
Reusable components: encoders, decoders, attention mechanisms, positional encodings.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import timm

from src.utils.constants import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    NUM_CHANNELS,
    MAX_SEQ_LENGTH
)


# ============================================================================
# POSITIONAL ENCODINGS
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class PositionalEncoding2D(nn.Module):
    """2D positional encoding for image features."""
    
    def __init__(self, d_model: int, height: int, width: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(height, width, d_model)
        
        y_pos = torch.arange(0, height).unsqueeze(1).float()
        x_pos = torch.arange(0, width).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        
        pe[:, :, 0:d_model//2:2] = torch.sin(y_pos * div_term).unsqueeze(1).expand(-1, width, -1)
        pe[:, :, 1:d_model//2:2] = torch.cos(y_pos * div_term).unsqueeze(1).expand(-1, width, -1)
        pe[:, :, d_model//2::2] = torch.sin(x_pos * div_term).unsqueeze(0).expand(height, -1, -1)
        pe[:, :, d_model//2+1::2] = torch.cos(x_pos * div_term).unsqueeze(0).expand(height, -1, -1)
        
        pe = pe.view(-1, d_model).unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


# ============================================================================
# CNN ENCODER (for Model A)
# ============================================================================

class CNNEncoder(nn.Module):
    """CNN backbone for feature extraction."""
    
    def __init__(self, input_channels: int = 1, hidden_dim: int = 256):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # Block 1: (1, 128, 768) -> (64, 64, 384)
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2: (64, 64, 384) -> (128, 32, 192)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3: (128, 32, 192) -> (256, 16, 96)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Block 4: (256, 16, 96) -> (512, 8, 96)
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            
            # Block 5: (512, 8, 96) -> (512, 4, 96)
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)  # (B, 512, H', W')
        
        # Reshape for sequence: (B, 512, H', W') -> (B, W', 512*H')
        B, C, H, W = features.shape
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, W', C, H')
        features = features.view(B, W, -1)  # (B, W', C*H')
        
        return features


# ============================================================================
# RESNET ENCODER (for Model B)
# ============================================================================

class ResNetEncoder(nn.Module):
    """ResNet-based encoder using timm."""
    
    def __init__(self, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        
        self.backbone = timm.create_model(
            'resnet18',
            pretrained=pretrained,
            in_chans=NUM_CHANNELS,
            features_only=True,
            out_indices=[4]
        )
        
        self.proj = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)[0]  # (B, 512, H', W')
        features = self.proj(features)  # (B, hidden_dim, H', W')
        
        B, D, H, W = features.shape
        features = features.flatten(2).permute(0, 2, 1)  # (B, T, D)
        
        return features


# ============================================================================
# VIT ENCODER (for Model C)
# ============================================================================

class ViTEncoder(nn.Module):
    """Vision Transformer encoder using timm."""
    
    def __init__(self, hidden_dim: int = 256, pretrained: bool = True):
        super().__init__()
        
        self.backbone = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            in_chans=NUM_CHANNELS,
            img_size=(TARGET_HEIGHT, TARGET_WIDTH),
            num_classes=0
        )
        
        backbone_dim = self.backbone.embed_dim
        self.proj = nn.Linear(backbone_dim, hidden_dim)
        self.hidden_dim = hidden_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(x)  # (B, num_patches + 1, embed_dim)
        features = self.proj(features)  # (B, T, hidden_dim)
        return features


# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

class BahdanauAttention(nn.Module):
    """Bahdanau (additive) attention mechanism."""
    
    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        decoder_hidden: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        att1 = self.encoder_att(encoder_out)  # (B, T, att_dim)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)  # (B, 1, att_dim)
        
        att = torch.tanh(att1 + att2)  # (B, T, att_dim)
        scores = self.full_att(att).squeeze(2)  # (B, T)
        
        attention_weights = F.softmax(scores, dim=1)  # (B, T)
        context = (encoder_out * attention_weights.unsqueeze(2)).sum(dim=1)  # (B, encoder_dim)
        
        return context, attention_weights


# ============================================================================
# DECODERS
# ============================================================================

class AttentionDecoder(nn.Module):
    """LSTM decoder with attention (for Model B)."""
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 256,
        encoder_dim: int = 256,
        attention_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)
        
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)
    
    def init_hidden(self, encoder_out: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean_encoder = encoder_out.mean(dim=1)
        h = torch.tanh(self.init_h(mean_encoder))
        c = torch.tanh(self.init_c(mean_encoder))
        return h, c
    
    def forward_step(
        self,
        token: torch.Tensor,
        h: torch.Tensor,
        c: torch.Tensor,
        encoder_out: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.embedding(token)
        embed = self.dropout(embed)
        
        context, attention_weights = self.attention(encoder_out, h)
        
        lstm_input = torch.cat([embed, context], dim=1)
        h, c = self.lstm_cell(lstm_input, (h, c))
        
        logits = self.fc(self.dropout(h))
        
        return logits, h, c, attention_weights
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        max_len: int = MAX_SEQ_LENGTH,
        teacher_forcing_ratio: float = 1.0,
        bos_id: int = 1
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = encoder_out.size(0)
        device = encoder_out.device
        
        if targets is not None:
            max_len = targets.size(1)
        
        h, c = self.init_hidden(encoder_out)
        
        outputs = []
        attention_weights_all = []
        
        current_token = torch.full((B,), bos_id, dtype=torch.long, device=device)
        
        for t in range(max_len):
            logits, h, c, att_weights = self.forward_step(current_token, h, c, encoder_out)
            outputs.append(logits)
            attention_weights_all.append(att_weights)
            
            if targets is not None and torch.rand(1).item() < teacher_forcing_ratio:
                current_token = targets[:, t]
            else:
                current_token = logits.argmax(dim=1)
        
        outputs = torch.stack(outputs, dim=1)
        attention_weights_all = torch.stack(attention_weights_all, dim=1)
        
        return outputs, attention_weights_all


class TransformerDecoder(nn.Module):
    """Transformer decoder (for Models C and D)."""
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        max_len: int = MAX_SEQ_LENGTH
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len, dropout)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def forward(
        self,
        encoder_out: torch.Tensor,
        targets: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        B, T = targets.shape
        device = targets.device
        
        if target_mask is None:
            target_mask = self.generate_square_subsequent_mask(T, device)
        
        tgt_embed = self.embedding(targets)
        tgt_embed = self.pos_encoding(tgt_embed)
        
        output = self.transformer_decoder(
            tgt_embed,
            encoder_out,
            tgt_mask=target_mask
        )
        
        logits = self.fc(output)
        
        return logits
