"""
Model Architectures for MER2LATEX
===================================
4 model architectures:
- Model A: CNN + BiLSTM + CTC (Baseline)
- Model B: ResNet Encoder + Attention Decoder
- Model C: ViT Encoder + Transformer Decoder
- Model D: TrOCR-style (OCR-free)
"""

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

from src.models.components import (
    CNNEncoder,
    ResNetEncoder,
    ViTEncoder,
    AttentionDecoder,
    TransformerDecoder
)


# ============================================================================
# MODEL A: CNN + BiLSTM + CTC (Baseline)
# ============================================================================

class ModelA_CTC(nn.Module):
    """
    Model A: CNN + BiLSTM + CTC
    Baseline OCR model with CTC loss.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        **kwargs  # Accept but ignore extra kwargs
    ):
        super().__init__()
        
        self.encoder = CNNEncoder(input_channels=NUM_CHANNELS, hidden_dim=hidden_dim)
        
        # CNN output: (B, W', 512*4) = (B, W', 2048)
        cnn_output_dim = 512 * 4
        
        self.lstm = nn.LSTM(
            input_size=cnn_output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)
        self.vocab_size = vocab_size
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CTC.
        
        Args:
            images: (B, 1, H, W)
        
        Returns:
            log_probs: (T, B, vocab_size) for CTC loss
        """
        features = self.encoder(images)  # (B, T, D)
        lstm_out, _ = self.lstm(features)  # (B, T, 2*hidden)
        logits = self.fc(lstm_out)  # (B, T, vocab_size)
        
        logits = logits.permute(1, 0, 2)  # (T, B, vocab_size)
        log_probs = F.log_softmax(logits, dim=2)
        
        return log_probs
    
    def decode_greedy(self, images: torch.Tensor) -> torch.Tensor:
        """Greedy decoding for inference."""
        log_probs = self.forward(images)  # (T, B, vocab)
        predictions = log_probs.argmax(dim=2)  # (T, B)
        return predictions.permute(1, 0)  # (B, T)


# ============================================================================
# MODEL B: ResNet Encoder + Attention Decoder
# ============================================================================

class ModelB_Attention(nn.Module):
    """
    Model B: ResNet Encoder + Attention Decoder
    Seq2Seq model with Bahdanau attention.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        embed_dim: int = 256,
        attention_dim: int = 256,
        dropout: float = 0.1,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.encoder = ResNetEncoder(hidden_dim=hidden_dim, pretrained=pretrained)
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=hidden_dim,
            attention_dim=attention_dim,
            dropout=dropout
        )
        
        self.vocab_size = vocab_size
    
    def forward(
        self,
        images: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoder_out = self.encoder(images)
        outputs, attention_weights = self.decoder(
            encoder_out,
            targets,
            teacher_forcing_ratio=teacher_forcing_ratio
        )
        return outputs, attention_weights
    
    def decode_greedy(
        self,
        images: torch.Tensor,
        max_len: int = MAX_SEQ_LENGTH,
        eos_id: int = 2
    ) -> torch.Tensor:
        """Greedy decoding for inference."""
        self.eval()
        with torch.no_grad():
            encoder_out = self.encoder(images)
            
            B = images.size(0)
            device = images.device
            
            h, c = self.decoder.init_hidden(encoder_out)
            current_token = torch.ones(B, dtype=torch.long, device=device)
            
            predictions = []
            for _ in range(max_len):
                logits, h, c, _ = self.decoder.forward_step(current_token, h, c, encoder_out)
                current_token = logits.argmax(dim=1)
                predictions.append(current_token)
                
                if (current_token == eos_id).all():
                    break
            
            predictions = torch.stack(predictions, dim=1)
        
        return predictions


# ============================================================================
# MODEL C: ViT Encoder + Transformer Decoder
# ============================================================================

class ModelC_Transformer(nn.Module):
    """
    Model C: ViT Encoder + Transformer Decoder
    Full transformer-based model.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.encoder = ViTEncoder(hidden_dim=hidden_dim, pretrained=pretrained)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        self.vocab_size = vocab_size
    
    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        encoder_out = self.encoder(images)
        logits = self.decoder(encoder_out, targets)
        return logits
    
    @torch.no_grad()
    def decode_greedy(
        self,
        images: torch.Tensor,
        max_len: int = MAX_SEQ_LENGTH,
        bos_id: int = 1,
        eos_id: int = 2
    ) -> torch.Tensor:
        """Greedy decoding for inference."""
        self.eval()
        
        B = images.size(0)
        device = images.device
        
        encoder_out = self.encoder(images)
        
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            logits = self.decoder(encoder_out, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token.squeeze(-1) == eos_id).all():
                break
        
        return generated


# ============================================================================
# MODEL D: TrOCR-style (OCR-free)
# ============================================================================

class ModelD_TrOCR(nn.Module):
    """
    Model D: TrOCR-style model
    Uses a pretrained vision encoder and transformer decoder.
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_decoder_layers: int = 4,
        num_heads: int = 8,
        ff_dim: int = 1024,
        dropout: float = 0.1,
        encoder_name: str = 'deit_small_patch16_224',
        **kwargs
    ):
        super().__init__()
        
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=True,
            in_chans=NUM_CHANNELS,
            img_size=(TARGET_HEIGHT, TARGET_WIDTH),
            num_classes=0
        )
        
        encoder_dim = self.encoder.embed_dim
        self.encoder_proj = nn.Linear(encoder_dim, hidden_dim)
        
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout=dropout
        )
        
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
    
    def forward(
        self,
        images: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        encoder_out = self.encoder.forward_features(images)
        encoder_out = self.encoder_proj(encoder_out)
        logits = self.decoder(encoder_out, targets)
        return logits
    
    @torch.no_grad()
    def decode_greedy(
        self,
        images: torch.Tensor,
        max_len: int = MAX_SEQ_LENGTH,
        bos_id: int = 1,
        eos_id: int = 2
    ) -> torch.Tensor:
        """Greedy decoding for inference."""
        self.eval()
        
        B = images.size(0)
        device = images.device
        
        encoder_out = self.encoder.forward_features(images)
        encoder_out = self.encoder_proj(encoder_out)
        
        generated = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        
        for _ in range(max_len - 1):
            logits = self.decoder(encoder_out, generated)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if (next_token.squeeze(-1) == eos_id).all():
                break
        
        return generated


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(
    model_name: str,
    vocab_size: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: 'model_a', 'model_b', 'model_c', or 'model_d'
        vocab_size: Vocabulary size
        **kwargs: Model-specific arguments
    
    Returns:
        Model instance
    """
    models = {
        'model_a': ModelA_CTC,
        'model_b': ModelB_Attention,
        'model_c': ModelC_Transformer,
        'model_d': ModelD_TrOCR
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name](vocab_size=vocab_size, **kwargs)
