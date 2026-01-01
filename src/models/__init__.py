"""
Models module for MER2LATEX
============================
Contains model architectures and components.
"""

from src.models.components import (
    PositionalEncoding,
    PositionalEncoding2D,
    CNNEncoder,
    ResNetEncoder,
    ViTEncoder,
    BahdanauAttention,
    AttentionDecoder,
    TransformerDecoder
)

from src.models.architectures import (
    ModelA_CTC,
    ModelB_Attention,
    ModelC_Transformer,
    ModelD_TrOCR,
    create_model
)

__all__ = [
    # Components
    'PositionalEncoding',
    'PositionalEncoding2D',
    'CNNEncoder',
    'ResNetEncoder',
    'ViTEncoder',
    'BahdanauAttention',
    'AttentionDecoder',
    'TransformerDecoder',
    # Models
    'ModelA_CTC',
    'ModelB_Attention',
    'ModelC_Transformer',
    'ModelD_TrOCR',
    'create_model'
]
