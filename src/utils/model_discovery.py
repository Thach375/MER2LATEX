"""
Model Discovery and Loading Utilities
======================================
Functions to discover, infer types, and load models from checkpoints.
"""

from pathlib import Path
from typing import Dict
import torch

from src.models.architectures import (
    ModelA_CTC,
    ModelB_Attention,
    ModelC_Transformer,
    ModelD_TrOCR
)


def get_available_models(checkpoint_dir: Path) -> Dict[str, str]:
    """
    Scan checkpoints directory for available models.
    
    Args:
        checkpoint_dir: Path to checkpoints directory
        
    Returns:
        Dictionary mapping model keys to checkpoint paths
    """
    models = {}
    
    if not checkpoint_dir.exists():
        print(f"[WARNING] Checkpoint directory not found: {checkpoint_dir}")
        return models
    
    # Scan for model checkpoints
    for model_dir in checkpoint_dir.iterdir():
        if model_dir.is_dir():
            model_name = model_dir.name
            
            # Find all timestamp directories
            for timestamp_dir in model_dir.iterdir():
                if timestamp_dir.is_dir():
                    best_pt = timestamp_dir / 'best.pt'
                    
                    if best_pt.exists():
                        key = f"{model_name}/{timestamp_dir.name}/best"
                        models[key] = str(best_pt)
    
    return models


def infer_model_type(checkpoint_path: str, state_dict: dict) -> str:
    """
    Infer model type from checkpoint path and state dict keys.
    
    Args:
        checkpoint_path: Path to checkpoint file
        state_dict: Model state dictionary
        
    Returns:
        Model type string ('model_a', 'model_b', 'model_c', or 'model_d')
    """
    # First check path
    path_lower = checkpoint_path.lower()
    if 'model_a' in path_lower:
        return 'model_a'
    elif 'model_b' in path_lower:
        return 'model_b'
    elif 'model_c' in path_lower:
        return 'model_c'
    elif 'model_d' in path_lower:
        return 'model_d'
    
    # Infer from weight keys
    keys = list(state_dict.keys())
    keys_str = ' '.join(keys[:20])
    
    if 'encoder.cnn' in keys_str and 'lstm' in keys_str:
        return 'model_a'
    elif 'encoder.backbone' in keys_str and 'decoder.attention' in keys_str:
        return 'model_b'
    elif 'encoder.backbone.cls_token' in keys_str or 'encoder.backbone.pos_embed' in keys_str:
        return 'model_c'
    elif 'encoder.cls_token' in keys_str or 'encoder.pos_embed' in keys_str:
        return 'model_d'
    
    return 'model_b'  # Default fallback


def load_model(checkpoint_path: str, vocab_size: int, device: torch.device):
    """
    Load model from checkpoint. Supports Model A, B, C, D.
    
    Args:
        checkpoint_path: Path to checkpoint file
        vocab_size: Vocabulary size
        device: Device to load model on
        
    Returns:
        Loaded model in eval mode
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Infer model type
        model_type = checkpoint.get('model_type', infer_model_type(checkpoint_path, state_dict))
        print(f"[INFO] Detected model type: {model_type}")
        
        # Get hyperparameters from checkpoint or infer from weights
        hyperparams = checkpoint.get('hyperparameters', {})
        
        # Infer dimensions from weights if not in hyperparams
        if 'hidden_dim' not in hyperparams:
            if 'encoder.proj.weight' in state_dict:  # Model B
                hidden_dim = state_dict['encoder.proj.weight'].shape[0]
            elif 'decoder.fc.weight' in state_dict:  # Model B, C, D
                hidden_dim = state_dict['decoder.fc.weight'].shape[1]
            elif 'fc.weight' in state_dict:  # Model A
                hidden_dim = state_dict['fc.weight'].shape[1] // 2  # BiLSTM
            else:
                hidden_dim = 256  # Fallback
            print(f"[INFO] Inferred hidden_dim={hidden_dim}")
        else:
            hidden_dim = hyperparams['hidden_dim']
        
        # Initialize model based on type
        if model_type == 'model_a':
            print(f"[INFO] Loading Model A (CNN + BiLSTM + CTC)")
            model = ModelA_CTC(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_layers=hyperparams.get('num_layers', 2),
                dropout=hyperparams.get('dropout', 0.1)
            )
        
        elif model_type == 'model_b':
            # Extract Model B specific params
            embed_dim = hyperparams.get('embed_dim', 256)
            attention_dim = hyperparams.get('attention_dim', hidden_dim)
            dropout = hyperparams.get('dropout', 0.3)
            use_coverage = hyperparams.get('use_coverage', True)
            
            # Check if coverage is actually in the checkpoint
            if 'decoder.attention.coverage_att.weight' not in state_dict:
                use_coverage = False
            
            print(f"[INFO] Loading Model B (ResNet + Attention)")
            print(f"  - hidden_dim: {hidden_dim}")
            print(f"  - embed_dim: {embed_dim}")
            print(f"  - attention_dim: {attention_dim}")
            print(f"  - use_coverage: {use_coverage}")
            
            model = ModelB_Attention(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                embed_dim=embed_dim,
                attention_dim=attention_dim,
                dropout=dropout,
                pretrained=False,
                use_coverage=use_coverage
            )
        
        elif model_type == 'model_c':
            # Infer num_decoder_layers from checkpoint
            if 'num_decoder_layers' not in hyperparams:
                layers = set()
                for key in state_dict.keys():
                    if 'decoder.transformer_decoder.layers.' in key:
                        layer_num = int(key.split('.layers.')[1].split('.')[0])
                        layers.add(layer_num)
                num_decoder_layers = max(layers) + 1 if layers else 3
            else:
                num_decoder_layers = hyperparams['num_decoder_layers']
            
            # Infer ff_dim from checkpoint
            if 'ff_dim' not in hyperparams and 'decoder.transformer_decoder.layers.0.linear1.weight' in state_dict:
                ff_dim = state_dict['decoder.transformer_decoder.layers.0.linear1.weight'].shape[0]
            else:
                ff_dim = hyperparams.get('ff_dim', 512)
            
            print(f"[INFO] Loading Model C (ViT + Transformer)")
            print(f"  - hidden_dim: {hidden_dim}")
            print(f"  - num_decoder_layers: {num_decoder_layers}")
            print(f"  - ff_dim: {ff_dim}")
            
            model = ModelC_Transformer(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_decoder_layers=num_decoder_layers,
                num_heads=hyperparams.get('num_heads', 8),
                ff_dim=ff_dim,
                dropout=hyperparams.get('dropout', 0.2),
                pretrained=False
            )
        
        elif model_type == 'model_d':
            print(f"[INFO] Loading Model D (TrOCR)")
            model = ModelD_TrOCR(
                vocab_size=vocab_size,
                hidden_dim=hidden_dim,
                num_decoder_layers=hyperparams.get('num_decoder_layers', 4),
                num_heads=hyperparams.get('num_heads', 8),
                ff_dim=hyperparams.get('ff_dim', 1024),
                dropout=hyperparams.get('dropout', 0.1),
                encoder_name=hyperparams.get('encoder_name', 'deit_small_patch16_224')
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Load state dict
        model.load_state_dict(state_dict)
        
        model.to(device)
        model.eval()
        
        print(f"[OK] Loaded {model_type} from {checkpoint_path}")
        return model
    
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        raise
