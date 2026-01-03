#!/usr/bin/env python3
"""
Test trained model on test set
================================
Quick script to evaluate a trained checkpoint on the test set.

Usage:
    python test_model.py --checkpoint checkpoints/model_b/20260103_180343/best_checkpoint.pt --model model_b
    python test_model.py --checkpoint-dir checkpoints/model_b/20260103_180343 --model model_b
"""

import argparse
import torch
from pathlib import Path

from src.models import create_model
from src.data.dataset import get_dataloader
from src.evaluation.metrics import evaluate_model
from src.tokenizer.tokenize import LaTeXTokenizer
from src.utils.constants import BATCH_SIZE, VOCAB_PATH


def find_best_checkpoint(checkpoint_dir: str) -> str:
    """Find the best checkpoint in a directory."""
    checkpoint_dir = Path(checkpoint_dir)
    
    # Look for best checkpoint
    best_ckpt = checkpoint_dir / 'best_checkpoint.pt'
    if best_ckpt.exists():
        return str(best_ckpt)
    
    # Look for latest checkpoint
    checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pt'))
    if checkpoints:
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        return str(latest)
    
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_name: str,
    device: torch.device
):
    """Load model from checkpoint."""
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    vocab_size = checkpoint.get('vocab_size', 500)
    
    # Create model
    model = create_model(
        model_name,
        vocab_size=vocab_size,
        hidden_dim=256,
        dropout=0.1,
        pretrained=False  # Don't need pretrained weights, loading from checkpoint
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"[INFO] Model loaded successfully")
    print(f"[INFO] Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"[INFO] Best val loss: {checkpoint.get('best_val_loss', 'unknown'):.4f}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Test MER2LATEX model on test set')
    parser.add_argument('--checkpoint', type=str, 
                       help='Path to specific checkpoint file')
    parser.add_argument('--checkpoint-dir', type=str,
                       help='Directory containing checkpoints (will use best)')
    parser.add_argument('--model', type=str, required=True,
                       choices=['model_a', 'model_b', 'model_c', 'model_d'],
                       help='Model architecture')
    parser.add_argument('--dataset', type=str, default='im2latex',
                       choices=['im2latex', 'crohme'],
                       help='Dataset to test on')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE,
                       help='Batch size for testing')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save evaluation results')
    parser.add_argument('--compute-sympy', action='store_true',
                       help='Compute SymPy mathematical equivalence (slower)')
    
    args = parser.parse_args()
    
    # Determine checkpoint path
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    elif args.checkpoint_dir:
        checkpoint_path = find_best_checkpoint(args.checkpoint_dir)
    else:
        parser.error("Either --checkpoint or --checkpoint-dir must be provided")
    
    print(f"\n{'='*60}")
    print(f"Testing Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")
    
    # Load tokenizer
    print(f"[INFO] Loading tokenizer from {VOCAB_PATH}")
    tokenizer = LaTeXTokenizer(vocab_path=VOCAB_PATH)
    
    # Load model
    model = load_model_from_checkpoint(checkpoint_path, args.model, device)
    
    # Load test data
    print(f"[INFO] Loading test dataset: {args.dataset}")
    test_loader, _ = get_dataloader(
        dataset_type=args.dataset,
        split='test',
        batch_size=args.batch_size,
        tokenizer=tokenizer,
        num_workers=4,
        use_preprocessed=True
    )
    print(f"[INFO] Test samples: {len(test_loader.dataset)}")
    
    # Evaluate
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        tokenizer=tokenizer,
        model_type=args.model,
        device=device,
        compute_sympy=args.compute_sympy,
        save_dir=args.save_dir
    )
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()
