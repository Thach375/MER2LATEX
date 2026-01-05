#!/usr/bin/env python3
"""
Evaluate model and save results to a single JSON file.

Usage:
    python -m src.evaluation.evaluate --checkpoint <path> --model model_a --tag baseline
    python -m src.evaluation.evaluate --checkpoint <path> --model model_a --dataset all --tag finetuned
    python -m src.evaluation.evaluate --show-results
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime

from src.models import create_model
from src.data.dataset import get_dataloader
from src.evaluation.metrics import evaluate_model
from src.tokenizer.tokenize import LaTeXTokenizer
from src.utils.constants import BATCH_SIZE, VOCAB_PATH

RESULTS_FILE = Path('results/all_results.json')


def load_results():
    """Load all results from file."""
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []


def save_result(model, dataset, tag, epoch, metrics):
    """Append a result to the results file."""
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    results = load_results()
    results.append({
        'model': model,
        'dataset': dataset,
        'tag': tag or 'default',
        'epoch': epoch,
        'bleu': metrics.get('bleu', 0),
        'exact_match': metrics.get('exact_match', 0),
        'edit_distance': metrics.get('edit_distance', 0),
        'time': datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2)


def show_results():
    """Display all results as a comparison table."""
    results = load_results()
    
    if not results:
        print("No results yet. Run evaluation first.")
        return
    
    # Sort by model, tag, dataset
    results = sorted(results, key=lambda x: (x['model'], x['tag'], x['dataset']))
    
    print("\n" + "="*95)
    print(f"{'Model':<10} {'Dataset':<10} {'Tag':<12} {'Epoch':<6} {'BLEU':<8} {'Exact':<8} {'EditDist':<8} {'Time':<16}")
    print("="*95)
    
    for r in results:
        print(f"{r['model']:<10} {r['dataset']:<10} {r['tag']:<12} {r['epoch']:<6} "
              f"{r['bleu']:<8.4f} {r['exact_match']:<8.4f} {r['edit_distance']:<8.4f} {r['time']:<16}")
    
    print("="*95)


def infer_model_config_from_checkpoint(checkpoint: dict, model_name: str) -> dict:
    """
    Infer model configuration from checkpoint state dict.
    This allows loading old checkpoints with different architectures.
    """
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    config = {
        'pretrained': False,  # Don't need pretrained when loading checkpoint
    }
    
    if model_name == 'model_b':
        # Detect hidden_dim from encoder projection
        if 'encoder.proj.weight' in state_dict:
            config['hidden_dim'] = state_dict['encoder.proj.weight'].shape[0]
        else:
            config['hidden_dim'] = 256
        
        # Detect attention_dim from attention layers
        if 'decoder.attention.encoder_att.weight' in state_dict:
            config['attention_dim'] = state_dict['decoder.attention.encoder_att.weight'].shape[0]
        else:
            config['attention_dim'] = 256
        
        # Detect if coverage is used
        config['use_coverage'] = 'decoder.attention.coverage_att.weight' in state_dict
        
        # Detect embed_dim from embedding
        if 'decoder.embedding.weight' in state_dict:
            config['embed_dim'] = state_dict['decoder.embedding.weight'].shape[1]
        else:
            config['embed_dim'] = 256
        
        # Detect dropout (default, can't infer from weights)
        config['dropout'] = 0.1
        
    elif model_name == 'model_c':
        # Detect hidden_dim
        if 'decoder.embedding.weight' in state_dict:
            config['hidden_dim'] = state_dict['decoder.embedding.weight'].shape[1]
        else:
            config['hidden_dim'] = 256
        
        # Count decoder layers
        layer_count = 0
        for key in state_dict.keys():
            if 'decoder.transformer_decoder.layers.' in key:
                layer_idx = int(key.split('.')[3])
                layer_count = max(layer_count, layer_idx + 1)
        config['num_decoder_layers'] = layer_count if layer_count > 0 else 4
        
        # Detect ff_dim from feedforward layers
        for key in state_dict.keys():
            if 'linear1.weight' in key and 'decoder' in key:
                config['ff_dim'] = state_dict[key].shape[0]
                break
        else:
            config['ff_dim'] = 1024
        
        config['num_heads'] = 8
        config['dropout'] = 0.1
    
    elif model_name == 'model_d':
        if 'decoder.embedding.weight' in state_dict:
            config['hidden_dim'] = state_dict['decoder.embedding.weight'].shape[1]
        else:
            config['hidden_dim'] = 256
        config['dropout'] = 0.1
    
    else:  # model_a
        config['hidden_dim'] = 256
        config['dropout'] = 0.1
    
    return config


def main():
    parser = argparse.ArgumentParser(description='Evaluate MER2LATEX model')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint')
    parser.add_argument('--model', type=str, choices=['model_a', 'model_b', 'model_c', 'model_d'])
    parser.add_argument('--dataset', type=str, default='im2latex', choices=['im2latex', 'crohme', 'all'])
    parser.add_argument('--tag', type=str, default=None, help='Experiment tag (e.g., baseline, finetuned)')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--show-results', action='store_true', help='Show comparison table')
    parser.add_argument('--clear-results', action='store_true', help='Clear all results')
    args = parser.parse_args()
    
    if args.show_results:
        show_results()
        return
    
    if args.clear_results:
        if RESULTS_FILE.exists():
            RESULTS_FILE.unlink()
            print("Results cleared.")
        return
    
    if not args.checkpoint or not args.model:
        parser.error("--checkpoint and --model are required")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    epoch = checkpoint.get('epoch', '?')
    vocab_size = checkpoint.get('vocab_size', 500)
    
    # Infer model config from checkpoint to handle different architectures
    model_config = infer_model_config_from_checkpoint(checkpoint, args.model)
    print(f"Inferred model config: {model_config}")
    
    # Create model with inferred config
    model = create_model(args.model, vocab_size=vocab_size, **model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Model loaded (epoch {epoch})")
    
    # Load tokenizer
    tokenizer = LaTeXTokenizer(vocab_path=VOCAB_PATH)
    
    # Datasets to evaluate
    datasets = ['im2latex', 'crohme'] if args.dataset == 'all' else [args.dataset]
    
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Evaluating on {dataset.upper()}")
        print(f"{'='*50}")
        
        test_loader, _ = get_dataloader(
            dataset_type=dataset, split='test', batch_size=args.batch_size,
            tokenizer=tokenizer, num_workers=4, use_preprocessed=True
        )
        print(f"Test samples: {len(test_loader.dataset)}")
        
        results = evaluate_model(
            model=model, dataloader=test_loader, tokenizer=tokenizer,
            model_type=args.model, device=device
        )
        
        metrics = results.get('metrics', {})
        print(f"\nResults:")
        print(f"  BLEU:         {metrics.get('bleu', 0):.4f}")
        print(f"  Exact Match:  {metrics.get('exact_match', 0):.4f}")
        print(f"  Edit Dist:    {metrics.get('edit_distance', 0):.4f}")
        
        # Save result
        save_result(args.model, dataset, args.tag, epoch, metrics)
        print(f"Result saved to {RESULTS_FILE}")
    
    print("\nDone! View all results: python -m src.evaluation.evaluate --show-results")


if __name__ == "__main__":
    main()
