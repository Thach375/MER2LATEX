"""
Evaluation Module for MER2LATEX
================================
Metrics computation, error analysis, and visualization.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from tqdm import tqdm

try:
    from sacrebleu.metrics import BLEU
    SACREBLEU_AVAILABLE = True
except ImportError:
    SACREBLEU_AVAILABLE = False

try:
    import sympy
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

from src.utils.constants import (
    LOGS_ROOT,
    MAX_SEQ_LENGTH
)
from src.tokenizer.tokenize import LaTeXTokenizer, normalize_latex


# ============================================================================
# METRICS
# ============================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein (edit) distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def token_edit_distance(pred_tokens: List[str], target_tokens: List[str]) -> int:
    """Compute edit distance at token level."""
    return levenshtein_distance(pred_tokens, target_tokens)


def compute_exact_match(predictions: List[str], targets: List[str]) -> float:
    """Compute exact match accuracy."""
    correct = sum(
        normalize_latex(pred) == normalize_latex(tgt)
        for pred, tgt in zip(predictions, targets)
    )
    return correct / len(predictions) if predictions else 0.0


def compute_bleu(predictions: List[str], targets: List[str]) -> float:
    """Compute BLEU score."""
    if not SACREBLEU_AVAILABLE:
        # Fallback to simple BLEU approximation
        return _simple_bleu(predictions, targets)
    
    bleu = BLEU()
    
    # Tokenize for BLEU
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in targets]
    
    try:
        score = bleu.corpus_score(
            [' '.join(p) for p in pred_tokens],
            [[' '.join(r[0])] for r in ref_tokens]
        )
        return score.score / 100.0  # Normalize to [0, 1]
    except Exception:
        return _simple_bleu(predictions, targets)


def _simple_bleu(predictions: List[str], targets: List[str], n: int = 4) -> float:
    """Simple BLEU implementation as fallback."""
    def ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    scores = []
    for pred, ref in zip(predictions, targets):
        pred_tokens = pred.split()
        ref_tokens = ref.split()
        
        if len(pred_tokens) == 0:
            scores.append(0.0)
            continue
        
        precisions = []
        for i in range(1, n + 1):
            pred_ngrams = Counter(ngrams(pred_tokens, i))
            ref_ngrams = Counter(ngrams(ref_tokens, i))
            
            matches = sum((pred_ngrams & ref_ngrams).values())
            total = sum(pred_ngrams.values())
            
            if total > 0:
                precisions.append(matches / total)
            else:
                precisions.append(0.0)
        
        if all(p > 0 for p in precisions):
            geo_mean = np.exp(np.mean(np.log(precisions)))
            
            # Brevity penalty
            bp = min(1.0, np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1)))
            scores.append(bp * geo_mean)
        else:
            scores.append(0.0)
    
    return np.mean(scores) if scores else 0.0


def compute_edit_distance(predictions: List[str], targets: List[str]) -> float:
    """Compute average normalized edit distance."""
    distances = []
    for pred, tgt in zip(predictions, targets):
        pred_norm = normalize_latex(pred)
        tgt_norm = normalize_latex(tgt)
        
        dist = levenshtein_distance(pred_norm, tgt_norm)
        max_len = max(len(pred_norm), len(tgt_norm), 1)
        normalized_dist = dist / max_len
        distances.append(normalized_dist)
    
    return np.mean(distances) if distances else 1.0


def compute_sympy_equivalence(predictions: List[str], targets: List[str]) -> float:
    """
    Compute mathematical equivalence using SymPy.
    Two expressions are equivalent if they simplify to the same form.
    """
    if not SYMPY_AVAILABLE:
        return -1.0  # Not available
    
    equivalent = 0
    valid_count = 0
    
    for pred, tgt in zip(predictions, targets):
        try:
            pred_expr = parse_latex(pred)
            tgt_expr = parse_latex(tgt)
            
            # Try to simplify and compare
            diff = sympy.simplify(pred_expr - tgt_expr)
            
            if diff == 0:
                equivalent += 1
            
            valid_count += 1
        except Exception:
            # Skip expressions that can't be parsed
            continue
    
    return equivalent / valid_count if valid_count > 0 else 0.0


def compute_metrics(
    predictions: List[str],
    targets: List[str],
    compute_sympy: bool = False
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        predictions: List of predicted LaTeX strings
        targets: List of target LaTeX strings
        compute_sympy: Whether to compute SymPy equivalence (slow)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'exact_match': compute_exact_match(predictions, targets),
        'bleu': compute_bleu(predictions, targets),
        'edit_distance': compute_edit_distance(predictions, targets),
        'num_samples': len(predictions)
    }
    
    if compute_sympy:
        metrics['sympy_equivalence'] = compute_sympy_equivalence(predictions, targets)
    
    return metrics


# ============================================================================
# DECODING
# ============================================================================

def decode_ctc_predictions(
    log_probs: torch.Tensor,
    tokenizer: LaTeXTokenizer
) -> List[str]:
    """
    Decode CTC output using greedy decoding with blank removal.
    
    Args:
        log_probs: (T, B, vocab_size) log probabilities
        tokenizer: Tokenizer for decoding
    
    Returns:
        List of decoded strings
    """
    # Greedy decode
    predictions = log_probs.argmax(dim=2).permute(1, 0)  # (B, T)
    
    decoded = []
    for pred in predictions:
        # Remove consecutive duplicates and blanks (pad tokens)
        tokens = []
        prev_token = None
        for token_id in pred.tolist():
            if token_id != tokenizer.pad_id and token_id != prev_token:
                tokens.append(token_id)
            prev_token = token_id
        
        # Decode to string
        decoded_str = tokenizer.decode(tokens, skip_special=True)
        decoded.append(decoded_str)
    
    return decoded


def decode_predictions(
    model: nn.Module,
    images: torch.Tensor,
    tokenizer: LaTeXTokenizer,
    model_type: str = 'model_b',
    max_len: int = MAX_SEQ_LENGTH
) -> List[str]:
    """
    Decode model predictions to strings.
    
    Args:
        model: Trained model
        images: Input images (B, C, H, W)
        tokenizer: Tokenizer
        model_type: Type of model
        max_len: Maximum decoding length
    
    Returns:
        List of decoded strings
    """
    model.eval()
    
    with torch.no_grad():
        if model_type == 'model_a':
            log_probs = model(images)
            return decode_ctc_predictions(log_probs, tokenizer)
        else:
            # Autoregressive decoding
            predictions = model.decode_greedy(
                images,
                max_len=max_len,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id
            )
            
            decoded = []
            for pred in predictions:
                decoded_str = tokenizer.decode(pred.tolist(), skip_special=True)
                decoded.append(decoded_str)
            
            return decoded


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

class ErrorAnalyzer:
    """Analyze prediction errors for insights."""
    
    def __init__(
        self,
        predictions: List[str],
        targets: List[str],
        images: Optional[List[Any]] = None
    ):
        self.predictions = predictions
        self.targets = targets
        self.images = images
        self.errors = self._find_errors()
    
    def _find_errors(self) -> List[Dict]:
        """Find all prediction errors."""
        errors = []
        
        for idx, (pred, tgt) in enumerate(zip(self.predictions, self.targets)):
            pred_norm = normalize_latex(pred)
            tgt_norm = normalize_latex(tgt)
            
            if pred_norm != tgt_norm:
                error = {
                    'index': idx,
                    'prediction': pred,
                    'target': tgt,
                    'edit_distance': levenshtein_distance(pred_norm, tgt_norm),
                    'pred_length': len(pred_norm),
                    'target_length': len(tgt_norm),
                    'error_type': self._classify_error(pred_norm, tgt_norm)
                }
                errors.append(error)
        
        return errors
    
    def _classify_error(self, pred: str, target: str) -> str:
        """Classify the type of error."""
        # Check for common error patterns
        if len(pred) == 0:
            return 'empty_prediction'
        
        if '\\frac' in target and '\\frac' not in pred:
            return 'missing_fraction'
        
        if '\\sqrt' in target and '\\sqrt' not in pred:
            return 'missing_sqrt'
        
        if '{' in target:
            pred_braces = pred.count('{') - pred.count('}')
            tgt_braces = target.count('{') - target.count('}')
            if pred_braces != tgt_braces:
                return 'brace_mismatch'
        
        if '^' in target or '_' in target:
            if ('^' in target and '^' not in pred) or ('_' in target and '_' not in pred):
                return 'missing_sub_superscript'
        
        # Check for character-level errors
        pred_set = set(pred)
        tgt_set = set(target)
        
        if len(tgt_set - pred_set) > 0:
            return 'missing_characters'
        
        if len(pred_set - tgt_set) > 0:
            return 'extra_characters'
        
        return 'other'
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        if not self.errors:
            return {'total_errors': 0, 'error_rate': 0.0}
        
        error_types = Counter(e['error_type'] for e in self.errors)
        edit_distances = [e['edit_distance'] for e in self.errors]
        
        return {
            'total_errors': len(self.errors),
            'error_rate': len(self.errors) / len(self.predictions),
            'error_types': dict(error_types),
            'avg_edit_distance': np.mean(edit_distances),
            'max_edit_distance': max(edit_distances),
            'median_edit_distance': np.median(edit_distances)
        }
    
    def get_worst_errors(self, n: int = 10) -> List[Dict]:
        """Get n worst errors by edit distance."""
        sorted_errors = sorted(self.errors, key=lambda x: -x['edit_distance'])
        return sorted_errors[:n]
    
    def plot_error_distribution(self, save_path: Optional[str] = None):
        """Plot error distribution."""
        if not self.errors:
            print("No errors to plot!")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Error types
        error_types = Counter(e['error_type'] for e in self.errors)
        types = list(error_types.keys())
        counts = list(error_types.values())
        
        axes[0].barh(types, counts)
        axes[0].set_xlabel('Count')
        axes[0].set_title('Error Types Distribution')
        
        # Edit distance distribution
        edit_dists = [e['edit_distance'] for e in self.errors]
        axes[1].hist(edit_dists, bins=30, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Edit Distance')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title('Edit Distance Distribution')
        
        # Length vs error
        target_lens = [e['target_length'] for e in self.errors]
        axes[2].scatter(target_lens, edit_dists, alpha=0.5)
        axes[2].set_xlabel('Target Length')
        axes[2].set_ylabel('Edit Distance')
        axes[2].set_title('Target Length vs Edit Distance')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"[INFO] Saved error analysis plot to {save_path}")
        
        plt.show()


# ============================================================================
# TRAINING VISUALIZATION
# ============================================================================

def plot_training_history(
    history: List[Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Plot training history curves.
    
    Args:
        history: List of epoch metrics
        save_path: Path to save the plot
    """
    epochs = [h['epoch'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    train_loss = [h.get('train_loss', 0) for h in history]
    val_loss = [h.get('val_loss', 0) for h in history]
    
    axes[0, 0].plot(epochs, train_loss, 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, val_loss, 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Exact Match
    exact_match = [h.get('exact_match', 0) for h in history]
    axes[0, 1].plot(epochs, exact_match, 'g-', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Exact Match')
    axes[0, 1].set_title('Exact Match Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    
    # BLEU Score
    bleu = [h.get('bleu', 0) for h in history]
    axes[1, 0].plot(epochs, bleu, 'm-', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('BLEU Score')
    axes[1, 0].set_title('BLEU Score')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Edit Distance
    edit_dist = [h.get('edit_distance', 1) for h in history]
    axes[1, 1].plot(epochs, edit_dist, 'c-', marker='o')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Edit Distance (normalized)')
    axes[1, 1].set_title('Edit Distance')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved training history plot to {save_path}")
    
    plt.show()


def compare_models(
    results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None
):
    """
    Compare metrics across different models.
    
    Args:
        results: Dict mapping model name to metrics
        save_path: Path to save the plot
    """
    models = list(results.keys())
    metrics = ['exact_match', 'bleu', 'edit_distance']
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for idx, metric in enumerate(metrics):
        values = [results[m].get(metric, 0) for m in models]
        
        bars = axes[idx].bar(models, values, color=colors)
        axes[idx].set_ylabel(metric.replace('_', ' ').title())
        axes[idx].set_title(f'{metric.replace("_", " ").title()} Comparison')
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{val:.3f}',
                ha='center',
                va='bottom'
            )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[INFO] Saved model comparison plot to {save_path}")
    
    plt.show()


# ============================================================================
# FULL EVALUATION
# ============================================================================

def evaluate_model(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer: LaTeXTokenizer,
    model_type: str = 'model_b',
    device: torch.device = None,
    compute_sympy: bool = False,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Full evaluation of a model.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        tokenizer: Tokenizer
        model_type: Type of model
        device: Device to use
        compute_sympy: Whether to compute SymPy equivalence
        save_dir: Directory to save results
    
    Returns:
        Evaluation results including metrics and error analysis
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    print("[INFO] Running evaluation...")
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch['images'].to(device)
        formulas = batch['formulas']
        
        predictions = decode_predictions(
            model,
            images,
            tokenizer,
            model_type=model_type
        )
        
        all_predictions.extend(predictions)
        all_targets.extend(formulas)
    
    # Compute metrics
    print("[INFO] Computing metrics...")
    metrics = compute_metrics(all_predictions, all_targets, compute_sympy=compute_sympy)
    
    # Error analysis
    print("[INFO] Analyzing errors...")
    error_analyzer = ErrorAnalyzer(all_predictions, all_targets)
    error_summary = error_analyzer.get_error_summary()
    worst_errors = error_analyzer.get_worst_errors(n=20)
    
    results = {
        'metrics': metrics,
        'error_summary': error_summary,
        'worst_errors': worst_errors,
        'num_samples': len(all_predictions)
    }
    
    # Save results
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(save_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save error summary
        with open(save_dir / 'error_summary.json', 'w') as f:
            json.dump(error_summary, f, indent=2)
        
        # Save predictions
        pred_df = pd.DataFrame({
            'prediction': all_predictions,
            'target': all_targets
        })
        pred_df.to_csv(save_dir / 'predictions.csv', index=False)
        
        # Plot error distribution
        error_analyzer.plot_error_distribution(save_path=str(save_dir / 'error_distribution.png'))
        
        print(f"[INFO] Results saved to {save_dir}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Samples: {metrics['num_samples']}")
    print(f"Exact Match: {metrics['exact_match']:.4f}")
    print(f"BLEU Score: {metrics['bleu']:.4f}")
    print(f"Edit Distance: {metrics['edit_distance']:.4f}")
    if 'sympy_equivalence' in metrics:
        print(f"SymPy Equivalence: {metrics['sympy_equivalence']:.4f}")
    print(f"\nError Rate: {error_summary['error_rate']:.4f}")
    print(f"Error Types: {error_summary.get('error_types', {})}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Test metrics
    print("[TEST] Testing evaluation metrics...")
    
    predictions = [
        "\\frac{1}{2}",
        "x^2 + y^2",
        "\\sum_{i=1}^{n} x_i",
        "a + b"
    ]
    
    targets = [
        "\\frac{1}{2}",
        "x^2 + y^2",
        "\\sum_{i=1}^{n} x_i",
        "a + c"  # Intentional error
    ]
    
    metrics = compute_metrics(predictions, targets)
    
    print(f"\nTest Metrics:")
    print(f"  Exact Match: {metrics['exact_match']:.4f}")
    print(f"  BLEU: {metrics['bleu']:.4f}")
    print(f"  Edit Distance: {metrics['edit_distance']:.4f}")
    
    # Test error analysis
    print("\n[TEST] Testing error analysis...")
    analyzer = ErrorAnalyzer(predictions, targets)
    summary = analyzer.get_error_summary()
    print(f"  Error Summary: {summary}")
    
    print("\n[DONE] Evaluation module test complete!")
