"""
Analysis Utilities for EDA
===========================
Consolidated analysis functions for MER2LATEX project.

This module combines functionality from:
- image.py: Image statistics analysis
- label.py: Label/sequence analysis
- structural.py: Formula structural analysis
- compare.py: Cross-dataset comparison

Run standalone:
    python -m src.utils.analysis
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


# ============================================================================
# IMAGE STATISTICS ANALYSIS
# ============================================================================

def extract_image_dimensions(df, sample_size=None):
    """
    Extract width and height from images.
    
    Args:
        df: DataFrame with 'image_path' column
        sample_size: If specified, randomly sample images (for faster processing)
    
    Returns:
        DataFrame with width, height, and aspect_ratio columns
    """
    if sample_size and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=42)
    else:
        df_sample = df.copy()
    
    dimensions = []
    
    for img_path in tqdm(df_sample['image_path'], desc="Loading images"):
        try:
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    width, height = img.size
                    dimensions.append({
                        'width': width,
                        'height': height,
                        'aspect_ratio': width / height if height > 0 else 0
                    })
            else:
                dimensions.append({'width': None, 'height': None, 'aspect_ratio': None})
        except Exception as e:
            dimensions.append({'width': None, 'height': None, 'aspect_ratio': None})
    
    dim_df = pd.DataFrame(dimensions)
    return pd.concat([df_sample.reset_index(drop=True), dim_df], axis=1)


def analyze_pixel_intensity(df, sample_size=500, dataset_name=""):
    """
    Analyze pixel intensity statistics.
    
    Args:
        df: DataFrame with 'image_path' column
        sample_size: Number of images to sample
        dataset_name: Dataset name for display
    
    Returns:
        Dictionary with mean, std, pixels, image_means, image_stds
    """
    df_sample = df.sample(n=min(sample_size, len(df)), random_state=42)
    
    all_pixels = []
    image_means = []
    image_stds = []
    
    for img_path in tqdm(df_sample['image_path'], desc=f"Analyzing {dataset_name}"):
        try:
            if os.path.exists(img_path):
                with Image.open(img_path) as img:
                    if img.mode != 'L':
                        img = img.convert('L')
                    
                    img_array = np.array(img).astype(np.float32) / 255.0
                    image_means.append(img_array.mean())
                    image_stds.append(img_array.std())
                    
                    sampled_pixels = np.random.choice(
                        img_array.flatten(), 
                        size=min(1000, img_array.size), 
                        replace=False
                    )
                    all_pixels.extend(sampled_pixels)
        except Exception:
            continue
    
    return {
        'mean': np.mean(image_means),
        'std': np.mean(image_stds),
        'pixels': np.array(all_pixels),
        'image_means': image_means,
        'image_stds': image_stds
    }


def plot_image_statistics(im2latex_clean, crohme_clean):
    """
    Plot image dimension distributions (width vs height scatter plots).
    
    Args:
        im2latex_clean: DataFrame with IM2LATEX image dimensions
        crohme_clean: DataFrame with CROHME image dimensions
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # IM2LATEX
    axes[0].scatter(im2latex_clean['width'], im2latex_clean['height'], alpha=0.3, s=10)
    axes[0].set_xlabel('Width (pixels)', fontsize=12)
    axes[0].set_ylabel('Height (pixels)', fontsize=12)
    axes[0].set_title('IM2LATEX: Width vs Height Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # CROHME
    axes[1].scatter(crohme_clean['width'], crohme_clean['height'], alpha=0.3, s=10, color='orange')
    axes[1].set_xlabel('Width (pixels)', fontsize=12)
    axes[1].set_ylabel('Height (pixels)', fontsize=12)
    axes[1].set_title('CROHME: Width vs Height Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_pixel_intensity(im2latex_clean, crohme_clean):
    """
    Plot aspect ratio distributions.
    
    Args:
        im2latex_clean: DataFrame with IM2LATEX image dimensions
        crohme_clean: DataFrame with CROHME image dimensions
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # IM2LATEX
    axes[0].hist(im2latex_clean['aspect_ratio'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(im2latex_clean['aspect_ratio'].median(), color='red', linestyle='--', 
                    label=f'Median: {im2latex_clean["aspect_ratio"].median():.2f}')
    axes[0].set_xlabel('Aspect Ratio (Width/Height)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('IM2LATEX: Aspect Ratio Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CROHME
    axes[1].hist(crohme_clean['aspect_ratio'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(crohme_clean['aspect_ratio'].median(), color='red', linestyle='--',
                    label=f'Median: {crohme_clean["aspect_ratio"].median():.2f}')
    axes[1].set_xlabel('Aspect Ratio (Width/Height)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('CROHME: Aspect Ratio Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# LABEL/SEQUENCE ANALYSIS
# ============================================================================

def tokenize_formula(formula):
    """
    Simple tokenization by splitting on whitespace.
    This is a rough estimate - actual tokenization may differ.
    
    Args:
        formula: LaTeX formula string
    
    Returns:
        List of tokens
    """
    if pd.isna(formula):
        return []
    return str(formula).split()


def plot_label_length_distribution(im2latex_df, crohme_df):
    """
    Plot sequence length distributions for both datasets.
    
    Args:
        im2latex_df: DataFrame with 'seq_len' column
        crohme_df: DataFrame with 'seq_len' column
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # IM2LATEX
    axes[0].hist(im2latex_df['seq_len'], bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(im2latex_df['seq_len'].median(), color='red', linestyle='--',
                    label=f"Median: {im2latex_df['seq_len'].median():.0f}")
    axes[0].set_xlabel('Sequence Length (tokens)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('IM2LATEX: Sequence Length Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # CROHME
    axes[1].hist(crohme_df['seq_len'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1].axvline(crohme_df['seq_len'].median(), color='red', linestyle='--',
                    label=f"Median: {crohme_df['seq_len'].median():.0f}")
    axes[1].set_xlabel('Sequence Length (tokens)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('CROHME: Sequence Length Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def percentile_analysis(im2latex_df, crohme_df):
    """
    Print percentile analysis for sequence lengths.
    
    Args:
        im2latex_df: DataFrame with 'seq_len' column
        crohme_df: DataFrame with 'seq_len' column
    """
    percentiles = [90, 95, 99, 99.5]

    print("\n=== IM2LATEX Sequence Length Percentiles ===")
    for p in percentiles:
        val = np.percentile(im2latex_df['seq_len'], p)
        coverage = (im2latex_df['seq_len'] <= val).mean() * 100
        print(f"{p}th percentile: {val:.0f} tokens (covers {coverage:.1f}% of data)")

    print("\n=== CROHME Sequence Length Percentiles ===")
    for p in percentiles:
        val = np.percentile(crohme_df['seq_len'], p)
        coverage = (crohme_df['seq_len'] <= val).mean() * 100
        print(f"{p}th percentile: {val:.0f} tokens (covers {coverage:.1f}% of data)")


def top_common_tokens(cnt):
    """
    Print top 50 most common tokens.
    
    Args:
        cnt: Counter object with token frequencies
    """
    print("\n=== IM2LATEX: Top 50 Most Common Tokens ===")
    for token, count in cnt.most_common(50):
        print(f"{token:20s} : {count:6,}")


def rare_token(im2latex_token_counter, crohme_token_counter):
    """
    Analyze and print rare tokens (appearing < 5 times).
    
    Args:
        im2latex_token_counter: Counter for IM2LATEX tokens
        crohme_token_counter: Counter for CROHME tokens
    """
    im2latex_rare = {token: count for token, count in im2latex_token_counter.items() if count < 5}
    crohme_rare = {token: count for token, count in crohme_token_counter.items() if count < 5}

    print(f"\nIM2LATEX - Rare tokens (count < 5): {len(im2latex_rare):,}")
    print(f"CROHME   - Rare tokens (count < 5): {len(crohme_rare):,}")

    print("\n=== Sample of Rare Tokens (IM2LATEX) ===")
    for token, count in list(im2latex_rare.items())[:30]:
        print(f"{token:30s} : {count}")


def plot_topk_token_frequencies(im2latex_token_counter, crohme_token_counter, top_k=50, y_log_scale=True):
    """
    Plot top-k token frequencies with optional log scale.
    
    Args:
        im2latex_token_counter: Counter for IM2LATEX tokens
        crohme_token_counter: Counter for CROHME tokens
        top_k: Number of top tokens to display
        y_log_scale: Whether to use log scale on y-axis
    """
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=False)

    def _plot(counter, ax, title):
        items = counter.most_common(top_k) if hasattr(counter, "most_common") else sorted(counter.items(), key=lambda x: x[1], reverse=True)[:top_k]
        tokens = [t for t, c in items]
        counts = [c for t, c in items]

        ax.bar(tokens, counts)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Token", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        if y_log_scale:
            ax.set_yscale("log")
        ax.grid(True, axis="y", alpha=0.3)
        ax.tick_params(axis="x", labelrotation=70)
        ax.set_xticklabels(tokens, ha="right")

    _plot(im2latex_token_counter, axes[0], f"IM2LATEX: Top-{top_k} Token Frequencies (log y)")
    _plot(crohme_token_counter, axes[1], f"CROHME: Top-{top_k} Token Frequencies (log y)")

    plt.tight_layout()
    plt.show()


# ============================================================================
# STRUCTURAL ANALYSIS
# ============================================================================

def calculate_nesting_depth(formula):
    """
    Calculate maximum nesting depth of braces in a formula.
    
    Args:
        formula: LaTeX formula string
    
    Returns:
        Maximum nesting depth (int)
    """
    if pd.isna(formula):
        return 0
    
    max_depth = 0
    current_depth = 0
    
    for char in str(formula):
        if char == '{':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == '}':
            current_depth = max(0, current_depth - 1)
    
    return max_depth


def plot_nesting_depth_distribution(im2latex_df, crohme_df):
    """
    Plot nesting depth distributions for both datasets.
    
    Args:
        im2latex_df: DataFrame with 'nesting_depth' column
        crohme_df: DataFrame with 'nesting_depth' column
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # IM2LATEX
    axes[0].hist(im2latex_df['nesting_depth'], bins=range(0, im2latex_df['nesting_depth'].max() + 2), 
                edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Nesting Depth', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('IM2LATEX: Nesting Depth Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # CROHME
    axes[1].hist(crohme_df['nesting_depth'], bins=range(0, crohme_df['nesting_depth'].max() + 2),
                edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Nesting Depth', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('CROHME: Nesting Depth Distribution', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def classify_formula(formula):
    """
    Classify formula type based on special tokens.
    
    Args:
        formula: LaTeX formula string
    
    Returns:
        Comma-separated string of formula types or 'simple'
    """
    if pd.isna(formula):
        return 'other'
    
    formula_str = str(formula).lower()
    types = []
    
    # Calculus
    if any(token in formula_str for token in ['\\int', '\\sum', '\\lim', '\\partial']):
        types.append('calculus')
    
    # Linear Algebra (matrices)
    if any(token in formula_str for token in ['\\begin{matrix', '\\begin{bmatrix', '\\begin{pmatrix']):
        types.append('matrix')
    
    # Fractions
    if '\\frac' in formula_str:
        types.append('fraction')
    
    # Subscript/Superscript heavy
    if formula_str.count('_') + formula_str.count('^') > 3:
        types.append('subscript_heavy')
    
    return ','.join(types) if types else 'simple'


# ============================================================================
# CROSS-DATASET COMPARISON
# ============================================================================

def length_distribution(im2latex_df, crohme_df):
    """
    Compare sequence length distributions across datasets.
    
    Args:
        im2latex_df: DataFrame with 'seq_len' column
        crohme_df: DataFrame with 'seq_len' column
    """
    plt.figure(figsize=(14, 6))

    plt.hist(im2latex_df['seq_len'], bins=50, alpha=0.5, label='IM2LATEX', edgecolor='black')
    plt.hist(crohme_df['seq_len'], bins=50, alpha=0.5, label='CROHME', edgecolor='black', color='orange')

    plt.xlabel('Sequence Length (tokens)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Domain Comparison: Sequence Length Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n=== Sequence Length Statistics ===")
    print(f"IM2LATEX - Mean: {im2latex_df['seq_len'].mean():.1f}, Median: {im2latex_df['seq_len'].median():.1f}")
    print(f"CROHME   - Mean: {crohme_df['seq_len'].mean():.1f}, Median: {crohme_df['seq_len'].median():.1f}")


def aspect_ratio_distribution(im2latex_clean, crohme_clean):
    """
    Compare aspect ratio distributions across datasets.
    
    Args:
        im2latex_clean: DataFrame with 'aspect_ratio' column
        crohme_clean: DataFrame with 'aspect_ratio' column
    """
    plt.figure(figsize=(14, 6))

    plt.hist(im2latex_clean['aspect_ratio'], bins=50, alpha=0.5, label='IM2LATEX', edgecolor='black')
    plt.hist(crohme_clean['aspect_ratio'], bins=50, alpha=0.5, label='CROHME', edgecolor='black', color='orange')

    plt.xlabel('Aspect Ratio (Width/Height)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Domain Comparison: Aspect Ratio Distribution', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    print("\n=== Aspect Ratio Statistics ===")
    print(f"IM2LATEX - Mean: {im2latex_clean['aspect_ratio'].mean():.2f}, Median: {im2latex_clean['aspect_ratio'].median():.2f}")
    print(f"CROHME   - Mean: {crohme_clean['aspect_ratio'].mean():.2f}, Median: {crohme_clean['aspect_ratio'].median():.2f}")


def nesting_depth_distribution(im2latex_df, crohme_df, im2latex_clean, crohme_clean):
    """
    Create box plot comparison for sequence length, aspect ratio, and nesting depth.
    
    Args:
        im2latex_df: DataFrame with sequence/nesting data
        crohme_df: DataFrame with sequence/nesting data
        im2latex_clean: DataFrame with aspect ratio data
        crohme_clean: DataFrame with aspect ratio data
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Sequence Length
    data_seq = [im2latex_df['seq_len'], crohme_df['seq_len']]
    axes[0].boxplot(data_seq, labels=['IM2LATEX', 'CROHME'])
    axes[0].set_ylabel('Sequence Length', fontsize=12)
    axes[0].set_title('Sequence Length Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Aspect Ratio
    data_aspect = [im2latex_clean['aspect_ratio'].dropna(), crohme_clean['aspect_ratio'].dropna()]
    axes[1].boxplot(data_aspect, labels=['IM2LATEX', 'CROHME'])
    axes[1].set_ylabel('Aspect Ratio', fontsize=12)
    axes[1].set_title('Aspect Ratio Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    # Nesting Depth
    data_nest = [im2latex_df['nesting_depth'], crohme_df['nesting_depth']]
    axes[2].boxplot(data_nest, labels=['IM2LATEX', 'CROHME'])
    axes[2].set_ylabel('Nesting Depth', fontsize=12)
    axes[2].set_title('Nesting Depth Comparison', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ============================================================================
# PREPROCESSING ANALYSIS
# ============================================================================

def _clean_finite(x):
    """Remove non-finite values from array."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return x


def _safe_bins(x, default=30, min_bins=5):
    """
    Calculate safe number of bins for histogram.
    
    Args:
        x: Data array
        default: Default number of bins
        min_bins: Minimum number of bins
    
    Returns:
        Number of bins (int)
    """
    x = _clean_finite(x)
    if x.size < 2:
        return 1
    data_range = x.max() - x.min()
    if not np.isfinite(data_range) or data_range <= 0:
        return 1
    b = int(np.sqrt(x.size))
    b = max(min_bins, min(default, b))
    return b


def analyze_preprocessing_statistics(df, preprocess_fn, dataset_type='im2latex', sample_size=500):
    """
    Analyze pixel statistics before and after preprocessing.
    
    Args:
        df: DataFrame with 'image' column
        preprocess_fn: Preprocessing function to apply
        dataset_type: 'im2latex' or 'crohme'
        sample_size: Number of images to sample
    """
    from .constants import IM2LATEX_IMAGE_PATH, CROHME_IMAGE_PATH
    
    if dataset_type == 'im2latex':
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        img_paths = [IM2LATEX_IMAGE_PATH / img for img in df['image'].values]
    else:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        img_paths = [CROHME_IMAGE_PATH / img for img in df['image'].values]

    original_means, original_stds = [], []
    preprocessed_means, preprocessed_stds = [], []

    for img_path in tqdm(img_paths, desc=f"Analyzing {dataset_type}"):
        try:
            original = np.array(Image.open(img_path).convert('L'), dtype=np.float32) / 255.0
            original_means.append(float(original.mean()))
            original_stds.append(float(original.std()))

            preprocessed = preprocess_fn(img_path, augment=False)
            if preprocessed is not None:
                preprocessed = np.asarray(preprocessed, dtype=np.float32)
                preprocessed_means.append(float(np.mean(preprocessed)))
                preprocessed_stds.append(float(np.std(preprocessed)))
        except Exception:
            continue

    # Clean finite values
    original_means = _clean_finite(original_means)
    original_stds = _clean_finite(original_stds)
    preprocessed_means = _clean_finite(preprocessed_means)
    preprocessed_stds = _clean_finite(preprocessed_stds)

    if preprocessed_means.size == 0 or preprocessed_stds.size == 0:
        print("Không có dữ liệu preprocessed (preprocess_fn trả về None hoặc lỗi).")
        return

    # Print statistics
    print(f"\n{'='*60}")
    print(f"{dataset_type.upper()} Statistics (n={len(original_means)})")
    print(f"{'='*60}")
    print(f"{'Metric':<20} {'Original':<20} {'Preprocessed':<20}")
    print(f"{'-'*60}")
    print(f"{'Mean (avg)':<20} {original_means.mean():<20.4f} {preprocessed_means.mean():<20.4f}")
    print(f"{'Mean (std)':<20} {original_means.std():<20.4f} {preprocessed_means.std():<20.4f}")
    print(f"{'Std (avg)':<20} {original_stds.mean():<20.4f} {preprocessed_stds.mean():<20.4f}")
    print(f"{'Std (std)':<20} {original_stds.std():<20.4f} {preprocessed_stds.std():<20.4f}")
    print(f"{'='*60}\n")

    # Safe bins
    bins_om = _safe_bins(original_means, default=30)
    bins_pm = _safe_bins(preprocessed_means, default=30)
    bins_os = _safe_bins(original_stds, default=30)
    bins_ps = _safe_bins(preprocessed_stds, default=30)

    # Histogram comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mean distribution
    axes[0, 0].hist(original_means, bins=bins_om, alpha=0.7, label='Original', edgecolor='black')
    axes[0, 0].axvline(original_means.mean(), linestyle='--', label=f'Mean: {original_means.mean():.4f}')
    axes[0, 0].set_xlabel('Pixel Mean'); axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'Original: Mean (bins={bins_om})')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].hist(preprocessed_means, bins=bins_pm, alpha=0.7, label='Preprocessed',
                    color='orange', edgecolor='black')
    axes[0, 1].axvline(preprocessed_means.mean(), color='red', linestyle='--',
                       label=f'Mean: {preprocessed_means.mean():.4f}')
    axes[0, 1].set_xlabel('Pixel Mean'); axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Preprocessed: Mean (bins={bins_pm})')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    # Std distribution
    axes[1, 0].hist(original_stds, bins=bins_os, alpha=0.7, label='Original', edgecolor='black')
    axes[1, 0].axvline(original_stds.mean(), linestyle='--', label=f'Mean: {original_stds.mean():.4f}')
    axes[1, 0].set_xlabel('Pixel Std'); axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Original: Std (bins={bins_os})')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(preprocessed_stds, bins=bins_ps, alpha=0.7, label='Preprocessed',
                    color='orange', edgecolor='black')
    axes[1, 1].axvline(preprocessed_stds.mean(), color='red', linestyle='--',
                       label=f'Mean: {preprocessed_stds.mean():.4f}')
    axes[1, 1].set_xlabel('Pixel Std'); axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Preprocessed: Std (bins={bins_ps})')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.suptitle(f'{dataset_type.upper()}: Pixel Statistics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    from .constants import (
        IM2LATEX_IMAGE_PATH,
        IM2LATEX_LABEL_PATH,
        CROHME_IMAGE_PATH,
        CROHME_CSV_PATH
    )
    
    print("=" * 80)
    print("IMAGE STATISTICS ANALYSIS")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading datasets...")
    
    # IM2LATEX
    im2latex_train = pd.read_csv(IM2LATEX_LABEL_PATH / "im2latex_train.csv")
    im2latex_val = pd.read_csv(IM2LATEX_LABEL_PATH / "im2latex_validate.csv")
    im2latex_test = pd.read_csv(IM2LATEX_LABEL_PATH / "im2latex_test.csv")
    
    im2latex_df = pd.concat([im2latex_train, im2latex_val, im2latex_test], ignore_index=True)
    im2latex_df['dataset'] = 'IM2LATEX'
    im2latex_df['image_path'] = im2latex_df['image'].apply(lambda x: IM2LATEX_IMAGE_PATH / x)
    
    # CROHME
    crohme_df = pd.read_csv(CROHME_CSV_PATH)
    crohme_df['dataset'] = 'CROHME'
    crohme_df['image_path'] = crohme_df['image'].apply(lambda x: CROHME_IMAGE_PATH / x)
    
    print(f"✓ IM2LATEX: {len(im2latex_df):,} samples")
    print(f"✓ CROHME: {len(crohme_df):,} samples")
    
    # Extract dimensions
    print("\nExtracting dimensions...")
    im2latex_with_dims = extract_image_dimensions(im2latex_df, sample_size=5000)
    crohme_with_dims = extract_image_dimensions(crohme_df, sample_size=None)
    
    # Clean
    im2latex_clean = im2latex_with_dims.dropna(subset=['width', 'height'])
    crohme_clean = crohme_with_dims.dropna(subset=['width', 'height'])
    
    # Print statistics
    print("\n" + "=" * 80)
    print("IM2LATEX DIMENSIONS")
    print("=" * 80)
    print(f"Width  - Mean: {im2latex_clean['width'].mean():.1f}, Median: {im2latex_clean['width'].median():.1f}")
    print(f"Height - Mean: {im2latex_clean['height'].mean():.1f}, Median: {im2latex_clean['height'].median():.1f}")
    print(f"Aspect Ratio - Mean: {im2latex_clean['aspect_ratio'].mean():.2f}")
    
    print("\n" + "=" * 80)
    print("CROHME DIMENSIONS")
    print("=" * 80)
    print(f"Width  - Mean: {crohme_clean['width'].mean():.1f}, Median: {crohme_clean['width'].median():.1f}")
    print(f"Height - Mean: {crohme_clean['height'].mean():.1f}, Median: {crohme_clean['height'].median():.1f}")
    print(f"Aspect Ratio - Mean: {crohme_clean['aspect_ratio'].mean():.2f}")
    
    # Pixel intensity
    print("\nAnalyzing pixel intensity...")
    im2latex_intensity = analyze_pixel_intensity(im2latex_clean, 500, "IM2LATEX")
    crohme_intensity = analyze_pixel_intensity(crohme_clean, 500, "CROHME")
    
    print("\n" + "=" * 80)
    print("PIXEL INTENSITY")
    print("=" * 80)
    print(f"IM2LATEX - Mean: {im2latex_intensity['mean']:.4f}, Std: {im2latex_intensity['std']:.4f}")
    print(f"CROHME   - Mean: {crohme_intensity['mean']:.4f}, Std: {crohme_intensity['std']:.4f}")
    
    print("\nAnalysis complete!")


# Export all public functions
__all__ = [
    # Image analysis
    'extract_image_dimensions',
    'analyze_pixel_intensity',
    'plot_image_statistics',
    'plot_pixel_intensity',
    # Label analysis
    'tokenize_formula',
    'plot_label_length_distribution',
    'percentile_analysis',
    'top_common_tokens',
    'rare_token',
    'plot_topk_token_frequencies',
    # Structural analysis
    'calculate_nesting_depth',
    'plot_nesting_depth_distribution',
    'classify_formula',
    # Comparison
    'length_distribution',
    'aspect_ratio_distribution',
    'nesting_depth_distribution',
    'analyze_preprocessing_statistics',
]
