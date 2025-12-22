import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from src.utils.constants import *


# Domain comparison functions
def length_distribution(im2latex_df, crohme_df):
    # Overlay histogram
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
    # Overlay aspect ratio histogram
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
    # Overlay nesting depth histogram
    # Box plot comparison
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


# Processing analysis 
def _clean_finite(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return x

def _safe_bins(x, default=30, min_bins=5):
    """
    Trả về số bins phù hợp.
    - Nếu data range ~ 0 (tất cả gần như bằng nhau) -> bins=1 (tránh lỗi).
    - Nếu ít điểm -> giảm bins.
    """
    x = _clean_finite(x)
    if x.size < 2:
        return 1
    data_range = x.max() - x.min()
    if not np.isfinite(data_range) or data_range <= 0:
        return 1
    # Quy tắc đơn giản theo số mẫu (ổn cho n≈500)
    b = int(np.sqrt(x.size))  # ~22 bins cho 500
    b = max(min_bins, min(default, b))
    return b

def analyze_preprocessing_statistics(df, preprocess_fn, dataset_type='im2latex', sample_size=500):
    """
    Analyze pixel statistics before and after preprocessing.
    """
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
                # đảm bảo là numpy array số
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

    # Nếu preprocess trả về rỗng -> báo rõ
    if preprocessed_means.size == 0 or preprocessed_stds.size == 0:
        print("⚠️ Không có dữ liệu preprocessed (preprocess_fn trả về None hoặc lỗi).")
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
