"""
Image Statistics Analysis
=========================
Functions để analyze image dimensions, pixel intensity.

Run standalone:
    python -m src.analysis.image_stats
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
    # Scatter plot: Width vs Height
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
    # Aspect Ratio Distribution
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

if __name__ == "__main__":
    from ..utils.constants import (IM2LATEX_IMAGE_PATH,
                                   IM2LATEX_LABEL_PATH,
                                   CROHME_IMAGE_PATH,
                                   CROHME_CSV_PATH
                                   )
    print("=" * 80)
    print("IMAGE STATISTICS ANALYSIS")
    print("=" * 80)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    
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
    print("\n📏 Extracting dimensions...")
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
    print("\n🎨 Analyzing pixel intensity...")
    im2latex_intensity = analyze_pixel_intensity(im2latex_clean, 500, "IM2LATEX")
    crohme_intensity = analyze_pixel_intensity(crohme_clean, 500, "CROHME")
    
    print("\n" + "=" * 80)
    print("PIXEL INTENSITY")
    print("=" * 80)
    print(f"IM2LATEX - Mean: {im2latex_intensity['mean']:.4f}, Std: {im2latex_intensity['std']:.4f}")
    print(f"CROHME   - Mean: {crohme_intensity['mean']:.4f}, Std: {crohme_intensity['std']:.4f}")
    
    # Visualization
    print("\n📊 Creating visualizations...")

    df_dict = {
        'IM2LATEX': im2latex_clean,
        'CROHME': crohme_clean
    }

    intensity_dict = {
        'IM2LATEX': im2latex_intensity,
        'CROHME': crohme_intensity
    }

    # Plot dimensions
    plot_image_statistics(df_dict)

    # Plot pixel intensity
    plot_pixel_intensity(intensity_dict)

    print("\n✅ Analysis complete!")