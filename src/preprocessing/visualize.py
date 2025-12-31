import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.utils.constants import (
    IM2LATEX_IMAGE_PATH,
    CROHME_IMAGE_PATH,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    NUM_CHANNELS
)
from .preprocess_pipelines import preprocess_im2latex, preprocess_crohme
from .transforms import crop_content, resize_keep_aspect_ratio, pad_to_width

# Progress of preprocessing
def visualize_preprocessing(df1, preprocess_df, dataset_type='im2latex', num_samples=3):
    """
    Visualize original vs preprocessed images with BORDER around plots.
    """
    if dataset_type == 'im2latex':
        df = df1.sample(n=num_samples, random_state=42)
        img_paths = [IM2LATEX_IMAGE_PATH / img for img in df['image'].values]
        preprocess_fn = preprocess_df
    else:
        df = df1.sample(n=num_samples, random_state=42)
        img_paths = [CROHME_IMAGE_PATH / img for img in df['image'].values]
        preprocess_fn = preprocess_df
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(16, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(img_paths):
        # ===== ORIGINAL =====
        original = Image.open(img_path)
        if original.mode != 'L':
            original = original.convert('L')
        
        ax = axes[idx, 0]
        ax.imshow(original, cmap='gray')
        ax.set_title(
            f'Original: {original.size[0]}x{original.size[1]}',
            fontsize=12, fontweight='bold'
        )
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Border
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_edgecolor('black')
        
        # ===== PREPROCESSED =====
        preprocessed = preprocess_fn(img_path, augment=False)
        ax = axes[idx, 1]
        if preprocessed is not None:
            ax.imshow(preprocessed, cmap='gray')
            ax.set_title(
                f'Preprocessed: {preprocessed.shape[1]}x{preprocessed.shape[0]}',
                fontsize=12, fontweight='bold'
            )
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(2)
                spine.set_edgecolor('black')
    
    plt.suptitle(
        f'{dataset_type.upper()} Preprocessing Comparison',
        fontsize=16,
        fontweight='bold'
    )
    
    plt.tight_layout()
    plt.show()

# Check CROHME cropping pipeline
def add_axes_border(ax, lw=2, color='black'):
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_edgecolor(color)

def visualize_crohme_cropping(crohme_df, num_samples=3):
    """
    Visualize the effect of content cropping on CROHME images.
    Shows: Original → Cropped → Resized → Final
    """
    df = crohme_df.sample(n=num_samples, random_state=43)
    img_paths = [CROHME_IMAGE_PATH / img for img in df['image'].values]
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(img_paths):
        # Step-by-step processing
        original = np.array(Image.open(img_path).convert('L'))
        cropped  = crop_content(original, margin=10)
        resized  = resize_keep_aspect_ratio(cropped, TARGET_HEIGHT)
        final    = pad_to_width(resized, TARGET_WIDTH)
        
        imgs = [original, cropped, resized, final]
        titles = [
            f'Original\n{original.shape[1]}×{original.shape[0]}',
            f'Cropped\n{cropped.shape[1]}×{cropped.shape[0]}',
            f'Resized\n{resized.shape[1]}×{resized.shape[0]}',
            f'Final (Padded)\n{final.shape[1]}×{final.shape[0]}'
        ]
        
        for j in range(4):
            ax = axes[idx, j]
            ax.imshow(imgs[j], cmap='gray')
            ax.set_title(titles[j])
            ax.set_xticks([]); ax.set_yticks([])   # hoặc ax.axis('off') nhưng sẽ tắt cả frame
            add_axes_border(ax, lw=2, color='black') # đổi màu/độ dày tuỳ bạn
    
    plt.suptitle('CROHME Cropping Pipeline', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    
# Augmentation Visualization
def add_black_border(ax, lw=2):
    # Ẩn ticks nhưng vẫn giữ khung
    ax.set_xticks([])
    ax.set_yticks([])
    # Bật viền (spines)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(lw)
        spine.set_edgecolor('black')

def visualize_augmentation(df1, preprocess_df, dataset_type='crohme', num_augmentations=5):
    """
    Show effect of augmentation.
    
    Args:
        dataset_type: 'im2latex' or 'crohme'
        num_augmentations: number of augmented versions to show
    """
    if dataset_type == 'im2latex':
        df = df1.sample(n=1, random_state=42)
        img_path = IM2LATEX_IMAGE_PATH / df['image'].values[0]
        preprocess_fn = preprocess_df
    else:
        df = df1.sample(n=1, random_state=42)
        img_path = CROHME_IMAGE_PATH / df['image'].values[0]
        preprocess_fn = preprocess_df
    
    fig, axes = plt.subplots(1, num_augmentations + 1, 
                             figsize=(4 * (num_augmentations + 1), 4))
    
    # Original (no augmentation)
    original = preprocess_fn(img_path, augment=False)
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title('Original\n(No Augmentation)', fontweight='bold')
    add_black_border(axes[0], lw=2)
    
    # Augmented versions
    for i in range(num_augmentations):
        augmented = preprocess_fn(img_path, augment=True)
        axes[i + 1].imshow(augmented, cmap='gray')
        axes[i + 1].set_title(f'Augmented #{i+1}')
        add_black_border(axes[i + 1], lw=2)
    
    plt.suptitle(f'{dataset_type.upper()} Augmentation Examples', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
# Summary of preprocessing
def summarize(im2latex_splits, crohme_splits):
    # Summary statistics
    print("="*70)
    print("PREPROCESSING SUMMARY")
    print("="*70)

    print("\nIM2LATEX Dataset:")
    print(f"  Train:      {len(im2latex_splits['train']):>6,} samples")
    print(f"  Validation: {len(im2latex_splits['val']):>6,} samples")
    print(f"  Test:       {len(im2latex_splits['test']):>6,} samples")
    print(f"  Total:      {sum(len(df) for df in im2latex_splits.values()):>6,} samples")

    print("\nCROHME Dataset:")
    print(f"  Train:      {len(crohme_splits['train']):>6,} samples")
    print(f"  Validation: {len(crohme_splits['val']):>6,} samples")
    print(f"  Test:       {len(crohme_splits['test']):>6,} samples")
    print(f"  Total:      {sum(len(df) for df in crohme_splits.values()):>6,} samples")

    print("\nCanonical Format:")
    print(f"  Shape:      [{NUM_CHANNELS}, {TARGET_HEIGHT}, {TARGET_WIDTH}]")
    print(f"  Channels:   Grayscale (1 channel)")
    print(f"  Height:     Fixed at {TARGET_HEIGHT} pixels")
    print(f"  Width:      Padded to {TARGET_WIDTH} pixels")

    print("\nPreprocessing Strategy:")
    print("  IM2LATEX:   Minimal processing, preserve sharpness")
    print("  CROHME:     Content cropping + contrast enhancement")

    print("\nOutput Files:")
    print(f"  IM2LATEX:   {IM2LATEX_OUTPUT_PATH}")
    print(f"  CROHME:     {CROHME_OUTPUT_PATH}")
    print("="*70)
    
    