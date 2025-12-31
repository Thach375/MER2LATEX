"""
Preprocessing Pipelines
========================
Full preprocessing pipelines for IM2LATEX and CROHME.

Run standalone:
    python -m src.preprocessing.preprocess_pipelines
"""

import numpy as np
from PIL import Image
import cv2
import albumentations as A
from .transforms import (
    convert_to_grayscale,
    resize_keep_aspect_ratio,
    pad_to_width,
    crop_content,
    TARGET_HEIGHT,
    TARGET_WIDTH
)


def preprocess_im2latex(image_path, augment=False):
    """
    Preprocess IM2LATEX image to canonical format.
    
    Args:
        image_path: path to image file
        augment: whether to apply augmentation (for training)
    
    Returns:
        Preprocessed image (128, 768) or None if error
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Step 1: Convert to grayscale
        image = convert_to_grayscale(image)
        
        # Step 2: Resize keeping aspect ratio
        image = resize_keep_aspect_ratio(image, TARGET_HEIGHT)
        
        # Step 3: Pad to target width
        image = pad_to_width(image, TARGET_WIDTH)
        
        # Step 4: Light augmentation (optional, for training only)
        if augment:
            transform = A.Compose([
                A.GaussianBlur(blur_limit=(3, 3), p=0.1),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1, 
                    contrast_limit=0.1, 
                    p=0.2
                ),
            ])
            image = transform(image=image)['image']
        
        # Step 5: Ensure uint8 format
        image = image.astype(np.uint8)
        
        return image
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def preprocess_crohme(image_path, augment=False):
    """
    Preprocess CROHME image to canonical format.
    Includes content cropping and contrast enhancement.
    
    Args:
        image_path: path to image file
        augment: whether to apply augmentation (for training)
    
    Returns:
        Preprocessed image (128, 768) or None if error
    """
    try:
        # Load image
        image = Image.open(image_path)
        
        # Step 1: Convert to grayscale
        image = convert_to_grayscale(image)
        
        # Step 2: CRITICAL - Crop content (remove whitespace)
        image = crop_content(image, margin=10)
        
        # Step 3: Resize keeping aspect ratio
        image = resize_keep_aspect_ratio(image, TARGET_HEIGHT)
        
        # Step 4: Pad to target width
        image = pad_to_width(image, TARGET_WIDTH)
        
        # Step 5: Augmentation (for training)
        if augment:
            transform = A.Compose([
                # Contrast enhancement (CROHME has low contrast)
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.3,
                    p=0.7
                ),
                # Simulate different pen thickness
                A.OneOf([
                    A.Sharpen(alpha=(0.2, 0.5), p=1.0),
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                ], p=0.3),
                # Simulate rotation/skew
                A.Rotate(limit=5, border_mode=cv2.BORDER_CONSTANT, p=0.3),
            ])
            image = transform(image=image)['image']
        
        # Step 6: Ensure uint8 format
        image = image.astype(np.uint8)
        
        return image
    
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


if __name__ == '__main__':
    print("[INFO] IM2LATEX preprocessing pipeline ready")
    print("[INFO] CROHME preprocessing pipeline ready")
    
    # Test the pipelines with sample images if they exist
    from pathlib import Path
    from src.utils.constants import IM2LATEX_IMAGE_PATH, CROHME_IMAGE_PATH
    
    # Test IM2LATEX
    im2latex_samples = list(IM2LATEX_IMAGE_PATH.glob("*.png"))[:1]
    if im2latex_samples:
        img = preprocess_im2latex(im2latex_samples[0], augment=False)
        if img is not None:
            print(f"[OK] IM2LATEX test - shape: {img.shape}")
    
    # Test CROHME
    crohme_samples = list(CROHME_IMAGE_PATH.glob("*.png"))[:1]
    if crohme_samples:
        img = preprocess_crohme(crohme_samples[0], augment=False)
        if img is not None:
            print(f"[OK] CROHME test - shape: {img.shape}")