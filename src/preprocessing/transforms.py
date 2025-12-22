"""
Image Preprocessing Transforms
===============================
Basic image transformation functions.

Run standalone:
    python -m src.preprocessing.transforms
"""

import numpy as np
import cv2
from PIL import Image
from src.utils.constants import *

def convert_to_grayscale(image):
    """
    Convert image to grayscale.
    
    Args:
        image: PIL Image or numpy array
    
    Returns:
        Grayscale numpy array (H, W)
    """
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        return np.array(image)
    
    if len(image.shape) == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    
    return image


def resize_keep_aspect_ratio(image, target_height=TARGET_HEIGHT):
    """
    Resize image to target height while preserving aspect ratio.
    
    Args:
        image: numpy array (H, W)
        target_height: target height in pixels
    
    Returns:
        Resized image (target_height, new_width)
    """
    h, w = image.shape[:2]
    
    # Calculate new width maintaining aspect ratio
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    
    # Resize
    resized = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    
    return resized

def pad_to_width(image, target_width=TARGET_WIDTH, pad_value=PAD_VALUE):
    """
    Pad image to target width with white padding on the right.
    If image is wider than target, resize it to fit.
    
    Args:
        image: numpy array (H, W)
        target_width: target width in pixels
        pad_value: padding pixel value (255 = white)
    
    Returns:
        Padded image (H, target_width)
    """
    h, w = image.shape[:2]
    
    if w > target_width:
        # If image is too wide, resize to fit
        scale = target_width / w
        new_h = int(h * scale)
        image = cv2.resize(image, (target_width, new_h), interpolation=cv2.INTER_AREA)
        
        # Pad vertically if needed
        if new_h < TARGET_HEIGHT:
            pad_h = TARGET_HEIGHT - new_h
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, 0, 
                                      cv2.BORDER_CONSTANT, value=pad_value)
        return image
    
    # Pad on the right
    pad_width = target_width - w
    padded = cv2.copyMakeBorder(image, 0, 0, 0, pad_width, 
                                cv2.BORDER_CONSTANT, value=pad_value)
    
    return padded

def crop_content(image, margin=5):
    """
    Crop image to bounding box of content (non-white pixels).
    Essential for CROHME to remove excess whitespace.
    
    Args:
        image: numpy array (H, W), grayscale
        margin: pixels to add around content
    
    Returns:
        Cropped image
    """
    # Threshold to binary (find non-white pixels)
    binary = (image < 250).astype(np.uint8)
    
    # Find contours
    coords = cv2.findNonZero(binary)
    
    if coords is None:
        # Empty image, return as is
        return image
    
    # Get bounding box
    x, y, w, h = cv2.boundingRect(coords)
    
    # Add margin
    x = max(0, x - margin)
    y = max(0, y - margin)
    w = min(image.shape[1] - x, w + 2 * margin)
    h = min(image.shape[0] - y, h + 2 * margin)
    
    # Crop
    cropped = image[y:y+h, x:x+w]
    
    return cropped


def normalize_image(image, mean=0.0, std=1.0):
    """
    Normalize image to [0, 1] range, then standardize.
    
    Args:
        image: numpy array
        mean: normalization mean
        std: normalization std
    
    Returns:
        Normalized image (float32)
    """
    # Convert to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Standardize
    if std > 0:
        image = (image - mean) / std
    
    return image

def per_image_normalize(image):
    """
    Per-image normalization (zero mean, unit variance).
    Important for CROHME handwritten images.
    
    Args:
        image: numpy array
    
    Returns:
        Normalized image
    """
    image = image.astype(np.float32) / 255.0
    mean = image.mean()
    std = image.std()
    
    if std > 0:
        image = (image - mean) / std
    
    return image


if __name__ == "__main__":
    print("=" * 80)
    print("IMAGE PREPROCESSING TRANSFORMS TEST")
    print("=" * 80)
    
    # Create test image with margin (simulating handwritten content)
    print("\n🎨 Creating test image (100x300, grayscale with content)...")
    test_img = np.ones((100, 300), dtype=np.uint8) * 255
    test_img[20:80, 50:250] = 128  # Gray box
    test_img[30:70, 100:200] = 0   # Black box (simulating ink)
    
    print(f"Original shape: {test_img.shape}")
    print(f"Original dtype: {test_img.dtype}")
    print(f"Original range: [{test_img.min()}, {test_img.max()}]")
    
    # Test grayscale conversion
    print("\n🔲 Testing convert_to_grayscale...")
    gray = convert_to_grayscale(test_img)
    print(f"Grayscale shape: {gray.shape}")
    print(f"Grayscale dtype: {gray.dtype}")
    
    # Test content cropping
    print("\n✂️ Testing crop_content...")
    cropped = crop_content(test_img, margin=5)
    print(f"Cropped shape: {cropped.shape}")
    print(f"Cropped dtype: {cropped.dtype}")
    print(f"Removed pixels: {test_img.shape[0] * test_img.shape[1] - cropped.shape[0] * cropped.shape[1]}")
    
    # Test resize keeping aspect ratio
    print("\n📏 Testing resize_keep_aspect_ratio...")
    resized = resize_keep_aspect_ratio(cropped, target_height=128)
    print(f"Resized shape: {resized.shape}")
    print(f"Aspect ratio preserved: {cropped.shape[1]/cropped.shape[0]:.2f} -> {resized.shape[1]/resized.shape[0]:.2f}")
    
    # Test padding to width
    print("\n📐 Testing pad_to_width...")
    padded = pad_to_width(resized, target_width=768, pad_value=255)
    print(f"Padded shape: {padded.shape}")
    print(f"Padded dtype: {padded.dtype}")
    print(f"Padded range: [{padded.min()}, {padded.max()}]")
    
    # Test global normalization
    print("\n📊 Testing normalize_image (mean=0.5, std=0.5)...")
    normalized_global = normalize_image(padded, mean=0.5, std=0.5)
    print(f"Normalized shape: {normalized_global.shape}")
    print(f"Normalized dtype: {normalized_global.dtype}")
    print(f"Normalized range: [{normalized_global.min():.4f}, {normalized_global.max():.4f}]")
    print(f"Normalized mean: {normalized_global.mean():.4f}")
    print(f"Normalized std: {normalized_global.std():.4f}")
    
    # Test per-image normalization
    print("\n📊 Testing per_image_normalize...")
    normalized_per = per_image_normalize(padded)
    print(f"Per-image normalized shape: {normalized_per.shape}")
    print(f"Per-image normalized dtype: {normalized_per.dtype}")
    print(f"Per-image normalized range: [{normalized_per.min():.4f}, {normalized_per.max():.4f}]")
    print(f"Per-image normalized mean: {normalized_per.mean():.4f}")
    print(f"Per-image normalized std: {normalized_per.std():.4f}")
    
    print("\n✅ All transform functions tested successfully!")
