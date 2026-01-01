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
from src.utils.constants import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    PAD_VALUE
)

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
