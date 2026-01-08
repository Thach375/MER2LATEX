"""
Inference Functions
===================
Functions for preprocessing images and running model inference.
"""

import os
import tempfile
import torch
import numpy as np
from PIL import Image

from src.models.architectures import ModelA_CTC
from src.tokenizer.tokenize import LaTeXTokenizer
from src.preprocessing.preprocess_pipelines import preprocess_im2latex, preprocess_crohme


def preprocess_image(image: np.ndarray, preprocessing_type: str = "IM2LATEX") -> np.ndarray:
    """
    Preprocess image for model input.
    
    Args:
        image: Input image as numpy array (RGB or grayscale)
        preprocessing_type: Type of preprocessing ("IM2LATEX" or "CROHME")
    
    Returns:
        Preprocessed image (128, 768)
    """
    # Convert numpy array to PIL Image
    if isinstance(image, np.ndarray):
        # Handle different color formats
        if len(image.shape) == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode='L')
        elif len(image.shape) == 3:
            # RGB or RGBA
            if image.shape[2] == 3:
                pil_image = Image.fromarray(image, mode='RGB')
            elif image.shape[2] == 4:
                pil_image = Image.fromarray(image, mode='RGBA')
            else:
                pil_image = Image.fromarray(image)
        else:
            pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Save to temporary file for preprocessing functions
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        pil_image.save(tmp.name)
        tmp_path = tmp.name
    
    try:
        # Apply preprocessing pipeline
        if preprocessing_type == "CROHME":
            processed = preprocess_crohme(tmp_path, augment=False)
        else:  # Default to IM2LATEX
            processed = preprocess_im2latex(tmp_path, augment=False)
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return processed


def predict_latex(
    model,
    tokenizer: LaTeXTokenizer,
    image: np.ndarray,
    device: torch.device,
    max_len: int = 200
) -> str:
    """
    Predict LaTeX from preprocessed image.
    
    Args:
        model: Loaded model (any of Model A, B, C, D)
        tokenizer: LaTeX tokenizer
        image: Preprocessed image (128, 768)
        device: Device to run inference on
        max_len: Maximum sequence length
    
    Returns:
        Predicted LaTeX string
    """
    try:
        # Convert to tensor
        # Image shape: (128, 768) -> (1, 1, 128, 768)
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0) / 255.0
        image_tensor = image_tensor.to(device)
        
        # Predict - different methods for different model types
        with torch.no_grad():
            if isinstance(model, ModelA_CTC):
                # Model A uses CTC decoding
                predictions = model.decode_greedy(image_tensor)
                # CTC predictions shape: (B, T)
                pred_ids = predictions[0].cpu().tolist()
                
                # Remove consecutive duplicates and blank tokens (0)
                decoded_ids = []
                prev_id = None
                for id in pred_ids:
                    if id != 0 and id != prev_id:  # Skip blank and duplicates
                        decoded_ids.append(id)
                    prev_id = id
                
                latex_output = tokenizer.decode(decoded_ids, skip_special=True)
            
            else:
                # Model B, C, D use greedy decoding
                predictions = model.decode_greedy(
                    image_tensor,
                    max_len=max_len,
                    bos_id=tokenizer.bos_id,
                    eos_id=tokenizer.eos_id
                )
                
                # Decode predictions
                pred_ids = predictions[0].cpu().tolist()
                latex_output = tokenizer.decode(pred_ids, skip_special=True)
        
        return latex_output
    
    except Exception as e:
        print(f"[ERROR] Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        raise
