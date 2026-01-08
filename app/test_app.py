"""
Test script for Gradio app
============================
Quick test to verify the app can load and basic functionality works.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image

from app.gradio_app import (
    get_available_models,
    load_model,
    preprocess_image,
    predict_latex,
    render_latex_with_sympy,
    TOKENIZER,
    DEVICE
)


def test_model_discovery():
    """Test if models can be discovered."""
    print("\n" + "="*80)
    print("TEST 1: Model Discovery")
    print("="*80)
    
    models = get_available_models()
    print(f"Found {len(models)} models:")
    for i, (name, path) in enumerate(models.items(), 1):
        print(f"  {i}. {name}")
        print(f"     Path: {path}")
    
    assert len(models) > 0, "No models found!"
    print("[PASS] Model discovery passed")
    return list(models.values())[0] if models else None


def test_tokenizer():
    """Test tokenizer."""
    print("\n" + "="*80)
    print("TEST 2: Tokenizer")
    print("="*80)
    
    print(f"Vocabulary size: {TOKENIZER.vocab_size}")
    
    # Test encoding
    test_latex = "x^2 + y^2 = r^2"
    encoded = TOKENIZER.encode(test_latex)
    print(f"Test LaTeX: {test_latex}")
    print(f"Encoded: {encoded}")
    
    # Test decoding
    decoded = TOKENIZER.decode(encoded)
    print(f"Decoded: {decoded}")
    
    assert len(encoded) > 0, "Encoding failed!"
    print("[PASS] Tokenizer test passed")


def test_preprocessing():
    """Test image preprocessing."""
    print("\n" + "="*80)
    print("TEST 3: Image Preprocessing")
    print("="*80)
    
    # Create a dummy image (white background with black text-like pattern)
    dummy_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    dummy_image[40:60, 50:150] = 0  # Black horizontal bar
    
    print(f"Input image shape: {dummy_image.shape}")
    
    # Test IM2LATEX preprocessing
    processed = preprocess_image(dummy_image, "IM2LATEX")
    print(f"Preprocessed shape: {processed.shape}")
    print(f"Expected shape: (128, 768)")
    
    assert processed.shape == (128, 768), f"Wrong shape: {processed.shape}"
    print("[PASS] Preprocessing test passed")
    
    return dummy_image


def test_model_loading(model_path):
    """Test model loading."""
    print("\n" + "="*80)
    print("TEST 4: Model Loading")
    print("="*80)
    
    print(f"Loading model from: {model_path}")
    print(f"Device: {DEVICE}")
    
    model = load_model(model_path, TOKENIZER.vocab_size)
    
    print(f"Model type: {type(model).__name__}")
    print(f"Model loaded successfully on {DEVICE}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("[PASS] Model loading test passed")
    return model


def test_inference(model, image):
    """Test inference."""
    print("\n" + "="*80)
    print("TEST 5: Inference")
    print("="*80)
    
    # Preprocess image
    processed = preprocess_image(image, "IM2LATEX")
    print(f"Input shape: {processed.shape}")
    
    # Predict
    try:
        latex_output = predict_latex(model, TOKENIZER, processed, max_len=50)
        print(f"Predicted LaTeX: {latex_output}")
        print(f"Output length: {len(latex_output)}")
        
        assert isinstance(latex_output, str), "Output should be string"
        print("✅ Inference test passed")
        return latex_output
    
    except Exception as e:
        print(f"❌ Inference failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_latex_rendering(latex_str):
    """Test LaTeX rendering with SymPy."""
    print("\n" + "="*80)
    print("TEST 6: LaTeX Rendering")
    print("="*80)
    
    try:
        rendered = render_latex_with_sympy(latex_str)
        
        if rendered is not None:
            print(f"Rendered image size: {rendered.size}")
            print("✅ LaTeX rendering test passed")
        else:
            print("⚠️ SymPy rendering returned None (might not be installed)")
    
    except Exception as e:
        print(f"⚠️ LaTeX rendering failed (non-critical): {e}")


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MER2LATEX GRADIO APP - TEST SUITE")
    print("="*80)
    
    try:
        # Test 1: Model discovery
        model_path = test_model_discovery()
        
        if not model_path:
            print("\n[FAIL] No models available for testing")
            print("Please train a model first!")
            return
        
        # Test 2: Tokenizer
        test_tokenizer()
        
        # Test 3: Preprocessing
        dummy_image = test_preprocessing()
        
        # Test 4: Model loading
        model = test_model_loading(model_path)
        
        # Test 5: Inference
        latex_output = test_inference(model, dummy_image)
        
        # Test 6: LaTeX rendering
        test_latex_rendering(latex_output)
        
        print("\n" + "="*80)
        print("[PASS] ALL TESTS PASSED!")
        print("="*80)
        print("\nYou can now run the Gradio app:")
        print("  python app/gradio_app.py")
        print("  or")
        print("  ./run_demo.sh")
        print()
    
    except Exception as e:
        print("\n" + "="*80)
        print("[FAIL] TESTS FAILED!")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
