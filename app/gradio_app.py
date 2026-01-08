"""
Gradio Demo App for Mathematical Expression Recognition (MER2LATEX)
====================================================================
Upload an image of a mathematical formula and convert it to LaTeX.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import torch
import numpy as np
from PIL import Image

# Import project modules
from src.tokenizer.tokenize import LaTeXTokenizer
from src.utils.model_discovery import get_available_models, load_model
from src.utils.latex_utils import clean_latex_output, render_latex_with_sympy
from src.inference.predict import preprocess_image, predict_latex


# ============================================================================
# Configuration
# ============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = Path(__file__).parent.parent / 'checkpoints'
VOCAB_PATH = Path(__file__).parent.parent / 'vocab.json'

print(f"[INFO] Using device: {DEVICE}")
print(f"[INFO] Checkpoint directory: {CHECKPOINT_DIR}")
print(f"[INFO] Vocab path: {VOCAB_PATH}")

# Global variables
MODELS = get_available_models(CHECKPOINT_DIR)
TOKENIZER = LaTeXTokenizer(vocab_path=VOCAB_PATH)
CURRENT_MODEL = None
CURRENT_MODEL_PATH = None

print(f"[INFO] Found {len(MODELS)} model checkpoints")
print(f"[INFO] Vocabulary size: {TOKENIZER.vocab_size}")


def process_image(
    image: np.ndarray,
    model_choice: str,
    preprocessing_type: str
) -> Tuple[str, Optional[Image.Image], Image.Image]:
    """
    Main processing function for Gradio interface.
    
    Args:
        image: Input image
        model_choice: Selected model checkpoint
        preprocessing_type: Type of preprocessing
    
    Returns:
        (latex_output, rendered_image, preprocessed_image)
    """
    global CURRENT_MODEL, CURRENT_MODEL_PATH
    
    if image is None:
        return "[WARNING] Please upload an image", None, None
    
    if not model_choice or model_choice not in MODELS:
        return "[WARNING] Please select a valid model", None, None
    
    try:
        # Load model if not already loaded or different model selected
        if CURRENT_MODEL is None or CURRENT_MODEL_PATH != model_choice:
            print(f"[INFO] Loading model: {model_choice}")
            CURRENT_MODEL = load_model(MODELS[model_choice], TOKENIZER.vocab_size, DEVICE)
            CURRENT_MODEL_PATH = model_choice
        
        # Get model type for display
        model_type = type(CURRENT_MODEL).__name__
        
        # Preprocess image
        print(f"[INFO] Preprocessing image with {preprocessing_type}")
        preprocessed = preprocess_image(image, preprocessing_type)
        
        # Convert preprocessed image to PIL for display
        preprocessed_pil = Image.fromarray(preprocessed)
        
        # Predict LaTeX
        print(f"[INFO] Predicting LaTeX")
        latex_output = predict_latex(CURRENT_MODEL, TOKENIZER, preprocessed, DEVICE)
        
        # Clean LaTeX for rendering
        cleaned_latex = clean_latex_output(latex_output)
        
        # Render LaTeX with SymPy (using cleaned version)
        rendered_image = render_latex_with_sympy(cleaned_latex)
        
        # Format output with model info
        model_info = f"**Model:** {model_type}\n"
        model_info += f"**Path:** `{model_choice}`\n\n"
        
        # Show both original and cleaned if different
        if cleaned_latex != latex_output:
            latex_display = model_info + f"**Original LaTeX:**\n\n```latex\n{latex_output}\n```\n\n"
            latex_display += f"**Cleaned LaTeX (for rendering):**\n\n```latex\n{cleaned_latex}\n```"
        else:
            latex_display = model_info + f"**LaTeX Output:**\n\n```latex\n{latex_output}\n```"
        
        return latex_display, rendered_image, preprocessed_pil
    
    except Exception as e:
        error_msg = f"[ERROR] {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        return error_msg, None, None


# ============================================================================
# Build Gradio App
# ============================================================================

def build_app():
    """Build and return Gradio interface."""
    
    # Check if models are available
    if not MODELS:
        print("[ERROR] No models found in checkpoints directory!")
        model_choices = ["No models available"]
    else:
        model_choices = list(MODELS.keys())
    
    with gr.Blocks(title="MER2LATEX - Math Expression Recognition") as app:
        gr.Markdown("""
        # MER2LATEX - Mathematical Expression Recognition
        
        Upload an image of a mathematical formula and convert it to LaTeX!
        
        **Supported Models:**
        - **Model A**: CNN + BiLSTM + CTC (Baseline OCR)
        - **Model B**: ResNet + Attention (Seq2Seq with coverage)
        - **Model C**: ViT + Transformer (Full transformer)
        - **Model D**: TrOCR-style (Pretrained vision encoder)
        
        **Instructions:**
        1. Select a trained model checkpoint
        2. Choose preprocessing type (IM2LATEX for rendered images, CROHME for handwritten)
        3. Upload an image of a mathematical formula
        4. Click "Convert to LaTeX" to see the result
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("### Input")
                
                model_dropdown = gr.Dropdown(
                    choices=model_choices,
                    label="Select Model Checkpoint",
                    value=model_choices[0] if model_choices else None,
                    interactive=True
                )
                
                preprocessing_radio = gr.Radio(
                    choices=["IM2LATEX", "CROHME"],
                    label="Preprocessing Type",
                    value="IM2LATEX",
                    info="IM2LATEX for rendered images, CROHME for handwritten"
                )
                
                image_input = gr.Image(
                    label="Upload Math Formula Image",
                    type="numpy"
                )
                
                submit_btn = gr.Button("Convert to LaTeX", variant="primary")
            
            with gr.Column(scale=1):
                # Output section
                gr.Markdown("### Output")
                
                latex_output = gr.Markdown(label="LaTeX Result")
                
                preprocessed_output = gr.Image(
                    label="Preprocessed Input",
                    type="pil"
                )
                
                rendered_output = gr.Image(
                    label="Rendered LaTeX (SymPy)",
                    type="pil"
                )
        
        # Connect components
        submit_btn.click(
            fn=process_image,
            inputs=[image_input, model_dropdown, preprocessing_radio],
            outputs=[latex_output, rendered_output, preprocessed_output]
        )
    
    return app


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import socket
    
    print("=" * 80)
    print("MER2LATEX Gradio Demo")
    print("=" * 80)
    
    app = build_app()
    
    # Get hostname for display
    hostname = socket.gethostname()
    port = 7860
    
    print(f"\nStarting Gradio server...")
    print(f"Access the app at:")
    print(f"   - Local:   http://localhost:{port}")
    print(f"   - Network: http://{hostname}:{port}")
    print(f"\nUse Ctrl+C to stop the server\n")
    
    app.launch(
        server_name="0.0.0.0",  # Bind to all interfaces
        server_port=port,
        share=False,
        quiet=False
    )
