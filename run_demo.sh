#!/bin/bash
# Run Gradio Demo App for MER2LATEX

echo "============================================"
echo "Starting MER2LATEX Gradio Demo"
echo "============================================"

# Check if vocab.json exists
if [ ! -f "vocab.json" ]; then
    echo "ERROR: vocab.json not found!"
    echo "Please ensure vocabulary file exists in the root directory."
    exit 1
fi

# Check if checkpoints directory exists
if [ ! -d "checkpoints" ]; then
    echo "ERROR: checkpoints directory not found!"
    echo "Please ensure you have trained models in the checkpoints/ directory."
    exit 1
fi

# Count available models
MODEL_COUNT=$(find checkpoints -name "best.pt" -o -name "latest.pt" | wc -l)
echo "Found $MODEL_COUNT model checkpoint(s)"

if [ $MODEL_COUNT -eq 0 ]; then
    echo "WARNING: No model checkpoints found!"
    echo "Please train a model first or place trained models in checkpoints/"
fi

# Run the app
echo ""
echo "Launching Gradio app..."
echo "Access the demo at: http://localhost:7860"
echo ""

python app/gradio_app.py
