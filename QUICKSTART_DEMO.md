# MER2LATEX - Gradio Demo Quick Start

## Run Demo

### Option 1: Direct Launch (Recommended)

```bash
# Quick start
./run_demo.sh

# Or manually
python app/gradio_app.py
```

Access: **http://localhost:7860**

### Option 2: Docker

```bash
# CPU
docker-compose --profile cpu up -d
docker exec -it mer2latex-container python app/gradio_app.py

# GPU
docker-compose --profile gpu up -d
docker exec -it mer2latex-container-gpu python app/gradio_app.py
```

## Usage

1. **Select Model**: Choose from dropdown (Model A/B/C/D)
2. **Preprocessing**: Select IM2LATEX (rendered) or CROHME (handwritten)
3. **Upload Image**: Drop math formula image
4. **Convert**: Click button to get LaTeX output

## Model Types

- **Model A**: CNN + BiLSTM + CTC
- **Model B**: ResNet + Attention (best performance)
- **Model C**: ViT + Transformer
- **Model D**: TrOCR

## Troubleshooting

### No models available
```bash
# Check checkpoint structure
tree checkpoints -L 3
```

### SymPy rendering fails
```bash
pip install sympy matplotlib
```

### Slow inference
- Use GPU if available
- Select smaller model (Model A)
- Check device with: `python -c "import torch; print(torch.cuda.is_available())"`

## Performance

| Component | CPU | GPU |
|-----------|-----|-----|
| Preprocessing | 50-100ms | 50-100ms |
| Inference | 1-2s | 100-300ms |
| Rendering | 200-500ms | 200-500ms |
| **Total** | **~2s** | **~500ms** |

## Tips

- Use high-quality images (>200 DPI)
- IM2LATEX for rendered formulas (white background)
- CROHME for handwritten formulas
- Select `best.pt` checkpoints
