# MER2LaTeX - Math Expression Recognition

Deep Learning Project: Image -> LaTeX -> (Optional) SymPy

## Project Structure

```
/app/
├── src/
│   ├── data/                     # Dataset loading
│   │   └── dataset.py            # MERDataset, get_dataloader
│   ├── models/                   # Model architectures
│   │   ├── components.py         # Encoders, decoders, attention
│   │   └── architectures.py      # Model A/B/C/D definitions
│   ├── training/                 # Training utilities
│   │   ├── trainer.py            # Main Trainer class
│   │   └── callbacks.py          # EarlyStopping, ModelCheckpoint
│   ├── evaluation/               # Evaluation & testing
│   │   ├── metrics.py            # Metrics, error analysis
│   │   └── evaluate.py           # Test script (runnable)
│   ├── preprocessing/            # Image preprocessing
│   │   ├── transforms.py         # Image transforms
│   │   ├── preprocess_pipelines.py
│   │   └── batch_process.py
│   ├── tokenizer/
│   │   └── tokenize.py           # LaTeX tokenizer
│   └── utils/
│       ├── constants.py          # Configuration and paths
│       ├── download_data.py      # Download datasets
│       └── analysis.py           # EDA functions
├── data/
│   ├── IM2LATEX/                 # IM2LATEX raw data (from Kaggle)
│   ├── CROHME/                   # CROHME raw InkML files
│   └── preprocessed/             # Processed outputs
│       ├── crohme/               # CROHME processed images
│       └── im2latex/             # IM2LATEX processed images
├── checkpoints/                  # Model checkpoints
├── logs/                         # Training logs
├── notebooks/                    # Jupyter notebooks for EDA
├── app/                          # Gradio web demo
└── pipeline.sh                   # Full pipeline script
```

## Models

| Model | Architecture | Params | Description |
|-------|-------------|--------|-------------|
| Model A | CNN + BiLSTM + CTC | 13.4M | Baseline OCR |
| Model B | ResNet + Attention | 12.6M | Seq2seq with Bahdanau attention |
| Model C | ViT + Transformer | 26.1M | Full transformer-based |
| Model D | TrOCR-style | 26.1M | Pretrained vision encoder |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Pipeline
```bash
./pipeline.sh
```

This will:
1. Download CROHME and IM2LATEX datasets from Kaggle
2. Process CROHME InkML files -> images
3. Run preprocessing pipeline
4. Train model (default: model_b)

### 3. Train Specific Model
```bash
python -m src.training.trainer --model model_a --epochs 30
python -m src.training.trainer --model model_b --epochs 30
python -m src.training.trainer --model model_c --epochs 30
python -m src.training.trainer --model model_d --epochs 30
```

### 4. Evaluate Trained Model
```bash
# Evaluate on single dataset
python -m src.evaluation.evaluate \
    --checkpoint checkpoints/model_a/.../best.pt \
    --model model_a \
    --dataset im2latex \
    --tag baseline

# Evaluate on both datasets
python -m src.evaluation.evaluate \
    --checkpoint checkpoints/model_a/.../best.pt \
    --model model_a \
    --dataset all \
    --tag baseline

# View comparison table
python -m src.evaluation.evaluate --show-results

# Clear results
python -m src.evaluation.evaluate --clear-results
```

## Training Options

```bash
python -m src.training.trainer \
    --model model_b \
    --dataset im2latex \
    --epochs 30 \
    --batch-size 4 \
    --lr 1e-4 \
    --checkpoint-metric val_loss \
    --patience 5 \
    --no-wandb              # Disable wandb logging
    --resume checkpoint.pt  # Resume from checkpoint
```

## Training Features

- **Wandb Integration**: Automatic experiment tracking
- **Checkpoint Saving**: Best + latest + epoch checkpoints
- **Early Stopping**: Stop when metric stops improving
- **Mixed Precision (AMP)**: Faster training on GPU
- **Gradient Clipping**: Prevent exploding gradients

### Checkpoint Structure
```
checkpoints/
|-- model_b/
|   |-- 20231231_120000/
|       |-- best.pt           # Best checkpoint
|       |-- latest.pt         # Latest checkpoint
```

## Usage in Code

```python
from src.models import create_model
from src.training import train_model
from src.data import get_dataloader, compute_metrics

# Train
results = train_model('model_b', num_epochs=30, batch_size=4)

# Or use Trainer class for more control
from src.training import Trainer
trainer = Trainer(
    model_name='model_b',
    num_epochs=30,
    use_wandb=True,
    early_stopping=True
)
results = trainer.train()
```

## Datasets

| Dataset | Type | Samples | Description |
|---------|------|---------|-------------|
| IM2LATEX | Printed | ~100k | Rendered LaTeX formulas |
| CROHME | Handwritten | ~10k | Handwritten math expressions |

## Evaluation Metrics

- **Exact Match**: Percentage of exact LaTeX matches
- **BLEU Score**: N-gram overlap with reference
- **Edit Distance**: Normalized Levenshtein distance
- **SymPy Equivalence**: Mathematical equivalence check

## Docker

```bash
# Nếu không có GPU:
docker-compose --profile cpu build
docker-compose --profile cpu up -d

# Nếu có GPU:
docker-compose --profile gpu build
docker-compose --profile gpu up -d
# Attach VSCode: Ctrl+Shift+P -> Dev Containers: Attach
```
