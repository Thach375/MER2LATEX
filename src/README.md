# MER2LATEX - Source Code Structure

## 📁 Cấu trúc Project

```
src/

├── analysis/               # EDA functions
│   ├── __init__.py
│   ├── compare.py          # Compare im2latex and crohme before and after preprocessed
│   ├── image.py            # Analyze image dimensions, pixel intensity
│   ├── label.py            # Critical for building tokenizer and determining sequence length
│   └── structural.py       # Analyze formula complexity for difficulty assessment
│
├── data_formatting/        # Format CROHME dataset
│   ├── __init__.py
│   └── crohme_formatter.py # Chạy được: python -m src.data_formatting.crohme_formatter
│
├── datasets/               # (Coming soon - dataset classes)
│   
├── engine/                 # (Coming soon - training engine)
│   
├── models/                 # (Coming soon - model architectures)
│   
├── preprocessing/          # Image preprocessing
│   ├── __init__.py
│   ├── batch_process.py    # Process entire dataset and save as .npy files
│   ├── preprocess_pipelines.py  # Full pipeline preprocess im2latex and crohme
│   ├── transforms.py       # Transform functions
│   └── visualize.py        # Visualize preprocessed data
│   
└── utils/                  # Utilities & constants
    ├── __init__.py
    └── constants.py        # Data paths, config

```

## 🚀 Quick Start

### 1. Format CROHME Dataset
```bash
python -m src.data_formatting.crohme_formatter
```

### 2. Run EDA
```bash
python -m src.analysis.image
python -m src.analysis.label
python -m src.analysis.compare
python -m src.analysis.structural
```

### 3. Preprocess Images
```bash
python -m src.preprocessing.batch_process
```

### 4. Run Full Pipeline
```bash
chmod +x pipeline.sh
./pipeline.sh
```

## 📊 Notebooks

**Ngắn gọn, chỉ visualize:**
- `2_EDA.ipynb` - Import hàm từ src, chỉ plot
- `3_Preprocessing.ipynb` - Import hàm từ src, chỉ visualize

## 💡 Nguyên tắc

1. **Notebooks**: CHỈ import + visualize
2. **src/*.py**: Logic + if __name__ == "__main__"
3. **Mọi thứ chạy được từ terminal**
