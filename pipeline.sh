#!/bin/bash

##############################################
# MER2LATEX - Full Pipeline
##############################################

set -e  # Exit on error

echo "=================================================="
echo "MER2LATEX PIPELINE"
echo "=================================================="

# Parse arguments
SKIP_DATA=false
SKIP_TRAIN=false
MODEL="model_b"
DATASET="im2latex"
EPOCHS=30
BATCH_SIZE=4

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA=true
            shift
            ;;
        --skip-train)
            SKIP_TRAIN=true
            shift
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ "$SKIP_DATA" = false ]; then
    # Step 1: Download Dataset
    echo ""
    echo "[STEP 1] Download Dataset"
    echo "--------------------------------------------------"
    
    # Download se tu kiem tra credentials va exit neu khong co
    python -m src.utils.download_data
    
    # Kiem tra ca 2 datasets da download thanh cong
    if [ ! -f "data/IM2LATEX/im2latex_train.csv" ]; then
        echo ""
        echo "[ERROR] IM2LATEX dataset khong ton tai!"
        exit 1
    fi

    # Step 2: EDA (optional, can be commented out)
    echo ""
    echo "[STEP 2] Exploratory Data Analysis"
    echo "--------------------------------------------------"
    python -m src.utils.analysis 

    # Step 3: Preprocessing
    echo ""
    echo "[STEP 3] Preprocess Images"
    echo "--------------------------------------------------"
    python -m src.preprocessing.preprocess_pipelines
    python -m src.preprocessing.batch_process

    # Step 4: Build Tokenizer
    echo ""
    echo "[STEP 4] Build Tokenizer"
    echo "--------------------------------------------------"
    python -m src.tokenizer.tokenize
fi

if [ "$SKIP_TRAIN" = false ]; then
    # Step 5: Training
    echo ""
    echo "[STEP 5] Training Model: $MODEL"
    echo "--------------------------------------------------"
    python -m src.training.trainer \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE"
fi

echo ""
echo "=================================================="
echo "[DONE] PIPELINE COMPLETED!"
echo "=================================================="
echo ""
echo "Usage examples:"
echo "  ./pipeline.sh                           # Full pipeline with defaults"
echo "  ./pipeline.sh --skip-data               # Skip data preparation"
echo "  ./pipeline.sh --model model_a           # Train Model A (CTC)"
echo "  ./pipeline.sh --model model_b           # Train Model B (Attention)"
echo "  ./pipeline.sh --model model_c           # Train Model C (Transformer)"
echo "  ./pipeline.sh --model model_d           # Train Model D (TrOCR)"
echo "  ./pipeline.sh --epochs 50 --batch-size 8 # Custom training settings"
echo ""
