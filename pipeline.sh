#!/bin/bash

##############################################
# MER2LATEX - Full Pipeline
##############################################

set -e  # Exit on error

echo "=================================================="
echo "🚀 MER2LATEX PIPELINE"
echo "=================================================="

# Step 1: Download Dataset
echo ""
echo "📋 STEP 1: Download Dataset"
echo "--------------------------------------------------"
python 


# Step 2: Preprocessing
echo ""
echo "🔧 STEP 2: Preprocess Images"
echo "--------------------------------------------------"
python -m src.preprocessing.transforms
python -m src.preprocessing.preprocess_pipelines
python -m src.preprocessing.batch_process

# Step 3: ...

echo ""
echo "=================================================="
echo "✅ PIPELINE COMPLETED!"
echo "=================================================="
