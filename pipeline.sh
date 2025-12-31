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
python -m src.data.download_dataset

# Step 2: EDA
echo ""
echo "🔍 STEP 2: Exploratory Data Analysis"
echo "--------------------------------------------------"
python -m src.utils.analysis 

# Step 3: Preprocessing
echo ""
echo "🔧 STEP 3: Preprocess Images"
echo "--------------------------------------------------"
python -m src.preprocessing.preprocess_pipelines
python -m src.preprocessing.batch_process

# Step 4: ...

echo ""
echo "=================================================="
echo "✅ PIPELINE COMPLETED!"
echo "=================================================="
