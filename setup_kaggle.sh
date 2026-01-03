#!/bin/bash
# Script de setup Kaggle credentials
# 
# CACH SU DUNG:
# 1. Lay API key tu: https://www.kaggle.com/settings -> API -> Create New Token
# 2. Chay script nay:
#    ./setup_kaggle.sh YOUR_USERNAME YOUR_API_KEY
#
# Hoac set environment variables:
#    export KAGGLE_USERNAME="your_username"
#    export KAGGLE_KEY="your_api_key"
#    ./setup_kaggle.sh

echo "=== KAGGLE CREDENTIALS SETUP ==="

# Kiem tra tham so hoac env vars
if [ -n "$1" ] && [ -n "$2" ]; then
    KAGGLE_USERNAME="$1"
    KAGGLE_KEY="$2"
elif [ -z "$KAGGLE_USERNAME" ] || [ -z "$KAGGLE_KEY" ]; then
    echo ""
    echo "Chua co credentials!"
    echo ""
    echo "Cach 1: Truyen tham so"
    echo "  ./setup_kaggle.sh YOUR_USERNAME YOUR_API_KEY"
    echo ""
    echo "Cach 2: Set environment variables"
    echo "  export KAGGLE_USERNAME='your_username'"
    echo "  export KAGGLE_KEY='your_api_key'"
    echo "  ./setup_kaggle.sh"
    echo ""
    echo "Lay API key tai: https://www.kaggle.com/settings"
    echo "  -> Account -> API -> Create New Token"
    exit 1
fi

# Tao thu muc .kaggle
mkdir -p ~/.kaggle

# Tao file kaggle.json
cat > ~/.kaggle/kaggle.json << EOF
{"username":"${KAGGLE_USERNAME}","key":"${KAGGLE_KEY}"}
EOF

# Set quyen
chmod 600 ~/.kaggle/kaggle.json

echo "[OK] Da tao ~/.kaggle/kaggle.json"
echo ""
echo "Test connection:"
kaggle datasets list -s im2latex --max-size 1 2>/dev/null && echo "[OK] Ket noi thanh cong!" || echo "[INFO] Cai kaggle CLI: pip install kaggle"
