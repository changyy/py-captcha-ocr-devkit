#!/bin/bash
set -euo pipefail

# ⭐ Ultra-CNN Evaluation Script
# 🎯 Evaluate Ultra-CNN model performance
# 📊 Target: 95% accuracy verification

NAME="ultra_cnn"
HANDLER="cnn_evaluate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="model/model.${NAME}"

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
  echo "❌ Ultra-CNN model not found: ${MODEL_PATH}"
  echo "🔧 Please train the model first: ./scripts/train_ultra_cnn.sh"
  exit 1
fi

echo "📊 Evaluating Ultra-CNN Model Performance..."
echo "🎯 Target: 95% accuracy (challenging Transformer's 93.98%)"
echo "🔥 Architecture: ResNet + CBAM + FPN"
echo "💾 Model: ${MODEL_PATH}"
echo "📂 Test data: ./data"
echo ""

# Evaluation Command
CMD_ARGS=(
  "--target" "./data"
  "--model" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
)
CMD_ARGS+=("$@")

echo "🔍 Executing: captcha-ocr-devkit evaluate ${CMD_ARGS[*]}"
echo ""

time captcha-ocr-devkit evaluate "${CMD_ARGS[@]}"

echo ""
echo "🎉 Ultra-CNN evaluation completed!"
echo "📈 Compare results with previous models:"
echo "  - Basic CNN: 0% accuracy"
echo "  - Optimized CNN: 71.26% accuracy"
echo "  - Ultra-CNN: [Current Results]"
echo "  - Transformer: 93.98% accuracy (target to beat)"