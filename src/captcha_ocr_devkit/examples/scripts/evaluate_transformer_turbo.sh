#!/bin/bash
set -euo pipefail

# 🚀 Transformer Turbo Evaluation Script
# Enhanced transformer with 4 layers, 8 attention heads, 384 dimensions
# Target: 96%+ accuracy evaluation

NAME="transformer_turbo"
HANDLER="${NAME}_evaluate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="model/model.${NAME}"

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
  echo "❌ Model not found: ${MODEL_PATH}"
  echo "🔧 Train first with: ./scripts/train_transformer_turbo.sh"
  exit 1
fi

echo "📊 Starting Transformer Turbo Evaluation..."
echo "🏗️  Enhanced Architecture: 384d × 4L × 8H"
echo "🎯 Target: 96%+ accuracy"
echo "📁 Model path: ${MODEL_PATH}"
echo ""

CMD_ARGS=(
  "--target" "./data"
  "--model" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
)

CMD_ARGS+=("$@")

# Enable MPS fallback for CTC loss + timing
PYTORCH_ENABLE_MPS_FALLBACK=1 time captcha-ocr-devkit evaluate "${CMD_ARGS[@]}"

echo ""
echo "✅ Transformer Turbo evaluation completed!"
echo "🏆 Compare with Transformer baseline: 94.59%"