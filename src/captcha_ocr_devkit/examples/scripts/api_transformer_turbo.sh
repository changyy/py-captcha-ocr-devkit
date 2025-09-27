#!/bin/bash
set -euo pipefail

# 🚀 Transformer Turbo API Server Script
# Enhanced transformer with 4 layers, 8 attention heads, 384 dimensions
# Target: 96%+ accuracy API service

NAME="transformer_turbo"
HANDLER="${NAME}_ocr"
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

echo "🌐 Starting Transformer Turbo API Server..."
echo "🏗️  Enhanced Architecture: 384d × 4L × 8H"
echo "🎯 Target: 96%+ accuracy"
echo "📁 Model path: ${MODEL_PATH}"
echo "🔗 Access at: http://localhost:8000"
echo ""

# Parse command line arguments for API server
CLI_ARGS=()
while [[ $# -gt 0 ]]; do
  CLI_ARGS+=("$1")
  shift
done

CMD_ARGS=(
  "--model" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
)

# Safely append CLI_ARGS if not empty
if [ ${#CLI_ARGS[@]} -gt 0 ]; then
  CMD_ARGS+=("${CLI_ARGS[@]}")
fi

# Enable MPS fallback for CTC loss
PYTORCH_ENABLE_MPS_FALLBACK=1 captcha-ocr-devkit api "${CMD_ARGS[@]}"
