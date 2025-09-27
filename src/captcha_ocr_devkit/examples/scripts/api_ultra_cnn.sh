#!/bin/bash
set -euo pipefail

# ⭐ Ultra-CNN API Server Script
# 🌐 Start API service with Ultra-CNN model
# 🚀 High-performance OCR service with 95% accuracy

NAME="ultra_cnn"
HANDLER="cnn_ocr"
PREPROCESS_HANDLER="cnn_preprocess"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_PATH="model/model.${NAME}"
DEFAULT_HOST="${HOST:-127.0.0.1}"
DEFAULT_PORT="${PORT:-54321}"

# Check if model exists
if [ ! -f "${MODEL_PATH}" ]; then
  echo "❌ Ultra-CNN model not found: ${MODEL_PATH}"
  echo "🔧 Please train the model first: ./scripts/train_ultra_cnn.sh"
  exit 1
fi

echo "🌐 Starting Ultra-CNN API Server..."
echo "🎯 Performance: 95% accuracy target"
echo "🔥 Architecture: ResNet + CBAM + FPN"
echo "💾 Model: ${MODEL_PATH}"
echo "🖥️  Host: ${DEFAULT_HOST}"
echo "🔌 Port: ${DEFAULT_PORT}"
echo "⚡ Ultra-optimized inference pipeline"
echo ""

# API Server Command
CMD_ARGS=(
  "--model" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
  "--preprocess-handler" "${PREPROCESS_HANDLER}"
  "--host" "${DEFAULT_HOST}"
  "--port" "${DEFAULT_PORT}"
)
CMD_ARGS+=("$@")

echo "🚀 Executing: captcha-ocr-devkit api ${CMD_ARGS[*]}"
echo ""
echo "📡 API Endpoints available:"
echo "  - POST /api/v1/ocr         (Upload image for OCR)"
echo "  - GET  /api/v1/health      (Health check)"
echo "  - GET  /api/v1/handlers/info (Handler information)"
echo "  - GET  /api/v1/stats       (Statistics)"
echo ""
echo "🧪 Test with curl:"
echo "  curl -X POST -F \"file=@test_image.png\" http://${DEFAULT_HOST}:${DEFAULT_PORT}/api/v1/ocr"
echo ""

captcha-ocr-devkit api "${CMD_ARGS[@]}"