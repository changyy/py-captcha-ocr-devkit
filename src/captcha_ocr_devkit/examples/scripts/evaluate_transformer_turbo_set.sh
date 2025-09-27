#!/bin/bash
set -euo pipefail

# 🧪 Transformer Turbo (Set Mode) Evaluation Script
# Currently reuses the transformer_turbo evaluation handler.
# Swap HANDLER once a dedicated set evaluator is implemented.

NAME="transformer_turbo_set"
BASE_NAME="transformer_turbo"
HANDLER="${BASE_NAME}_evaluate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="handlers/${NAME}_handler-config.json"
MODEL_PATH="model/model.${NAME}"

if [ ! -f "${MODEL_PATH}" ]; then
  echo "❌ Model not found: ${MODEL_PATH}"
  echo "🔧 Train first with: ./scripts/train_${NAME}.sh"
  exit 1
fi

if [ ! -d "./data" ]; then
  echo "❌ Default evaluation data directory ./data not found."
  echo "   Use --target /path/to/images to override."
  exit 1
fi

CMD_ARGS=(
  "--target" "./data"
  "--model" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
)

if [ -f "${CONFIG}" ]; then
  CMD_ARGS+=("--handler-config" "${HANDLER}=${CONFIG}")
fi

CMD_ARGS+=("$@")

PYTORCH_ENABLE_MPS_FALLBACK=1 time captcha-ocr-devkit evaluate "${CMD_ARGS[@]}"

echo ""
echo "✅ Transformer Turbo (Set Mode) evaluation completed."
echo "➡️  Serve API with: ./scripts/api_transformer_turbo_set.sh"
