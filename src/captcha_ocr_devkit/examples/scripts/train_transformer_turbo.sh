#!/bin/bash
set -euo pipefail

# 🚀 Transformer Turbo Training Script
# Enhanced transformer with 4 layers, 8 attention heads, 384 dimensions
# Target: 96%+ accuracy with advanced training strategies

NAME="transformer_turbo"
HANDLER="${NAME}_train"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="handlers/${NAME}_handler-config.json"
MODEL_PATH="model/model.${NAME}"

mkdir -p "$(dirname "${MODEL_PATH}")"

CONFIG_ARGS=()
if [ -f "${CONFIG}" ]; then
  while IFS= read -r token; do
    CONFIG_ARGS+=("$token")
  done < <(python3 - <<'PY' "${CONFIG}"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
except Exception:
    sys.exit(0)
mapping = [
    ('epochs', '--epochs'),
    ('batch_size', '--batch-size'),
    ('learning_rate', '--learning-rate'),
    ('validation_split', '--validation-split'),
    ('device', '--device'),
    ('seed', '--seed'),
]
for key, flag in mapping:
    if key in data:
        print(flag)
        print(str(data[key]))
PY
  )
fi

echo "🚀 Starting Transformer Turbo Training..."
echo "🏗️  Enhanced Architecture: 384d × 4L × 8H"
echo "⚡ Advanced Features: Cosine annealing, gradient clipping, pre-layer norm"
echo "🎯 Target: 96%+ accuracy"
echo "💾 Model output: ${MODEL_PATH}"
echo "⚙️ Configuration: ${CONFIG}"
echo ""

CMD_ARGS=(
  "--input" "./data"
  "--output" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
  "--handler-config" "${HANDLER}=${CONFIG}"
)

# Safely append CONFIG_ARGS if not empty
if [ ${#CONFIG_ARGS[@]} -gt 0 ]; then
  CMD_ARGS+=("${CONFIG_ARGS[@]}")
fi

CMD_ARGS+=("$@")

# Enable MPS fallback for CTC loss + timing
PYTORCH_ENABLE_MPS_FALLBACK=1 time captcha-ocr-devkit train "${CMD_ARGS[@]}"

echo ""
echo "🎉 Transformer Turbo training completed!"
echo "📈 Check model performance with: ./scripts/evaluate_transformer_turbo.sh"
echo "🌐 Start API service with: ./scripts/api_transformer_turbo.sh"