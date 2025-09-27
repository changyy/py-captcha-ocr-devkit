#!/bin/bash
set -euo pipefail

# ⭐ Ultra-CNN Training Script
# 🎯 Target: 95% accuracy with ResNet+CBAM+FPN architecture
# 🔥 Features: Advanced data augmentation + mixed precision + ultra optimizations

NAME="ultra_cnn"
HANDLER="cnn_train"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Ultra-CNN specific configuration
CONFIG="handlers/cnn_handler-config.json"
MODEL_PATH="model/model.${NAME}"

mkdir -p "$(dirname "${MODEL_PATH}")"

# Ensure Ultra configuration exists
if [ ! -f "${CONFIG}" ]; then
  echo "🔧 Creating Ultra-CNN configuration..."
  cat > "${CONFIG}" << 'EOF'
{
  "epochs": 200,
  "batch_size": 32,
  "learning_rate": 0.001,
  "use_optimized": true,
  "use_ultra": true,
  "dropout": 0.2,
  "weight_decay": 0.0001,
  "label_smoothing": 0.1,
  "cosine_annealing": true,
  "early_stopping_patience": 15,
  "device": "cpu"
}
EOF
  echo "✅ Ultra-CNN configuration created!"
fi

# Parse configuration for command line arguments
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

echo "🚀 Starting Ultra-CNN Training (95% accuracy target)..."
echo "🎯 Architecture: ResNet + CBAM + FPN"
echo "🎨 Data Augmentation: UltraOCRDataset enabled"
echo "⚡ Optimizations: Mixed precision + Enhanced AdamW"
echo "💾 Model output: ${MODEL_PATH}"
echo "⚙️ Configuration: ${CONFIG}"
echo ""

# Ultra-CNN Training Command
CMD_ARGS=(
  "--input" "./data"
  "--output" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
  "--handler-config" "${HANDLER}=${CONFIG}"
)
CMD_ARGS+=("${CONFIG_ARGS[@]}")
CMD_ARGS+=("$@")

# Start training with timing
echo "🔥 Executing: captcha-ocr-devkit train ${CMD_ARGS[*]}"
echo "📊 Monitor training progress in logs..."
echo ""

time captcha-ocr-devkit train "${CMD_ARGS[@]}"

echo ""
echo "🎉 Ultra-CNN training completed!"
echo "📈 Check model performance with: ./scripts/evaluate_ultra_cnn.sh"
echo "🌐 Start API service with: ./scripts/api_ultra_cnn.sh"