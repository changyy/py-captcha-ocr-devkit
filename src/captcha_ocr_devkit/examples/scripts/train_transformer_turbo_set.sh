#!/bin/bash
set -euo pipefail

# 🧪 Transformer Turbo (Set Mode) Training Script
# NOTE: The set-based training loop is still under development.
#       This script reuses the standard transformer_turbo training handler for now.
#       Replace HANDLER once the dedicated set trainer is ready.

NAME="transformer_turbo_set"
BASE_NAME="transformer_turbo"
HANDLER="${BASE_NAME}_train"
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

echo "🧪 Starting Transformer Turbo (Set Mode) Training..."
echo "  • Current workflow reuses transformer_turbo_train handler"
echo "  • Update HANDLER once the multi-label set trainer is available"
echo "💾 Model output: ${MODEL_PATH}"
echo "⚙️ Configuration: ${CONFIG}"
echo ""

CMD_ARGS=(
  "--input" "./data"
  "--output" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
  "--handler-config" "${HANDLER}=${CONFIG}"
)

if [ ${#CONFIG_ARGS[@]} -gt 0 ]; then
  CMD_ARGS+=("${CONFIG_ARGS[@]}")
fi

CMD_ARGS+=("$@")

PYTORCH_ENABLE_MPS_FALLBACK=1 time captcha-ocr-devkit train "${CMD_ARGS[@]}"

echo ""
echo "✅ Training run finished."
echo "➡️  Evaluate with: ./scripts/evaluate_transformer_turbo_set.sh"
echo "➡️  Serve API with: ./scripts/api_transformer_turbo_set.sh"
