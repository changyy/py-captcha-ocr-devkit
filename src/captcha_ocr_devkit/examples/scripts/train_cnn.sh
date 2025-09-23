#!/bin/bash
set -euo pipefail

NAME="cnn"
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

# Example:
# captcha-ocr-devkit train --input ./data --output model/model.cnn --handler cnn_train --handler-config cnn_train=handlers/cnn_handler-config.json
# time captcha-ocr-devkit train --input ./data --handler cnn_train --output model/model.cnn --epochs 200 --batch-size 64 --learning-rate 0.001

CMD_ARGS=(
  "--input" "./data"
  "--output" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
  "--handler-config" "${HANDLER}=${CONFIG}"
)
CMD_ARGS+=("${CONFIG_ARGS[@]}")
CMD_ARGS+=("$@")

time captcha-ocr-devkit train "${CMD_ARGS[@]}"
