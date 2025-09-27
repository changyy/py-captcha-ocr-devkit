#!/bin/bash
set -euo pipefail

NAME="transformer"
OCR_HANDLER="${NAME}_ocr"
PREPROCESS_HANDLER="${NAME}_preprocess"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="handlers/${NAME}_handler-config.json"
MODEL_PATH="model/model.${NAME}"

CLI_ARGS=()
if [ -f "${CONFIG}" ]; then
  while IFS= read -r token; do
    CLI_ARGS+=("$token")
  done < <(python3 - <<'PY' "${CONFIG}"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
except Exception:
    sys.exit(0)
mapping = [
    ('host', '--host'),
    ('port', '--port'),
    ('workers', '--workers'),
]
for key, flag in mapping:
    if key in data:
        print(flag)
        print(str(data[key]))
if isinstance(data.get('reload'), bool) and data['reload']:
    print('--reload')
PY
  )
fi

# Example:
# captcha-ocr-devkit api --handler transformer_ocr --model model/model.transformer --handler-config transformer_ocr=handlers/transformer_handler-config.json

CMD_ARGS=(
  "--handler" "${OCR_HANDLER}"
  "--handler-config" "${OCR_HANDLER}=${CONFIG}"
  "--preprocess-handler" "${PREPROCESS_HANDLER}"
  "--handler-config" "${PREPROCESS_HANDLER}=${CONFIG}"
  "--model" "${MODEL_PATH}"
)

# Safely append CLI_ARGS if not empty
if [ ${#CLI_ARGS[@]} -gt 0 ]; then
  CMD_ARGS+=("${CLI_ARGS[@]}")
fi

CMD_ARGS+=("$@")

captcha-ocr-devkit api "${CMD_ARGS[@]}"
