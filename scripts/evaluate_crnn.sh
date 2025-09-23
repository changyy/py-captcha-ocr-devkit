#!/bin/bash
set -euo pipefail

NAME="crnn"
HANDLER="${NAME}_evaluate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="handlers/${NAME}_handler-config.json"
MODEL_PATH="model/model.${NAME}"

# Example:
# captcha-ocr-devkit evaluate --target ./data --model model/model.crnn --handler crnn_evaluate --handler-config crnn_evaluate=handlers/crnn_handler-config.json

CMD_ARGS=(
  "--target" "./data"
  "--model" "${MODEL_PATH}"
  "--handler" "${HANDLER}"
  "--handler-config" "${HANDLER}=${CONFIG}"
)
CMD_ARGS+=("$@")

time captcha-ocr-devkit evaluate "${CMD_ARGS[@]}"
