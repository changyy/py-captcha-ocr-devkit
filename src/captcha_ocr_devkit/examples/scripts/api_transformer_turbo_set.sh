#!/bin/bash
set -euo pipefail

# 🌐 Transformer Turbo (Set Mode) API Script
# Uses the set-based inference handler that ignores character ordering.

NAME="transformer_turbo_set"
HANDLER="${NAME}_ocr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

CONFIG="handlers/${NAME}_handler-config.json"
MODEL_PATH="model/model.${NAME}"
PORT="54321"
HOST="0.0.0.0"

CMD_ARGS=(
  "--handler" "${HANDLER}"
  "--model" "${MODEL_PATH}"
  "--handler-config" "${HANDLER}=${CONFIG}"
  "--port" "${PORT}"
  "--host" "${HOST}"
)

CMD_ARGS+=("$@")

PYTORCH_ENABLE_MPS_FALLBACK=1 captcha-ocr-devkit api "${CMD_ARGS[@]}"
