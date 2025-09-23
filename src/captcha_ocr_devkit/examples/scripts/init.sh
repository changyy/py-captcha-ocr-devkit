#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ ! -d "./venv" ]; then
  python3 -m venv venv
fi

if [ -f "venv/bin/activate" ]; then
  source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
  source venv/Scripts/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

captcha-ocr-devkit init --force

if [ -f "handlers/transformer_handler-requirements.txt" ]; then
  pip install -r handlers/transformer_handler-requirements.txt
fi
if [ -f "handlers/cnn_handler-requirements.txt" ]; then
  pip install -r handlers/cnn_handler-requirements.txt
fi
if [ -f "handlers/crnn_handler-requirements.txt" ]; then
  pip install -r handlers/crnn_handler-requirements.txt
fi
