# captcha-ocr-devkit

[![PyPI version](https://img.shields.io/pypi/v/captcha-ocr-devkit.svg)](https://pypi.org/project/captcha-ocr-devkit)
[![PyPI Downloads](https://static.pepy.tech/badge/captcha-ocr-devkit)](https://pepy.tech/projects/captcha-ocr-devkit)

`captcha-ocr-devkit` 是一套跨平台的 CAPTCHA OCR 開發工具箱，專注於「四字元小寫英文」驗證碼範例。它提供完整的插件化 Handler 架構，內建 demo 與 transformer 範例，可初始化 handler 專案、訓練與評估模型、啟動 FastAPI 服務，並支援 JSON/multipart API 呼叫。

## Installation

### 1. Install Package
```bash
pip install captcha-ocr-devkit
```

### 2. Optional Extras
```bash
# Pillow / PyTorch / Dev tools
pip install "captcha-ocr-devkit[pillow]"
pip install "captcha-ocr-devkit[pytorch]"
pip install "captcha-ocr-devkit[dev]"
```

> PyTorch builders differ by OS/硬體—請依官方建議安裝對應版本。

## Quick Start

```bash
# 建立專案結構 (複製 demo + transformer handlers)
captcha-ocr-devkit init

# 查看 CLI 幫助
captcha-ocr-devkit --help
```

### Train Transformer Example
```bash
captcha-ocr-devkit train \
  --input ./data \
  --output ./models/transformer.pt \
  --handler transformer_train \
  --epochs 250 --batch-size 32 --learning-rate 0.000125
```
訓練期間會輸出 core/handler 版本與每個 epoch 的 loss / val_acc / val_cer。

### Evaluate
```bash
captcha-ocr-devkit evaluate \
  --target ./test_data \
  --model ./models/transformer.pt \
  --handler transformer_evaluate
```

### Launch API Service
```bash
captcha-ocr-devkit api \
  --handler transformer_ocr \
  --model ./models/transformer.pt \
  --host 0.0.0.0 --port 54321
```
API 會自動搭配 `transformer_preprocess`，GET `/api/v1/ocr` 可做健康檢查。

## API Usage

### GET Health-like Response
```bash
curl 'http://127.0.0.1:54321/api/v1/ocr'
```

### POST – JSON (Base64)
```bash
curl 'http://127.0.0.1:54321/api/v1/ocr' \
  -H 'Content-Type: application/json' \
  --data '{"image": "<BASE64_STRING>", "format": "png"}'
```

### POST – Multipart
```bash
curl -X POST 'http://127.0.0.1:54321/api/v1/ocr' \
  -F 'file=@captcha.png'
```
回傳的 `details` 會包含原始/處理後尺寸與 per-character confidences。

## CLI Reference

| Command | Description |
| --- | --- |
| `captcha-ocr-devkit init` | 複製 `demo` 與 `transformer` handlers；可用 `--handler-dir` 複製自訂 handler |
| `captcha-ocr-devkit train` | 根據 handler 進行訓練 (ex: `transformer_train`) |
| `captcha-ocr-devkit evaluate` | 評估模型 (ex: `transformer_evaluate`) |
| `captcha-ocr-devkit api` | 啟動 FastAPI 服務 (ex: `transformer_ocr`) |
| `captcha-ocr-devkit create-handler` | 產生 handler 骨架 |

同時提供別名 `captcha-ocr-helper` 指向相同 CLI。

## Project Layout
```
py-captcha-ocr-devkit/
├── handlers/                       # 使用者自訂 handlers (init 後生成)
├── src/captcha_ocr_devkit/
│   ├── core/                       # pipeline、registry、base handlers
│   ├── api/                        # FastAPI 服務與 schemas
│   ├── cli/                        # Typer CLI
│   └── examples/handlers/          # demo + transformer handlers
├── tests/                          # pytest suites
├── docs/
├── README.md
├── requirements.txt
└── setup.py
```

## Transformer Highlights
- `transformer_preprocess`, `transformer_train`, `transformer_evaluate`, `transformer_ocr` handlers 以 `HANDLER_ID` 註冊，可同時兼容例外來源。
- 訓練期間強制 flush log，節錄 core/handler version、loss 與 validation 指標。
- API 回傳加上 `image_size` 與 per-character confidence。
- JSON POST 支援 `image` 或 `image_base64` 欄位。

## Development
```
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
pytest tests/unit
```
使用 `captcha-ocr-devkit init` 同步最新範例 handlers，避免版本不一致。

## License
MIT License © changyy

