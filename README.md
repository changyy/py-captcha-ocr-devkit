# captcha-ocr-devkit

[![PyPI version](https://img.shields.io/pypi/v/captcha-ocr-devkit.svg)](https://pypi.org/project/captcha-ocr-devkit)
[![PyPI Downloads](https://static.pepy.tech/badge/captcha-ocr-devkit)](https://pepy.tech/projects/captcha-ocr-devkit)

`captcha-ocr-devkit` 是一套跨平台的 CAPTCHA OCR 開發工具箱，專注於「四字元小寫英文」驗證碼範例。框架提供完整的插件化 Handler 架構、內建 demo 與 transformer 範例，可初始化 handler 專案、訓練與評估模型、啟動 FastAPI 服務，並支援 JSON 與 multipart API 呼叫。

## 安裝 Installation

```bash
pip install captcha-ocr-devkit
# 依需求安裝額外功能
pip install "captcha-ocr-devkit[pillow]"
pip install "captcha-ocr-devkit[pytorch]"
pip install "captcha-ocr-devkit[dev]"
```
> PyTorch builders 會依作業系統與硬體差異，請參考官方指引安裝對應版本。

## 快速上手 Quick Start

```bash
# 建立專案骨架 (複製 demo + transformer handlers)
captcha-ocr-devkit init

# 查看 CLI 使用說明
captcha-ocr-devkit --help
```

### 主要指令 CLI Reference

| Command | 說明 |
| --- | --- |
| `captcha-ocr-devkit init` | 複製 `demo` 與 `transformer` handlers，支援 `--handler-dir` 指定自訂模板 |
| `captcha-ocr-devkit train` | 依指定 handler 執行模型訓練 (如 `transformer_train`) |
| `captcha-ocr-devkit evaluate` | 使用 handler 進行模型評估 (如 `transformer_evaluate`) |
| `captcha-ocr-devkit api` | 啟動 FastAPI 服務 (如 `transformer_ocr`) |
| `captcha-ocr-devkit create-handler` | 產生全新的 handler 骨架 |

別名 `captcha-ocr-helper` 等同於上述 CLI。

## Handler 概觀

- **DemoHandler**：展示用範例，透過 fake OCR 回傳固定/隨機結果，設計目的是示範 handler 架構、流程與 metadata。適合複製模版來擴充自己的 handler。
- **TransformerHandlers**：實務可用的一組 handler (`transformer_preprocess`, `transformer_train`, `transformer_evaluate`, `transformer_ocr`)。提供真實的資料前處理、訓練、推論與 API 整合，處理過程會回報版本資訊、損失與驗證指標，API 回傳包含 `image_size` 與 per-character confidence。
- **🚀 TransformerTurboHandlers**：突破性增強版 handler 系列，專為小數據集環境設計。透過架構擴容 (384d×4L×8H) 與先進訓練策略，達成 **96.91%** SOTA 準確率表現，為小數據集 OCR 的技術巔峰。
- **CNNHandlers**：卷積神經網路實作，包含基礎版與優化版。
- **UltraCNNHandlers**：極致優化的 CNN 架構 (ResNet+CBAM+FPN)，達成 94.17% 準確率，展現 CNN 架構的極限潛能。
- **CRNNHandlers**：結合 CNN 與 RNN 的混合架構實作。

## Transformer 實務流程範例

以下示範在 macOS (Python 3.12.11) 上建立環境、訓練並啟動 API：

```bash
sw_vers
ProductName:		macOS
ProductVersion:		26.0
BuildVersion:		25A354

python3 -V
Python 3.12.10

python3 -m venv venv
source venv/bin/activate
pip install captcha-ocr-devkit
cp -r /path/data/ data/
captcha-ocr-devkit init
pip install -r handlers/transformer_handler-requirements.txt
PYTORCH_ENABLE_MPS_FALLBACK=1 captcha-ocr-devkit train \
  --input ./data \
  --handler transformer_train \
  --output model \
  --epochs 250 --batch-size 32 --learning-rate 0.000125
captcha-ocr-devkit evaluate \
  --target ./data \
  --model model \
  --handler transformer_evaluate
captcha-ocr-devkit api \
  --handler transformer_ocr \
  --model model
```

訓練過程會持續 flush log 顯示 core/handler 版本與每個 epoch 的 loss、val_acc、val_cer；API 啟動後可透過 GET `/api/v1/ocr` 檢查服務狀態。

## API 使用範例

```bash
# GET 健康檢查
curl 'http://127.0.0.1:54321/api/v1/ocr'

# POST (JSON + Base64)
curl 'http://127.0.0.1:54321/api/v1/ocr' \
  -H 'Content-Type: application/json' \
  --data '{"image": "<BASE64_STRING>", "format": "png"}'

# POST (Multipart)
curl -X POST 'http://127.0.0.1:54321/api/v1/ocr' \
  -F 'file=@captcha.png'
```
回傳的 `details` 會附上原始尺寸、處理後尺寸與 per-character confidences。

## 專案結構 Project Layout

```
py-captcha-ocr-devkit/
├── handlers/                       # 使用者自訂 handlers (init 後生成)
├── src/captcha_ocr_devkit/
│   ├── core/                       # pipeline、registry、base handlers
│   ├── api/                        # FastAPI routes 與 schemas
│   ├── cli/                        # Typer CLI
│   └── examples/handlers/          # demo + transformer 範例
├── tests/                          # pytest suites
├── docs/
├── main.py
├── requirements.txt
└── setup.py
```

## 模型效能對比

專案內建多種深度學習架構實作，以下為在相同小數據集 (518 樣本) 的效能對比：

| 模型架構 | 整體準確率 | 字符準確率 | 參數量 | 特色 |
|---------|-----------|-----------|--------|------|
| Transformer Turbo | 96.91% | 99.23% | 6.2M | SOTA 新紀錄 |
| Transformer (基線) | 94.59% | 98.12% | 3.2M | 基礎注意力機制 |
| Ultra CNN | 94.17% | 95.83% | ~8M | CNN 架構極限 |
| Optimized CNN | 71.26% | 85.47% | ~5M | 優化後的 CNN |
| CRNN | [待補充] | [待補充] | ~4M | CNN + RNN 混合 |
| Basic CNN | 0% | [失效] | ~3M | 基礎實作 |

### Transformer Turbo 
- **歷史最高準確率**: 96.91% (小數據集環境下的驚人表現)
- **近乎完美字符識別**: 99.23% 字符準確率 (僅 0.77% 誤差)
- **高效推理**: 1.18s 評估 518 樣本
- **穩定訓練**: 44.6 分鐘達到 SOTA 性能

### 技術創新亮點
1. **架構擴容**: d_model 384, 4層, 8頭注意力機制
2. **Pre-Layer Normalization**: 穩定深層訓練
3. **多重正則化**: Dropout + Weight Decay + Label Smoothing
4. **先進調度**: Cosine Annealing + Gradient Clipping + Warmup

## 開發指南 Development

```bash
python -m venv venv
source venv/bin/activate
pip install -e .[dev]
captcha-ocr-devkit init
pytest -v --cov=src/captcha_ocr_devkit
```
更新 handler 後記得重新執行 `captcha-ocr-devkit init` 以同步最新範例資產。

## 授權 License

MIT License 
