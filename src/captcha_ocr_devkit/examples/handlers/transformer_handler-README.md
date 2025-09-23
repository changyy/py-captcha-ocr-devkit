# transformer_handler 架構說明

`transformer_handler.py` 提供以 Transformer 為核心的 CAPTCHA OCR 實作，並搭配 `ocr_common.py` 中的共用模組來減少重複程式碼。整體由四個 handler 組成，可覆蓋預處理、訓練、評估與推論流程。

## 主要組成

- **共用基礎層 (`ocr_common.py`)**  
  - `TorchHandlerDependencyMixin`：統一處理 requirements 檔案、缺漏提示與安裝指引。  
  - `OCRDataset`：封裝圖片載入、灰階化、等比例縮放與張量化邏輯。  
  - `Charset` / `labels_to_targets` / `greedy_decode_batch`：處理 CTC 編碼與解碼流程。  
  - `resolve_device`、`set_seed` 等工具：提供裝置偵測與隨機種子設定。

- **模型層**  
  - `ConvFeatureExtractor`（定義於共用層）：擷取空間特徵並轉成序列。  
  - `PositionalEncoding` + `nn.TransformerEncoder`：建構時序關係。  
  - `OCRModel`：將卷積特徵轉為 Transformer 輸出並接線性分類層。

- **Handlers**  
  - `TransformerPreprocessHandler`：沿用 `OCRDataset` 的 resize/normalize 流程，輸出張量與中繼資訊。  
  - `TransformerTrainHandler`：負責資料集切分、DataLoader 建立、CTC loss 訓練與 checkpoint 儲存。  
  - `TransformerEvaluateHandler`：載入 checkpoint、跑測試集並計算 accuracy / CER。  
  - `TransformerOCRHandler`：在推論時載入模型與字集，提供單筆預測結果。

## 資料流

1. **Preprocess**：將任意尺寸的 captcha 圖片轉成固定大小的灰階張量。  
2. **Train**：載入資料集、建構字集 (`Charset`)、經由 `OCRModel` + CTC 訓練。  
3. **Evaluate**：使用與訓練相同的 dataset 工具，比對標籤並計算 CER。  
4. **OCR**：封裝模型載入與 `predict()`，可配合 API / CLI 快速上線。

## 延伸建議

- 依情境調整 `OCRModel` 的 Transformer 深度、注意力頭數或卷積架構。  
- 若需客製資料增強，可擴充 `OCRDataset` 或建立獨立的 Preprocess handler。  
- 建議搭配 `transformer_handler-requirements.txt` 安裝 PyTorch、Torchvision、NumPy、Pillow，確保共用層功能可用。

