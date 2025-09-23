# crnn_handler 架構說明

`crnn_handler.py` 以 CNN + 雙向 LSTM + CTC (CRNN) 為骨幹，提供針對 4 字元 CAPTCHA 的完整訓練、評估與推論流程。它重用 `ocr_common.py` 的共用模組與 `TransformerPreprocessHandler`，但改以序列化輸出結合 CTC loss，適合處理可變長度或重複字元的情境。

## 架構概覽

- **共用層** (`ocr_common.py`)：提供 `OCRDataset`、`Charset`、`labels_to_targets`、`evaluate_model` 等工具，統一處理影像載入、字集建構與 CER 計算。
- **模型層**：
  - `ConvFeatureExtractor` 先擷取空間特徵並壓縮高度。  
  - `CRNNOCRModel` 將卷積特徵餵入多層雙向 LSTM，再接線性層輸出 `time_step × vocab_size` logits，供 CTC loss / decoding 使用。  
- **Handlers**：
  - `CRNNPreprocessHandler`：沿用 Transformer 的 resize/normalize 設定。  
  - `CRNNTrainHandler`：建構 `OCRDataset` 與 `Charset`，以 AdamW + CTC loss 進行訓練，並將模型超參（hidden_size / num_layers / bidirectional / dropout）存入 checkpoint。  
  - `CRNNEvaluateHandler`：載入 checkpoint、重建模型拓撲、跑評估資料並回傳 accuracy / CER。  
  - `CRNNOCRHandler`：封裝模型載入與 greedy decoding，提供單張圖片的預測服務。

## 資料流

1. **Preprocess**：圖片轉成固定高度張量，寬度自動補齊，確保時間序列長度穩定。  
2. **Train**：
   - 建立字集 (`Charset`) 並轉成 CTC target。  
   - `train_crnn_one_epoch` 執行前向、CTC loss、梯度更新。  
   - 儲存 checkpoint（模型參數 + 超參 + 字集 + 影像尺寸）。
3. **Evaluate / OCR**：載入 checkpoint 後，沿用 greedy decoding 計算整體與逐字表現，或輸出最終 captcha 字串。

## 客製建議

- 想要語言模型式後處理時，可在 `CRNNOCRHandler.predict` 增加 beam search 或字典約束。  
- 若資料集長度不固定，可把 `DEFAULT_IMG_WIDTH` 調大或在資料載入階段記錄原始寬度，搭配動態裁切。  
- 需混合 CNN / Transformer 風格時，可改寫 `CRNNOCRModel`，但保持 CTC 期望的 `time_step × vocab_size` 輸出介面。

