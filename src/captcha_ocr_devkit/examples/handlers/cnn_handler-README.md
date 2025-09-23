# cnn_handler 架構說明

`cnn_handler.py` 提供以卷積分類器解四字元小寫 CAPTCHA 的參考實作，沿用 `ocr_common.py` 與 `transformer_handler.py` 中的共用工具，強調多頭分類器的訓練與推論流程。

## 主要元件

- **共用工具**：透過 `ocr_common.py` 中的 `OCRDataset`、`TorchHandlerDependencyMixin`、`resolve_device` 等函式，統一處理資料載入與依賴檢查。預處理仍沿用 `TransformerPreprocessHandler` 的 resize/normalize 流程。
- **CNNClassifier**：採三層卷積 + BatchNorm + ReLU + AdaptiveAvgPool，輸出固定長度特徵後接線性層，拆成 `num_characters` 個 head（每個 head 為 26 類小寫字母），以 CrossEntropyLoss 同時訓練多個位置的分類。
- **Handlers**  
  - `CNNPreprocessHandler`：包裝預處理設定與 `get_info()`。  
  - `CNNTrainHandler`：
    - 建構 `OCRDataset` 後過濾不符合長度/字母表的樣本。  
    - `DataLoader` 以 `collate_batch` 堆疊張量。  
    - 以 AdamW + CrossEntropyLoss 訓練並儲存 checkpoint（包含 `alphabet`/`num_characters`）。
  - `CNNEvaluateHandler`：載入模型，逐批推論並計算 captcha-level 與 character-level accuracy。  
  - `CNNOCRHandler`：載入 checkpoint 與字母表，封裝單筆推論。

## 資料流與模型行為

1. **Dataset 篩選**：透過檔名取得標籤（例如 `abcd_001.png`），只保留長度正確且字元落在允許集合的樣本。  
2. **訓練**：CNN 產生 `batch × num_characters × alphabet_size` 的 logits，展平成多分類問題並以 CrossEntropyLoss 最佳化。  
3. **推論**：對每個字元位置取 `argmax` 組合成最終 captcha 字串。  
4. **評估**：計算整體正確率與逐字正確率，並輸出詳細比對紀錄。

## 客製建議

- 可替換 `CNNClassifier` 為更深層或含注意力機制的骨幹，但仍維持多 head 的輸出格式。  
- 若資料集包含大寫或數字，可調整 `DEFAULT_ALPHABET`，並確保 `requirements` 同步更新。  
- 需要額外資料增強時，可在讀取影像後加入隨機裁切、噪音或顏色轉換邏輯。

