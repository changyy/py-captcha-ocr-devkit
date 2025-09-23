# demo_handler 架構說明

`demo_handler.py` 提供一組純隨機的範例 handlers，用來示範 `captcha-ocr-devkit` 的插件化介面。所有元件都集中在同一個檔案中，方便快速理解類別繼承關係與 CLI 整合方式。

## 元件切分

- **DemoPreprocessHandler**：模擬輸入圖片的預處理流程，隨機產生尺寸／步驟結果，協助理解 `HandlerResult` 的結構。
- **DemoTrainHandler**：假裝讀資料並進行訓練，輸出隨機 loss 與模型摘要，示範 `TrainingConfig` 的欄位使用方式。
- **DemoEvaluateHandler**：以亂數產生推論結果，展示如何回填 `EvaluationResult` 與額外的比對資訊。
- **DemoOCRHandler**：完全隨機回傳一組字串與中繼資料，說明 `predict()` / `load_model()` 需要回傳或拋錯的格式。

## 架構重點

1. **最小依賴**：不需額外套件，讓新手能在乾淨環境直接執行 CLI 指令。  
2. **完整生命週期**：四種類型的 handler 全部實作，對應 CLI 的 preprocess/train/evaluate/ocr 指令。  
3. **清楚範本**：每個 handler 的 `get_info()`、`get_supported_formats()`、`HandlerResult` 都提供可複製的骨架。

## 延伸建議

- 以 demo handler 為起點，逐步替換隨機邏輯為實際的資料處理與模型呼叫。  
- 若拆分成多檔案，可依照 handler 類型建立子模組，並更新 `__all__` 以利自動發現。

