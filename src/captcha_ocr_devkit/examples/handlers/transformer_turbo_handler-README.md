# transformer_turbo_handler 架構說明

`transformer_turbo_handler.py` 是 Transformer OCR 的增強版實作，專為小數據集環境下的高準確率 OCR 任務設計。透過架構擴容、先進優化技術與穩定訓練策略，達到了 **96.91%** 的 SOTA 準確率表現。

## 🚀 核心創新

### 1. 架構大幅增強
- **d_model**: 256 → **384** (+50% 模型維度)
- **layers**: 2 → **4** (雙倍編碼層深度)
- **attention heads**: 4 → **8** (雙倍多頭注意力)
- **feedforward**: 1024 → **1536** (保持 4:1 比例)
- **參數總量**: ~6.2M (高效平衡)

### 2. 先進架構特性
- **Pre-Layer Normalization**: 替代 Post-LN，提升深層訓練穩定性
- **GELU 激活函數**: 替代 ReLU，提供更平滑的梯度特性
- **Enhanced Positional Encoding**: 改良位置編碼，支持更長序列
- **分層 Dropout**: 更精細的正則化策略 (0.1-0.15)

### 3. 優化訓練策略
- **Cosine Annealing**: 餘弦退火學習率調度，避免局部最優
- **Gradient Clipping**: 梯度裁剪 (max_norm=1.0)，穩定深層訓練
- **Weight Decay**: L2 正則化 (0.01)，防止過擬合
- **Label Smoothing**: 標籤平滑 (0.1)，提升泛化能力
- **Warmup Steps**: 學習率預熱 (1000 steps)，穩健啟動

## 🏗️ 主要組成

### 核心模型
- **TurboOCRModel**: 增強版 Transformer 架構
  - Enhanced CNN Feature Extractor: 更深的特徵提取
  - Multi-layer Transformer Encoder: 4層×8頭注意力機制
  - Pre-LN + GELU: 穩定的深層架構

### Handler 系列
- **TransformerTurboPreprocessHandler**: 增強預處理 (64×192 默認尺寸)
- **TransformerTurboTrainHandler**: 先進訓練循環與優化策略
- **TransformerTurboEvaluateHandler**: 高精度評估與指標計算
- **TransformerTurboOCRHandler**: 高性能推理接口

## 📊 性能表現

### SOTA 成果 (2025-09-26)
- **整體準確率**: **96.91%** (502/518 正確)
- **字符準確率**: **99.23%** (近乎完美)
- **評估速度**: 1.18s (518 樣本)
- **訓練時間**: 44.6 分鐘 (300 epochs)

### 對比分析
| 模型 | 整體準確率 | 字符準確率 | 參數量 |
|------|-----------|-----------|--------|
| Transformer | 94.59% | 98.12% | ~3.2M |
| **Transformer Turbo** | **96.91%** | **99.23%** | **6.2M** |
| 提升 | **+2.32%** | **+1.11%** | **+93%** |

## 🛠️ 使用方式

### 快速開始
```bash
# 🚀 訓練增強版 Transformer
./scripts/train_transformer_turbo.sh

# 📊 評估模型性能
./scripts/evaluate_transformer_turbo.sh

# 🌐 啟動 API 服務
./scripts/api_transformer_turbo.sh
```

### 配置調優
- 調整 `transformer_turbo_handler-config.json` 中的超參數
- 默認配置已針對小數據集環境優化
- 支持設備自動偵測 (CPU/CUDA/MPS)

## 💡 技術洞察

### 小數據集優化策略
1. **模型容量平衡**: 足夠的複雜度但避免過擬合
2. **多重正則化**: Dropout + Weight Decay + Label Smoothing
3. **穩定訓練**: Pre-LN + 梯度裁剪 + 學習率預熱
4. **精細調度**: Cosine Annealing 避免局部最優

### 與基線 Transformer 差異
- **架構深度**: 2層 → 4層，增強表達能力
- **注意力機制**: 4頭 → 8頭，更細緻的特徵捕獲
- **訓練穩定性**: Post-LN → Pre-LN，深層訓練更穩定
- **優化策略**: 基礎 AdamW → 多重正則化組合

## 📦 依賴需求

使用 `transformer_turbo_handler-requirements.txt`：
- PyTorch >= 1.9.0
- Torchvision
- NumPy
- Pillow
- 其他基礎依賴

## 🌟 應用場景

- **小數據集 OCR**: 在有限訓練數據下追求極致準確率
- **生產環境部署**: 高準確率 + 快速推理的平衡方案
- **模型研究**: Transformer 架構優化的最佳實踐範例
- **基準對比**: 小數據集深度學習的 SOTA 參考

**🏆 Transformer Turbo 代表了小數據集 OCR 任務的技術巔峰！**