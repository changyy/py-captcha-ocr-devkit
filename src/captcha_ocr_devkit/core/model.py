"""
OCR 模型定義
支援 4 個字母的 CAPTCHA 圖片辨識
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import string


class CaptchaOCRModel(nn.Module):
    """
    CAPTCHA OCR 模型
    使用 CNN + LSTM 架構來辨識 4 個字母的驗證碼
    """

    def __init__(self,
                 vocab_size: int = None,
                 hidden_dim: int = 128,
                 num_layers: int = 2,
                 max_length: int = 4):
        super(CaptchaOCRModel, self).__init__()

        # 設定字母表
        self.alphabet = string.ascii_lowercase + string.ascii_uppercase + string.digits
        self.vocab_size = vocab_size or len(self.alphabet)
        self.max_length = max_length
        self.hidden_dim = hidden_dim

        # 建立字元到索引的映射
        self.char_to_idx = {char: idx for idx, char in enumerate(self.alphabet)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}

        # CNN 特徵提取器
        self.features = nn.Sequential(
            # 第一層卷積
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 第二層卷積
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 第三層卷積
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # 第四層卷積
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        # 自適應平均池化
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 8))

        # LSTM 序列處理
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 輸出層
        self.classifier = nn.Linear(hidden_dim * 2, self.vocab_size)

        # Dropout
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向傳播

        Args:
            x: 輸入圖片張量 (batch_size, 3, height, width)

        Returns:
            輸出預測 (batch_size, max_length, vocab_size)
        """
        batch_size = x.size(0)

        # CNN 特徵提取
        features = self.features(x)  # (batch_size, 256, h, w)

        # 自適應池化
        features = self.adaptive_pool(features)  # (batch_size, 256, 1, 8)

        # 重塑為序列格式
        features = features.squeeze(2).permute(0, 2, 1)  # (batch_size, 8, 256)

        # LSTM 處理
        lstm_out, _ = self.lstm(features)  # (batch_size, 8, hidden_dim * 2)

        # Dropout
        lstm_out = self.dropout(lstm_out)

        # 分類預測
        output = self.classifier(lstm_out)  # (batch_size, 8, vocab_size)

        # 只取前 max_length 個位置
        output = output[:, :self.max_length, :]

        return output

    def predict(self, x: torch.Tensor) -> List[str]:
        """
        預測圖片中的文字

        Args:
            x: 輸入圖片張量

        Returns:
            預測的文字列表
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            predictions = torch.argmax(output, dim=-1)

            results = []
            for pred in predictions:
                text = ''.join([self.idx_to_char[idx.item()] for idx in pred])
                results.append(text)

            return results

    def decode_predictions(self, predictions: torch.Tensor) -> List[str]:
        """
        解碼預測結果

        Args:
            predictions: 預測張量 (batch_size, max_length)

        Returns:
            解碼後的文字列表
        """
        results = []
        for pred in predictions:
            text = ''.join([self.idx_to_char[idx.item()] for idx in pred])
            results.append(text)
        return results

    def encode_labels(self, labels: List[str]) -> torch.Tensor:
        """
        編碼標籤

        Args:
            labels: 標籤文字列表

        Returns:
            編碼後的張量 (batch_size, max_length)
        """
        batch_size = len(labels)
        encoded = torch.zeros(batch_size, self.max_length, dtype=torch.long)

        for i, label in enumerate(labels):
            for j, char in enumerate(label[:self.max_length]):
                if char in self.char_to_idx:
                    encoded[i, j] = self.char_to_idx[char]

        return encoded

    def get_model_info(self) -> Dict:
        """
        獲取模型資訊

        Returns:
            模型資訊字典
        """
        return {
            'vocab_size': self.vocab_size,
            'max_length': self.max_length,
            'hidden_dim': self.hidden_dim,
            'alphabet': self.alphabet,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class CaptchaLoss(nn.Module):
    """
    CAPTCHA 專用損失函數
    """

    def __init__(self, ignore_index: int = -1):
        super(CaptchaLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        計算損失

        Args:
            predictions: 預測結果 (batch_size, max_length, vocab_size)
            targets: 目標標籤 (batch_size, max_length)

        Returns:
            損失值
        """
        batch_size, max_length, vocab_size = predictions.shape

        # 重塑張量
        predictions = predictions.view(-1, vocab_size)
        targets = targets.view(-1)

        # 計算損失
        loss = self.criterion(predictions, targets)

        return loss


def create_model(config: Optional[Dict] = None) -> CaptchaOCRModel:
    """
    創建模型實例

    Args:
        config: 模型配置參數

    Returns:
        模型實例
    """
    if config is None:
        config = {}

    model = CaptchaOCRModel(
        vocab_size=config.get('vocab_size'),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        max_length=config.get('max_length', 4)
    )

    return model