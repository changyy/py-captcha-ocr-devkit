"""
工具函數
處理圖片載入、預處理、標籤解析等功能
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Dict, Optional, Union
import pickle
import json
from pathlib import Path
import logging

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_label_from_filename(filename: str) -> str:
    """
    從檔名解析標籤
    範例：abcd_001.png -> "abcd"

    Args:
        filename: 檔案名稱

    Returns:
        解析出的標籤
    """
    # 移除副檔名，以底線分割，取第一部分
    base_name = os.path.splitext(filename)[0]
    return base_name.split('_')[0]


def load_image(image_path: str, target_size: Tuple[int, int] = (128, 64)) -> np.ndarray:
    """
    載入並預處理圖片

    Args:
        image_path: 圖片路徑
        target_size: 目標尺寸 (width, height)

    Returns:
        預處理後的圖片陣列
    """
    try:
        # 使用 OpenCV 載入圖片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"無法載入圖片: {image_path}")

        # 轉換為 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 調整大小
        image = cv2.resize(image, target_size)

        return image

    except Exception as e:
        logger.error(f"載入圖片失敗 {image_path}: {e}")
        raise


def load_image_pil(image_path: str, target_size: Tuple[int, int] = (128, 64)) -> Image.Image:
    """
    使用 PIL 載入圖片

    Args:
        image_path: 圖片路徑
        target_size: 目標尺寸 (width, height)

    Returns:
        PIL 圖片物件
    """
    try:
        image = Image.open(image_path).convert('RGB')
        image = image.resize(target_size)
        return image

    except Exception as e:
        logger.error(f"載入圖片失敗 {image_path}: {e}")
        raise


def preprocess_image(image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
    """
    預處理圖片為模型輸入格式

    Args:
        image: 輸入圖片

    Returns:
        預處理後的張量 (3, height, width)
    """
    if isinstance(image, Image.Image):
        image = np.array(image)

    # 正規化到 [0, 1]
    image = image.astype(np.float32) / 255.0

    # 轉換為張量並調整維度順序
    image_tensor = torch.from_numpy(image).permute(2, 0, 1)

    return image_tensor


def load_dataset(data_dir: str, target_size: Tuple[int, int] = (128, 64)) -> Tuple[List[torch.Tensor], List[str]]:
    """
    載入資料集

    Args:
        data_dir: 資料目錄
        target_size: 目標圖片尺寸

    Returns:
        (圖片列表, 標籤列表)
    """
    images = []
    labels = []

    data_path = Path(data_dir)
    if not data_path.exists():
        raise ValueError(f"資料目錄不存在: {data_dir}")

    # 支援的圖片格式
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}

    for image_file in data_path.iterdir():
        if image_file.suffix.lower() in image_extensions:
            try:
                # 載入圖片
                image = load_image(str(image_file), target_size)
                image_tensor = preprocess_image(image)
                images.append(image_tensor)

                # 解析標籤
                label = parse_label_from_filename(image_file.name)
                labels.append(label)

            except Exception as e:
                logger.warning(f"跳過圖片 {image_file}: {e}")

    logger.info(f"載入了 {len(images)} 張圖片")
    return images, labels


def save_model(model: torch.nn.Module, model_path: str, model_info: Optional[Dict] = None):
    """
    保存模型

    Args:
        model: 要保存的模型
        model_path: 保存路徑
        model_info: 額外的模型資訊
    """
    try:
        model_data = {
            'model_state_dict': model.state_dict(),
            'model_info': model_info or {},
            'model_class': model.__class__.__name__
        }

        torch.save(model_data, model_path)
        logger.info(f"模型已保存到: {model_path}")

    except Exception as e:
        logger.error(f"保存模型失敗: {e}")
        raise


def load_model(model_class, model_path: str, device: str = 'cpu') -> torch.nn.Module:
    """
    載入模型

    Args:
        model_class: 模型類別
        model_path: 模型路徑
        device: 設備

    Returns:
        載入的模型
    """
    try:
        model_data = torch.load(model_path, map_location=device)

        # 創建模型實例
        model_info = model_data.get('model_info', {})
        model = model_class(**model_info.get('config', {}))

        # 載入權重
        model.load_state_dict(model_data['model_state_dict'])
        model.to(device)
        model.eval()

        logger.info(f"模型已從 {model_path} 載入")
        return model

    except Exception as e:
        logger.error(f"載入模型失敗: {e}")
        raise


def calculate_accuracy(predictions: List[str], targets: List[str]) -> Dict[str, float]:
    """
    計算準確率

    Args:
        predictions: 預測結果
        targets: 真實標籤

    Returns:
        準確率統計
    """
    if len(predictions) != len(targets):
        raise ValueError("預測結果和目標標籤數量不匹配")

    total = len(predictions)
    exact_matches = sum(1 for pred, target in zip(predictions, targets) if pred == target)

    # 計算字元級準確率
    char_correct = 0
    char_total = 0

    for pred, target in zip(predictions, targets):
        min_len = min(len(pred), len(target))
        char_correct += sum(1 for i in range(min_len) if pred[i] == target[i])
        char_total += max(len(pred), len(target))

    return {
        'exact_accuracy': exact_matches / total if total > 0 else 0.0,
        'character_accuracy': char_correct / char_total if char_total > 0 else 0.0,
        'total_samples': total,
        'exact_matches': exact_matches
    }


def create_confusion_matrix(predictions: List[str], targets: List[str]) -> Dict:
    """
    創建混淆矩陣

    Args:
        predictions: 預測結果
        targets: 真實標籤

    Returns:
        混淆矩陣統計
    """
    from collections import defaultdict

    confusion = defaultdict(lambda: defaultdict(int))

    for pred, target in zip(predictions, targets):
        confusion[target][pred] += 1

    return dict(confusion)


def ensure_dir(dir_path: str):
    """
    確保目錄存在

    Args:
        dir_path: 目錄路徑
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def save_results(results: Dict, output_path: str):
    """
    保存結果到 JSON 檔案

    Args:
        results: 結果字典
        output_path: 輸出路徑
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"結果已保存到: {output_path}")

    except Exception as e:
        logger.error(f"保存結果失敗: {e}")
        raise


def load_config(config_path: str) -> Dict:
    """
    載入配置檔案

    Args:
        config_path: 配置檔案路徑

    Returns:
        配置字典
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config

    except Exception as e:
        logger.error(f"載入配置失敗: {e}")
        raise


def get_device() -> str:
    """
    獲取可用的設備

    Returns:
        設備名稱 ('cuda' 或 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def set_seed(seed: int = 42):
    """
    設定隨機種子

    Args:
        seed: 隨機種子
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class AverageMeter:
    """
    平均值計算器
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count