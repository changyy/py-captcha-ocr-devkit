"""
訓練邏輯
負責模型訓練和驗證流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import logging
from pathlib import Path
import time
from tqdm import tqdm

from .model import CaptchaOCRModel, CaptchaLoss, create_model
from .utils import (
    load_dataset, preprocess_image, save_model, calculate_accuracy,
    AverageMeter, get_device, set_seed, ensure_dir
)

logger = logging.getLogger(__name__)


class CaptchaDataset(Dataset):
    """
    CAPTCHA 資料集類
    """

    def __init__(self, images: List[torch.Tensor], labels: List[str], model: CaptchaOCRModel):
        self.images = images
        self.labels = labels
        self.model = model

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # 編碼標籤
        encoded_label = self.model.encode_labels([label])[0]

        return image, encoded_label, label


class CaptchaTrainer:
    """
    CAPTCHA OCR 訓練器
    """

    def __init__(self,
                 model: CaptchaOCRModel,
                 device: str = None,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-4):
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)

        # 設定優化器和損失函數
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = CaptchaLoss()

        # 學習率調度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # 訓練歷史
        self.train_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

    def prepare_data(self,
                     data_dir: str,
                     batch_size: int = 32,
                     val_split: float = 0.2,
                     shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        準備訓練和驗證資料

        Args:
            data_dir: 資料目錄
            batch_size: 批次大小
            val_split: 驗證集比例
            shuffle: 是否打亂資料

        Returns:
            (訓練資料載入器, 驗證資料載入器)
        """
        logger.info(f"從 {data_dir} 載入資料...")

        # 載入資料集
        images, labels = load_dataset(data_dir)
        dataset = CaptchaDataset(images, labels, self.model)

        # 分割訓練和驗證集
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = random_split(
            dataset, [train_size, val_size]
        )

        # 建立資料載入器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )

        logger.info(f"訓練集: {len(train_dataset)} 樣本")
        logger.info(f"驗證集: {len(val_dataset)} 樣本")

        return train_loader, val_loader

    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        訓練一個 epoch

        Args:
            train_loader: 訓練資料載入器

        Returns:
            訓練指標
        """
        self.model.train()

        loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []

        progress_bar = tqdm(train_loader, desc="Training")

        for batch_idx, (images, labels, label_texts) in enumerate(progress_bar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # 前向傳播
            self.optimizer.zero_grad()
            outputs = self.model(images)

            # 計算損失
            loss = self.criterion(outputs, labels)

            # 反向傳播
            loss.backward()
            self.optimizer.step()

            # 更新指標
            loss_meter.update(loss.item(), images.size(0))

            # 收集預測結果
            predictions = torch.argmax(outputs, dim=-1)
            pred_texts = self.model.decode_predictions(predictions)
            all_predictions.extend(pred_texts)
            all_targets.extend(label_texts)

            # 更新進度條
            progress_bar.set_postfix({
                'Loss': f'{loss_meter.avg:.4f}'
            })

        # 計算準確率
        accuracy_stats = calculate_accuracy(all_predictions, all_targets)

        return {
            'loss': loss_meter.avg,
            'exact_accuracy': accuracy_stats['exact_accuracy'],
            'char_accuracy': accuracy_stats['character_accuracy']
        }

    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        驗證一個 epoch

        Args:
            val_loader: 驗證資料載入器

        Returns:
            驗證指標
        """
        self.model.eval()

        loss_meter = AverageMeter()
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating")

            for images, labels, label_texts in progress_bar:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # 前向傳播
                outputs = self.model(images)

                # 計算損失
                loss = self.criterion(outputs, labels)
                loss_meter.update(loss.item(), images.size(0))

                # 收集預測結果
                predictions = torch.argmax(outputs, dim=-1)
                pred_texts = self.model.decode_predictions(predictions)
                all_predictions.extend(pred_texts)
                all_targets.extend(label_texts)

                # 更新進度條
                progress_bar.set_postfix({
                    'Loss': f'{loss_meter.avg:.4f}'
                })

        # 計算準確率
        accuracy_stats = calculate_accuracy(all_predictions, all_targets)

        return {
            'loss': loss_meter.avg,
            'exact_accuracy': accuracy_stats['exact_accuracy'],
            'char_accuracy': accuracy_stats['character_accuracy']
        }

    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              epochs: int = 50,
              save_best: bool = True,
              model_dir: str = "./models",
              early_stopping_patience: int = 10) -> Dict:
        """
        執行完整訓練流程

        Args:
            train_loader: 訓練資料載入器
            val_loader: 驗證資料載入器
            epochs: 訓練輪數
            save_best: 是否保存最佳模型
            model_dir: 模型保存目錄
            early_stopping_patience: 早停耐心值

        Returns:
            訓練歷史
        """
        logger.info("開始訓練...")
        logger.info(f"設備: {self.device}")
        logger.info(f"模型參數數量: {self.model.get_model_info()['total_params']}")

        # 確保模型目錄存在
        ensure_dir(model_dir)

        best_val_accuracy = 0.0
        patience_counter = 0
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start_time = time.time()

            # 訓練
            train_metrics = self.train_epoch(train_loader)

            # 驗證
            val_metrics = self.validate_epoch(val_loader)

            # 更新學習率
            self.scheduler.step(val_metrics['loss'])

            # 記錄歷史
            self.train_history['train_loss'].append(train_metrics['loss'])
            self.train_history['train_accuracy'].append(train_metrics['exact_accuracy'])
            self.train_history['val_loss'].append(val_metrics['loss'])
            self.train_history['val_accuracy'].append(val_metrics['exact_accuracy'])

            # 計算 epoch 時間
            epoch_time = time.time() - epoch_start_time

            # 輸出訓練狀態
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['exact_accuracy']:.4f}, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['exact_accuracy']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

            # 保存最佳模型
            if save_best and val_metrics['exact_accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['exact_accuracy']
                patience_counter = 0

                model_path = Path(model_dir) / "best_model.pth"
                save_model(
                    self.model,
                    str(model_path),
                    {
                        'config': {
                            'vocab_size': self.model.vocab_size,
                            'hidden_dim': self.model.hidden_dim,
                            'max_length': self.model.max_length
                        },
                        'best_accuracy': best_val_accuracy,
                        'epoch': epoch + 1
                    }
                )
                logger.info(f"保存最佳模型: {model_path}")

            else:
                patience_counter += 1

            # 早停檢查
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        logger.info(f"訓練完成! 總時間: {total_time:.2f}s")
        logger.info(f"最佳驗證準確率: {best_val_accuracy:.4f}")

        return self.train_history


def train_model(data_dir: str,
                output_dir: str,
                config: Optional[Dict] = None) -> str:
    """
    訓練模型的主要入口函數

    Args:
        data_dir: 訓練資料目錄
        output_dir: 輸出目錄
        config: 訓練配置

    Returns:
        最佳模型路徑
    """
    # 預設配置
    default_config = {
        'model': {
            'hidden_dim': 128,
            'num_layers': 2,
            'max_length': 4
        },
        'training': {
            'batch_size': 32,
            'epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'val_split': 0.2,
            'early_stopping_patience': 10
        },
        'seed': 42
    }

    if config:
        # 合併配置
        for key in default_config:
            if key in config:
                default_config[key].update(config[key])

    config = default_config

    # 設定隨機種子
    set_seed(config['seed'])

    # 建立模型
    model = create_model(config['model'])

    # 建立訓練器
    trainer = CaptchaTrainer(
        model=model,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # 準備資料
    train_loader, val_loader = trainer.prepare_data(
        data_dir=data_dir,
        batch_size=config['training']['batch_size'],
        val_split=config['training']['val_split']
    )

    # 訓練
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config['training']['epochs'],
        model_dir=output_dir,
        early_stopping_patience=config['training']['early_stopping_patience']
    )

    # 保存訓練歷史
    import json
    history_path = Path(output_dir) / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)

    # 返回最佳模型路徑
    best_model_path = Path(output_dir) / "best_model.pth"
    return str(best_model_path)