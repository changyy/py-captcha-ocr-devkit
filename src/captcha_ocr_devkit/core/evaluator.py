"""
驗證評估邏輯
負責模型性能評估和分析
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging
from pathlib import Path
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import json

from .model import CaptchaOCRModel, create_model
from .utils import (
    load_dataset, load_model, calculate_accuracy, create_confusion_matrix,
    get_device, save_results, AverageMeter
)
from .trainer import CaptchaDataset

logger = logging.getLogger(__name__)


class CaptchaEvaluator:
    """
    CAPTCHA OCR 評估器
    """

    def __init__(self, model: CaptchaOCRModel, device: str = None):
        self.model = model
        self.device = device or get_device()
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_checkpoint(cls, model_path: str, device: str = None):
        """
        從檢查點載入評估器

        Args:
            model_path: 模型檔案路徑
            device: 設備

        Returns:
            評估器實例
        """
        device = device or get_device()
        model = load_model(CaptchaOCRModel, model_path, device)
        return cls(model, device)

    def evaluate_single_image(self, image_path: str) -> Dict:
        """
        評估單張圖片

        Args:
            image_path: 圖片路徑

        Returns:
            評估結果
        """
        from .utils import load_image, preprocess_image, parse_label_from_filename

        try:
            # 載入和預處理圖片
            image = load_image(image_path)
            image_tensor = preprocess_image(image).unsqueeze(0).to(self.device)

            # 預測
            start_time = time.time()
            with torch.no_grad():
                output = self.model(image_tensor)
                predictions = torch.argmax(output, dim=-1)
                pred_text = self.model.decode_predictions(predictions)[0]
            inference_time = time.time() - start_time

            # 解析真實標籤
            true_label = parse_label_from_filename(Path(image_path).name)

            # 計算信心度
            probs = torch.softmax(output[0], dim=-1)
            confidences = torch.max(probs, dim=-1)[0].cpu().numpy()
            mean_confidence = np.mean(confidences)

            return {
                'image_path': image_path,
                'predicted_text': pred_text,
                'true_label': true_label,
                'is_correct': pred_text == true_label,
                'confidence': float(mean_confidence),
                'inference_time': inference_time,
                'character_confidences': confidences.tolist()
            }

        except Exception as e:
            logger.error(f"評估圖片失敗 {image_path}: {e}")
            return {
                'image_path': image_path,
                'error': str(e)
            }

    def evaluate_dataset(self, data_dir: str, batch_size: int = 32) -> Dict:
        """
        評估整個資料集

        Args:
            data_dir: 資料目錄
            batch_size: 批次大小

        Returns:
            詳細評估報告
        """
        logger.info(f"評估資料集: {data_dir}")

        # 載入資料
        images, labels = load_dataset(data_dir)
        dataset = CaptchaDataset(images, labels, self.model)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        all_predictions = []
        all_targets = []
        all_confidences = []
        inference_times = []
        detailed_results = []

        total_samples = 0
        start_time = time.time()

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc="Evaluating")

            for batch_idx, (batch_images, batch_labels, batch_texts) in enumerate(progress_bar):
                batch_images = batch_images.to(self.device)
                batch_start_time = time.time()

                # 預測
                outputs = self.model(batch_images)
                predictions = torch.argmax(outputs, dim=-1)
                pred_texts = self.model.decode_predictions(predictions)

                batch_inference_time = time.time() - batch_start_time
                inference_times.append(batch_inference_time)

                # 計算信心度
                probs = torch.softmax(outputs, dim=-1)
                batch_confidences = torch.max(probs, dim=-1)[0].cpu().numpy()

                # 收集結果
                all_predictions.extend(pred_texts)
                all_targets.extend(batch_texts)
                all_confidences.extend(batch_confidences.mean(axis=1).tolist())

                # 詳細結果
                for i, (pred, target, conf) in enumerate(zip(pred_texts, batch_texts, batch_confidences)):
                    detailed_results.append({
                        'index': total_samples + i,
                        'predicted': pred,
                        'target': target,
                        'is_correct': pred == target,
                        'confidence': float(conf.mean())
                    })

                total_samples += len(pred_texts)

                # 更新進度條
                current_acc = sum(1 for p, t in zip(all_predictions, all_targets) if p == t) / len(all_predictions)
                progress_bar.set_postfix({'Accuracy': f'{current_acc:.4f}'})

        total_time = time.time() - start_time

        # 計算詳細指標
        accuracy_stats = calculate_accuracy(all_predictions, all_targets)
        confusion_matrix = create_confusion_matrix(all_predictions, all_targets)

        # 字元級分析
        char_analysis = self._analyze_character_errors(all_predictions, all_targets)

        # 信心度分析
        confidence_analysis = self._analyze_confidence(all_confidences, all_predictions, all_targets)

        # 建立評估報告
        evaluation_report = {
            'dataset_info': {
                'total_samples': total_samples,
                'evaluation_time': total_time,
                'avg_inference_time': np.mean(inference_times),
                'images_per_second': total_samples / total_time
            },
            'accuracy_metrics': accuracy_stats,
            'character_analysis': char_analysis,
            'confidence_analysis': confidence_analysis,
            'confusion_matrix': confusion_matrix,
            'detailed_results': detailed_results
        }

        logger.info(f"評估完成! 準確率: {accuracy_stats['exact_accuracy']:.4f}")
        return evaluation_report

    def _analyze_character_errors(self, predictions: List[str], targets: List[str]) -> Dict:
        """
        分析字元級錯誤

        Args:
            predictions: 預測結果
            targets: 真實標籤

        Returns:
            字元級分析結果
        """
        position_errors = defaultdict(int)
        character_errors = defaultdict(int)
        error_patterns = defaultdict(int)

        for pred, target in zip(predictions, targets):
            if pred != target:
                min_len = min(len(pred), len(target))

                # 位置錯誤統計
                for pos in range(min_len):
                    if pred[pos] != target[pos]:
                        position_errors[pos] += 1
                        character_errors[f"{target[pos]}->{pred[pos]}"] += 1
                        error_patterns[f"pos_{pos}_{target[pos]}->{pred[pos]}"] += 1

                # 長度不匹配
                if len(pred) != len(target):
                    error_patterns[f"length_mismatch_{len(target)}->{len(pred)}"] += 1

        return {
            'position_errors': dict(position_errors),
            'character_substitutions': dict(character_errors),
            'error_patterns': dict(sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:20])
        }

    def _analyze_confidence(self, confidences: List[float], predictions: List[str], targets: List[str]) -> Dict:
        """
        分析信心度分佈

        Args:
            confidences: 信心度列表
            predictions: 預測結果
            targets: 真實標籤

        Returns:
            信心度分析結果
        """
        correct_confidences = []
        incorrect_confidences = []

        for conf, pred, target in zip(confidences, predictions, targets):
            if pred == target:
                correct_confidences.append(conf)
            else:
                incorrect_confidences.append(conf)

        return {
            'overall_confidence': {
                'mean': np.mean(confidences),
                'std': np.std(confidences),
                'median': np.median(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'correct_predictions': {
                'count': len(correct_confidences),
                'mean_confidence': np.mean(correct_confidences) if correct_confidences else 0,
                'std_confidence': np.std(correct_confidences) if correct_confidences else 0
            },
            'incorrect_predictions': {
                'count': len(incorrect_confidences),
                'mean_confidence': np.mean(incorrect_confidences) if incorrect_confidences else 0,
                'std_confidence': np.std(incorrect_confidences) if incorrect_confidences else 0
            }
        }

    def generate_report(self, evaluation_results: Dict, output_dir: str):
        """
        生成評估報告

        Args:
            evaluation_results: 評估結果
            output_dir: 輸出目錄
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # 保存 JSON 報告
        json_path = output_path / "evaluation_report.json"
        save_results(evaluation_results, str(json_path))

        # 生成可視化圖表
        self._plot_confusion_matrix(
            evaluation_results['confusion_matrix'],
            str(output_path / "confusion_matrix.png")
        )

        self._plot_confidence_distribution(
            evaluation_results,
            str(output_path / "confidence_distribution.png")
        )

        self._plot_character_errors(
            evaluation_results['character_analysis'],
            str(output_path / "character_errors.png")
        )

        # 生成文字報告
        self._generate_text_report(evaluation_results, str(output_path / "report.txt"))

        logger.info(f"評估報告已保存到: {output_dir}")

    def _plot_confusion_matrix(self, confusion_matrix: Dict, output_path: str):
        """
        繪製混淆矩陣
        """
        try:
            # 獲取所有標籤
            all_labels = set()
            for true_label in confusion_matrix:
                all_labels.add(true_label)
                for pred_label in confusion_matrix[true_label]:
                    all_labels.add(pred_label)

            all_labels = sorted(list(all_labels))

            # 建立矩陣
            matrix = np.zeros((len(all_labels), len(all_labels)))
            for i, true_label in enumerate(all_labels):
                for j, pred_label in enumerate(all_labels):
                    if true_label in confusion_matrix and pred_label in confusion_matrix[true_label]:
                        matrix[i, j] = confusion_matrix[true_label][pred_label]

            # 繪圖
            plt.figure(figsize=(12, 10))
            sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues',
                       xticklabels=all_labels, yticklabels=all_labels)
            plt.title('Confusion Matrix')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"無法生成混淆矩陣圖: {e}")

    def _plot_confidence_distribution(self, evaluation_results: Dict, output_path: str):
        """
        繪製信心度分佈圖
        """
        try:
            detailed_results = evaluation_results['detailed_results']
            correct_confidences = [r['confidence'] for r in detailed_results if r['is_correct']]
            incorrect_confidences = [r['confidence'] for r in detailed_results if not r['is_correct']]

            plt.figure(figsize=(10, 6))
            plt.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
            plt.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            plt.title('Confidence Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            logger.warning(f"無法生成信心度分佈圖: {e}")

    def _plot_character_errors(self, char_analysis: Dict, output_path: str):
        """
        繪製字元錯誤分析圖
        """
        try:
            position_errors = char_analysis['position_errors']

            if position_errors:
                positions = list(position_errors.keys())
                errors = list(position_errors.values())

                plt.figure(figsize=(8, 6))
                plt.bar(positions, errors, color='skyblue')
                plt.xlabel('Character Position')
                plt.ylabel('Number of Errors')
                plt.title('Character Errors by Position')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()

        except Exception as e:
            logger.warning(f"無法生成字元錯誤分析圖: {e}")

    def _generate_text_report(self, evaluation_results: Dict, output_path: str):
        """
        生成文字報告
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("CAPTCHA OCR 評估報告\n")
            f.write("=" * 50 + "\n\n")

            # 基本資訊
            dataset_info = evaluation_results['dataset_info']
            f.write("資料集資訊:\n")
            f.write(f"  總樣本數: {dataset_info['total_samples']}\n")
            f.write(f"  評估時間: {dataset_info['evaluation_time']:.2f}s\n")
            f.write(f"  平均推理時間: {dataset_info['avg_inference_time']:.4f}s\n")
            f.write(f"  處理速度: {dataset_info['images_per_second']:.1f} images/s\n\n")

            # 準確率指標
            accuracy_metrics = evaluation_results['accuracy_metrics']
            f.write("準確率指標:\n")
            f.write(f"  完全匹配準確率: {accuracy_metrics['exact_accuracy']:.4f}\n")
            f.write(f"  字元級準確率: {accuracy_metrics['character_accuracy']:.4f}\n")
            f.write(f"  正確樣本數: {accuracy_metrics['exact_matches']}\n")
            f.write(f"  總樣本數: {accuracy_metrics['total_samples']}\n\n")

            # 信心度分析
            conf_analysis = evaluation_results['confidence_analysis']
            f.write("信心度分析:\n")
            f.write(f"  平均信心度: {conf_analysis['overall_confidence']['mean']:.4f}\n")
            f.write(f"  信心度標準差: {conf_analysis['overall_confidence']['std']:.4f}\n")
            f.write(f"  正確預測平均信心度: {conf_analysis['correct_predictions']['mean_confidence']:.4f}\n")
            f.write(f"  錯誤預測平均信心度: {conf_analysis['incorrect_predictions']['mean_confidence']:.4f}\n\n")

            # 字元錯誤分析
            char_analysis = evaluation_results['character_analysis']
            f.write("字元錯誤分析:\n")
            if char_analysis['position_errors']:
                f.write("  位置錯誤統計:\n")
                for pos, count in char_analysis['position_errors'].items():
                    f.write(f"    位置 {pos}: {count} 個錯誤\n")

            f.write("\n  常見錯誤模式:\n")
            for pattern, count in list(char_analysis['error_patterns'].items())[:10]:
                f.write(f"    {pattern}: {count} 次\n")


def evaluate_model(model_path: str, data_dir: str, output_dir: str = "./evaluation") -> Dict:
    """
    評估模型的主要入口函數

    Args:
        model_path: 模型檔案路徑
        data_dir: 測試資料目錄
        output_dir: 評估結果輸出目錄

    Returns:
        評估結果
    """
    logger.info(f"開始評估模型: {model_path}")

    # 載入評估器
    evaluator = CaptchaEvaluator.from_checkpoint(model_path)

    # 執行評估
    results = evaluator.evaluate_dataset(data_dir)

    # 生成報告
    evaluator.generate_report(results, output_dir)

    return results