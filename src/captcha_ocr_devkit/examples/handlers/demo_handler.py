"""
Demo Handler 範例

純 random 實作，完全不依賴第三方庫，適合：
- 快速理解 Handler 概念
- 教學和演示用途
- 系統功能測試
- 新手入門範例

這個範例展示了如何用最簡單的方式實作所有四種 Handler
"""

import json
import time
import random
import string
from pathlib import Path
from typing import Any, List, Union

from captcha_ocr_devkit.core.handlers.base import (
    BasePreprocessHandler,
    BaseTrainHandler,
    BaseEvaluateHandler,
    BaseOCRHandler,
    HandlerResult,
    TrainingConfig,
    EvaluationResult
)


class DemoPreprocessHandler(BasePreprocessHandler):
    HANDLER_ID = "demo_preprocess"
    """
    Demo 圖片預處理 Handler

    用 random 模擬圖片處理效果
    """

    def process(self, image_data: Union[bytes, str, Path]) -> HandlerResult:
        """模擬圖片預處理"""
        start_time = time.time()

        # 模擬處理時間
        processing_delay = random.uniform(0.01, 0.05)
        time.sleep(processing_delay)

        # 模擬處理效果
        original_size = (random.randint(100, 200), random.randint(50, 100))
        processed_size = (128, 64)  # 標準化尺寸

        # 模擬處理步驟
        processing_steps = []
        if random.random() > 0.5:
            processing_steps.append("noise_reduction")
        if random.random() > 0.5:
            processing_steps.append("contrast_enhancement")
        if random.random() > 0.3:
            processing_steps.append("resized")

        processing_time = time.time() - start_time

        return HandlerResult(
            success=True,
            data=f"processed_image_data_{random.randint(1000, 9999)}",
            metadata={
                "original_size": f"{original_size[0]}x{original_size[1]}",
                "processed_size": f"{processed_size[0]}x{processed_size[1]}",
                "processing_steps": processing_steps,
                "processing_time": processing_time,
                "demo_mode": True
            }
        )

    def get_supported_formats(self) -> List[str]:
        return [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"]

    def get_info(self):
        return {
            "name": self.name,
            "handler_id": self.HANDLER_ID,
            "version": "1.0.0",
            "description": "Demo 圖片預處理，純 random 模擬",
            "dependencies": [],
            "demo_mode": True
        }


class DemoTrainHandler(BaseTrainHandler):
    HANDLER_ID = "demo_train"
    """
    Demo 訓練 Handler

    用 random 模擬訓練過程和結果
    """

    def train(self, config: TrainingConfig) -> HandlerResult:
        """模擬訓練過程"""
        print(f"🎯 Demo 訓練開始！")
        print(f"📂 輸入目錄: {config.input_dir}")
        print(f"💾 輸出路徑: {config.output_path}")
        print(f"🎲 Demo 模式: 純 random 模擬")

        start_time = time.time()

        # 檢查輸入目錄
        if not config.input_dir.exists():
            return HandlerResult(
                success=False,
                error=f"輸入目錄不存在: {config.input_dir}"
            )

        # 模擬尋找圖片
        image_files = list(config.input_dir.glob("*.png")) + list(config.input_dir.glob("*.jpg"))
        total_images = len(image_files) if image_files else random.randint(50, 500)

        print(f"🖼️  模擬處理 {total_images} 張圖片")

        # 解析標籤（如果有實際檔案）
        if image_files:
            labels = self.parse_labels_from_filenames(image_files)
            unique_labels = set(labels)
        else:
            # 模擬標籤
            unique_labels = set()
            for _ in range(random.randint(10, 50)):
                label_length = random.randint(3, 6)
                label = ''.join(random.choices(string.ascii_lowercase + string.digits, k=label_length))
                unique_labels.add(label)

        print(f"🏷️  發現 {len(unique_labels)} 個不同標籤")

        # 模擬訓練進度
        total_epochs = min(config.epochs, 10)  # 最多顯示 10 個 epochs
        for epoch in range(total_epochs):
            # 模擬訓練時間
            time.sleep(random.uniform(0.05, 0.15))

            # 模擬 loss 下降
            loss = max(0.01, 1.0 - (epoch / total_epochs) + random.uniform(-0.1, 0.1))

            if epoch % max(1, total_epochs // 5) == 0:
                print(f"  Epoch {epoch + 1}/{config.epochs} - Loss: {loss:.4f} (模擬)")

        # 創建模擬模型資料
        model_data = {
            "model_type": "demo_random",
            "demo_mode": True,
            "training_config": {
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "validation_split": config.validation_split
            },
            "dataset_info": {
                "total_images": total_images,
                "unique_labels": len(unique_labels),
                "sample_labels": list(unique_labels)[:10],  # 保存前 10 個作為示例
                "alphabet": string.ascii_lowercase + string.digits
            },
            "model_performance": {
                "final_loss": random.uniform(0.01, 0.1),
                "validation_accuracy": random.uniform(0.85, 0.98),
                "training_accuracy": random.uniform(0.90, 0.99)
            },
            "training_time": time.time() - start_time,
            "timestamp": time.time()
        }

        # 保存模型
        success = self.save_model(model_data, config.output_path)
        if not success:
            return HandlerResult(
                success=False,
                error="模型保存失敗"
            )

        training_time = time.time() - start_time
        print(f"✅ Demo 訓練完成! 耗時: {training_time:.2f}s")
        print(f"🎲 模擬準確率: {model_data['model_performance']['validation_accuracy']:.4f}")

        return HandlerResult(
            success=True,
            data={
                "model_path": str(config.output_path),
                "total_images": total_images,
                "unique_labels": len(unique_labels),
                "demo_mode": True
            },
            metadata={
                "training_time": training_time,
                "epochs_completed": config.epochs,
                "dataset_size": total_images,
                "model_performance": model_data["model_performance"]
            }
        )

    def save_model(self, model_data: Any, output_path: Path) -> bool:
        """保存模擬模型到 JSON 檔案"""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 如果沒有副檔名，加上 .json
            if not output_path.suffix:
                output_path = output_path.with_suffix('.json')

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(model_data, f, indent=2, ensure_ascii=False, default=str)

            return True
        except Exception as e:
            print(f"保存模型失敗: {e}")
            return False

    def load_model(self, model_path: Path) -> Any:
        """載入模擬模型"""
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"載入模型失敗: {e}")
            return None

    def get_info(self):
        return {
            "name": self.name,
            "handler_id": self.HANDLER_ID,
            "version": "1.0.0",
            "description": "Demo 訓練，純 random 模擬訓練過程",
            "dependencies": [],
            "demo_mode": True
        }


class DemoEvaluateHandler(BaseEvaluateHandler):
    HANDLER_ID = "demo_evaluate"
    """
    Demo 評估 Handler

    用 random 模擬評估過程和結果
    """

    def evaluate(self, model_path: Path, test_data_path: Path) -> HandlerResult:
        """模擬評估過程"""
        print(f"📊 Demo 評估開始！")
        print(f"🤖 模型: {model_path}")
        print(f"📂 測試資料: {test_data_path}")
        print(f"🎲 Demo 模式: 純 random 模擬")

        start_time = time.time()

        # 檢查檔案存在
        if not model_path.exists():
            return HandlerResult(
                success=False,
                error=f"模型檔案不存在: {model_path}"
            )

        if not test_data_path.exists():
            return HandlerResult(
                success=False,
                error=f"測試資料目錄不存在: {test_data_path}"
            )

        # 載入模型
        model_data = self.load_model(model_path)
        if model_data is None:
            return HandlerResult(
                success=False,
                error="模型載入失敗"
            )

        # 檢查測試圖片
        test_images = list(test_data_path.glob("*.png")) + list(test_data_path.glob("*.jpg"))
        total_samples = len(test_images) if test_images else random.randint(20, 200)

        print(f"🖼️  模擬評估 {total_samples} 張圖片")

        # 解析標籤
        if test_images:
            labels = self.parse_labels_from_filenames(test_images)
        else:
            # 模擬標籤
            alphabet = model_data.get("dataset_info", {}).get("alphabet", string.ascii_lowercase + string.digits)
            labels = []
            for _ in range(total_samples):
                label_length = random.randint(3, 6)
                label = ''.join(random.choices(alphabet, k=label_length))
                labels.append(label)

        # 模擬預測結果
        predictions = []
        base_accuracy = model_data.get("model_performance", {}).get("validation_accuracy", 0.85)

        for label in labels:
            # 模擬評估時間
            if len(predictions) % 10 == 0:
                time.sleep(0.01)

            # 根據模型性能模擬預測準確率
            if random.random() < base_accuracy:
                predictions.append(label)  # 正確預測
            else:
                # 隨機錯誤預測
                alphabet = string.ascii_lowercase + string.digits
                wrong_pred = ''.join(random.choices(alphabet, k=len(label)))
                predictions.append(wrong_pred)

        # 計算指標
        eval_result = self.calculate_metrics(predictions, labels)

        evaluation_time = time.time() - start_time
        print(f"✅ Demo 評估完成! 耗時: {evaluation_time:.2f}s")
        print(f"🎯 模擬準確率: {eval_result.accuracy:.4f}")
        print(f"🔤 字元準確率: {eval_result.character_accuracy:.4f}")

        return HandlerResult(
            success=True,
            data=eval_result,
            metadata={
                "evaluation_time": evaluation_time,
                "model_info": model_data.get("model_type", "unknown"),
                "demo_mode": True
            }
        )

    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> EvaluationResult:
        """計算評估指標"""
        total = len(predictions)
        if total == 0:
            return EvaluationResult(
                accuracy=0.0,
                total_samples=0,
                correct_predictions=0,
                character_accuracy=0.0
            )

        # 完整匹配準確率
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
        accuracy = correct / total

        # 字元級準確率
        total_chars = 0
        correct_chars = 0

        for pred, true in zip(predictions, ground_truth):
            min_len = min(len(pred), len(true))
            max_len = max(len(pred), len(true))
            total_chars += max_len

            # 計算正確的字元數
            for i in range(min_len):
                if pred[i] == true[i]:
                    correct_chars += 1

        char_accuracy = correct_chars / total_chars if total_chars > 0 else 0.0

        return EvaluationResult(
            accuracy=accuracy,
            total_samples=total,
            correct_predictions=correct,
            character_accuracy=char_accuracy
        )

    def load_model(self, model_path: Path) -> Any:
        """載入模型"""
        try:
            with open(model_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"載入模型失敗: {e}")
            return None

    def parse_labels_from_filenames(self, image_paths: List[Path]) -> List[str]:
        """從檔名解析標籤"""
        labels = []
        for path in image_paths:
            filename = path.stem  # 移除副檔名
            label = filename.split('_')[0]  # 以底線分割，取第一部分
            labels.append(label)
        return labels

    def get_info(self):
        return {
            "name": self.name,
            "handler_id": self.HANDLER_ID,
            "version": "1.0.0",
            "description": "Demo 評估，純 random 模擬評估過程",
            "dependencies": [],
            "demo_mode": True
        }


class DemoOCRHandler(BaseOCRHandler):
    HANDLER_ID = "demo_ocr"
    """
    Demo OCR Handler

    用 random 模擬 OCR 識別
    """

    def __init__(self, name: str, config=None):
        super().__init__(name, config)
        self.model_data = None
        self.alphabet = string.ascii_lowercase + string.digits

    def predict(self, processed_image: Any) -> HandlerResult:
        """模擬 OCR 預測"""
        start_time = time.time()

        # 模擬處理時間
        processing_delay = random.uniform(0.02, 0.08)
        time.sleep(processing_delay)

        # 根據載入的模型決定預測策略
        if self.model_data:
            # 使用模型中的字母表
            alphabet = self.model_data.get("dataset_info", {}).get("alphabet", self.alphabet)
            base_accuracy = self.model_data.get("model_performance", {}).get("validation_accuracy", 0.85)

            # 從樣本標籤中隨機選擇一個作為"識別結果"
            sample_labels = self.model_data.get("dataset_info", {}).get("sample_labels", [])
            if sample_labels and random.random() < base_accuracy:
                # 高機率返回訓練過的標籤
                predicted_text = random.choice(sample_labels)
            else:
                # 隨機生成
                text_length = random.randint(3, 6)
                predicted_text = ''.join(random.choices(alphabet, k=text_length))
        else:
            # 沒有模型時純隨機
            text_length = random.randint(3, 6)
            predicted_text = ''.join(random.choices(self.alphabet, k=text_length))
            base_accuracy = 0.5

        # 模擬信心度（基於模型性能）
        confidence = min(0.99, max(0.1, base_accuracy + random.uniform(-0.15, 0.15)))

        processing_time = time.time() - start_time

        return HandlerResult(
            success=True,
            data=predicted_text,
            metadata={
                "confidence": confidence,
                "processing_time": processing_time,
                "model_type": "demo_random",
                "demo_mode": True,
                "alphabet_used": len(self.alphabet),
                "text_length": len(predicted_text)
            }
        )

    def load_model(self, model_path: Path) -> bool:
        """載入 Demo 模型"""
        try:
            print(f"🤖 載入 Demo 模型: {model_path}")

            if not model_path.exists():
                print(f"⚠️  模型檔案不存在: {model_path}")
                return False

            # 載入模型資料
            with open(model_path, 'r', encoding='utf-8') as f:
                self.model_data = json.load(f)

            # 更新字母表
            if "dataset_info" in self.model_data and "alphabet" in self.model_data["dataset_info"]:
                self.alphabet = self.model_data["dataset_info"]["alphabet"]

            print(f"✅ Demo 模型載入成功")
            if self.model_data and "model_performance" in self.model_data:
                performance = self.model_data["model_performance"]
                print(f"📊 模型性能: {performance.get('validation_accuracy', 0):.4f}")

            return True

        except Exception as e:
            print(f"❌ Demo 模型載入失敗: {e}")
            return False

    def get_info(self):
        info = {
            "name": self.name,
            "handler_id": self.HANDLER_ID,
            "version": "1.0.0",
            "description": "Demo OCR，純 random 模擬識別",
            "dependencies": [],
            "demo_mode": True,
            "model_loaded": self.model_data is not None,
            "alphabet_size": len(self.alphabet)
        }

        if self.model_data:
            info["model_info"] = {
                "type": self.model_data.get("model_type", "unknown"),
                "training_time": self.model_data.get("training_time", 0),
                "sample_count": self.model_data.get("dataset_info", {}).get("total_images", 0)
            }

        return info


# 便利函數：檢查 Demo Handler 可用性
def check_demo_handlers():
    """檢查 Demo Handlers 可用性"""
    print("🎯 Demo Handlers 狀態檢查")
    print("✅ demo_preprocess (DemoPreprocessHandler) - 純 Python，無依賴")
    print("✅ demo_train (DemoTrainHandler) - 純 Python，無依賴")
    print("✅ demo_evaluate (DemoEvaluateHandler) - 純 Python，無依賴")
    print("✅ demo_ocr (DemoOCRHandler) - 純 Python，無依賴")
    print("🎲 所有功能都是 random 模擬，適合教學和測試")
    return True


if __name__ == '__main__':
    # 簡單測試
    check_demo_handlers()

    # 測試創建 handlers
    try:
        preprocess = DemoPreprocessHandler('demo_preprocess')
        train = DemoTrainHandler('demo_train')
        evaluate = DemoEvaluateHandler('demo_evaluate')
        ocr = DemoOCRHandler('demo_ocr')

        print("\n🎉 所有 Demo Handlers 創建成功！")
        print(f"📝 Preprocess: {preprocess.get_info()['description']}")
        print(f"🏋️  Train: {train.get_info()['description']}")
        print(f"📊 Evaluate: {evaluate.get_info()['description']}")
        print(f"👁️  OCR: {ocr.get_info()['description']}")

    except Exception as e:
        print(f"❌ 測試失敗: {e}")
