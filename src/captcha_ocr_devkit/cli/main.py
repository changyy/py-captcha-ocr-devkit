"""
新版 CLI 介面
基於 Handler 架構的插件化 CAPTCHA OCR 工具
"""

import click
import json
import logging
import os
import re
import shutil
import sys
import uvicorn
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from captcha_ocr_devkit import __version__ as CORE_VERSION

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option(version=CORE_VERSION)
def cli():
    """
    CAPTCHA OCR 開發助手 v2.0

    插件化架構，支援自定義 handlers

    \b
      基本使用流程:
        1. captcha-ocr-devkit init  # 初始化 handlers
        2. captcha-ocr-devkit train --input image_dir --output model
        3. captcha-ocr-devkit evaluate --target image_dir --model model_path
        4. captcha-ocr-devkit api --model model_path
    """
    pass


@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='訓練資料目錄')
@click.option('--output', '-o', required=True, type=click.Path(),
              help='模型輸出路徑')
@click.option('--handler', help='指定 train handler 名稱')
@click.option('--epochs', default=100, help='訓練輪數')
@click.option('--batch-size', default=32, help='批次大小')
@click.option('--learning-rate', default=0.001, help='學習率')
@click.option('--validation-split', default=0.2, help='驗證集比例')
@click.option('--device', default='auto', help='設備 (cpu/cuda/mps/auto)')
@click.option('--seed', type=int, help='隨機種子')
@click.option('--handler-config', '-c', multiple=True,
              help='指定 handler 設定檔 (格式: handler_id=path.json)，可重複指定')
def train(input, output, handler, epochs, batch_size, learning_rate, validation_split, device, seed, handler_config):
    """
    訓練 CAPTCHA OCR 模型

    範例:
    captcha-ocr-devkit train --input ./images --output ./model.pkl
    captcha-ocr-devkit train --input ./images --output ./model.pkl --handler pytorch_handler
    """
    try:
        from ..core.handlers.registry import auto_discover_and_select
        from ..core.pipeline import create_pipeline_from_handlers
        from ..core.handlers.base import TrainingConfig

        logger.info("🚀 開始訓練模式")

        # 自動發現並選擇 handler
        selected_handler = auto_discover_and_select('train', handler, interactive=False)
        if not selected_handler:
            logger.error("無法找到或選擇 train handler")
            sys.exit(1)

        # 讀取 handler 設定
        handler_config_map = _parse_handler_configs(handler_config)

        # 創建 pipeline
        pipeline = create_pipeline_from_handlers(
            train_handler=selected_handler,
            handler_configs=handler_config_map,
        )

        # 準備訓練配置
        training_config = TrainingConfig(
            input_dir=Path(input),
            output_path=Path(output),
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            validation_split=validation_split,
            device=device,
            seed=seed
        )

        logger.info(f"📂 輸入目錄: {input}")
        logger.info(f"💾 輸出路徑: {output}")
        logger.info(f"🔧 使用 handler: {selected_handler}")

        # 執行訓練
        result = pipeline.train_model(training_config)

        if result.success:
            logger.info("✅ 訓練完成!")
            if result.metadata:
                training_time = result.metadata.get('training_time', 0)
                logger.info(f"⏱️  訓練時間: {training_time:.2f}s")
        else:
            logger.error(f"❌ 訓練失敗: {result.error}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⏹️  訓練被使用者中斷")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 訓練失敗: {e}")
        sys.exit(1)


@cli.command()
@click.option('--target', '-t', required=True, type=click.Path(exists=True),
              help='測試資料目錄')
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='模型檔案路徑')
@click.option('--handler', help='指定 evaluate handler 名稱')
@click.option('--handler-config', '-c', multiple=True,
              help='指定 handler 設定檔 (格式: handler_id=path.json)，可重複指定')
def evaluate(target, model, handler, handler_config):
    """
    評估 CAPTCHA OCR 模型

    範例:
    captcha-ocr-devkit evaluate --target ./test_images --model ./model.pkl
    captcha-ocr-devkit evaluate --target ./test_images --model ./model.pkl --handler pytorch_handler
    """
    try:
        from ..core.handlers.registry import auto_discover_and_select
        from ..core.pipeline import create_pipeline_from_handlers

        logger.info("📊 開始評估模式")

        # 自動發現並選擇 handler
        selected_handler = auto_discover_and_select('evaluate', handler, interactive=False)
        if not selected_handler:
            logger.error("無法找到或選擇 evaluate handler")
            sys.exit(1)

        # 讀取 handler 設定
        handler_config_map = _parse_handler_configs(handler_config)

        # 創建 pipeline
        pipeline = create_pipeline_from_handlers(
            evaluate_handler=selected_handler,
            handler_configs=handler_config_map,
        )

        logger.info(f"📂 測試資料: {target}")
        logger.info(f"🤖 模型檔案: {model}")
        logger.info(f"🔧 使用 handler: {selected_handler}")

        # 執行評估
        result = pipeline.evaluate_model(Path(model), Path(target))

        if result.success:
            logger.info("✅ 評估完成!")

            # 顯示結果
            from ..core.handlers.base import EvaluationResult
            if isinstance(result.data, EvaluationResult):
                eval_result = result.data
                logger.info(f"🎯 總體準確率: {eval_result.accuracy:.4f}")
                logger.info(f"🔤 字元準確率: {eval_result.character_accuracy:.4f}")
                logger.info(f"📊 測試樣本數: {eval_result.total_samples}")
                logger.info(f"✔️  正確預測數: {eval_result.correct_predictions}")

            if result.metadata:
                eval_time = result.metadata.get('evaluation_time', 0)
                logger.info(f"⏱️  評估時間: {eval_time:.2f}s")
        else:
            logger.error(f"❌ 評估失敗: {result.error}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n⏹️  評估被使用者中斷")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ 評估失敗: {e}")
        sys.exit(1)


@cli.command()
@click.option('--model', '-m', required=True, type=click.Path(exists=True),
              help='模型檔案路徑')
@click.option('--host', default="127.0.0.1", help='服務器主機')
@click.option('--port', default=54321, help='服務器端口')
@click.option('--handler', help='指定 OCR handler 名稱')
@click.option('--preprocess-handler', help='指定 preprocess handler 名稱')
@click.option('--workers', default=1, help='工作進程數')
@click.option('--reload', is_flag=True, help='自動重載')
@click.option('--handler-config', '-c', multiple=True,
              help='指定 handler 設定檔 (格式: handler_id=path.json)，可重複指定')
def api(model, host, port, handler, preprocess_handler, workers, reload, handler_config):
    """
    啟動 CAPTCHA OCR API 服務

    範例:
    captcha-ocr-devkit api --model ./model.pkl
    captcha-ocr-devkit api --model ./model.pkl --port 8080 --handler pytorch_handler
    """
    try:
        from ..core.handlers.registry import auto_discover_and_select

        logger.info("🌐 啟動 API 服務模式")

        # 自動發現並選擇 handlers
        default_ocr = handler or 'demo_ocr'
        selected_ocr_handler = auto_discover_and_select('ocr', default_ocr, interactive=False)
        if not selected_ocr_handler and handler is None:
            selected_ocr_handler = auto_discover_and_select('ocr', None, interactive=False)
        if not selected_ocr_handler:
            logger.error("無法找到或選擇 OCR handler")
            sys.exit(1)

        selected_preprocess_handler = None
        default_preprocess = preprocess_handler
        if not default_preprocess:
            ocr_key = selected_ocr_handler.lower()
            if ocr_key.startswith('transformer'):
                default_preprocess = 'transformer_preprocess'
            else:
                default_preprocess = 'demo_preprocess'

        selected_preprocess_handler = auto_discover_and_select('preprocess', default_preprocess, interactive=False)
        if not selected_preprocess_handler and preprocess_handler is None:
            selected_preprocess_handler = auto_discover_and_select('preprocess', None, interactive=False)
        if not selected_preprocess_handler and preprocess_handler:
            logger.error("無法找到指定的 preprocess handler")
            sys.exit(1)

        handler_config_map = _parse_handler_configs(handler_config)

        if handler_config_map:
            os.environ['CAPTCHA_HANDLER_CONFIGS'] = json.dumps(handler_config_map)

        logger.info(f"🤖 模型檔案: {model}")
        logger.info(f"🌍 服務地址: http://{host}:{port}")
        logger.info(f"🔧 OCR handler: {selected_ocr_handler}")
        if selected_preprocess_handler:
            logger.info(f"🖼️  Preprocess handler: {selected_preprocess_handler}")

        # 設定環境變數傳遞配置
        os.environ['CAPTCHA_MODEL_PATH'] = str(Path(model).absolute())
        os.environ['CAPTCHA_OCR_HANDLER'] = selected_ocr_handler
        if selected_preprocess_handler:
            os.environ['CAPTCHA_PREPROCESS_HANDLER'] = selected_preprocess_handler

        # 啟動 FastAPI 服務
        uvicorn.run(
            "captcha_ocr_devkit.api.server:app",
            host=host,
            port=port,
            workers=workers,
            reload=reload,
            log_level="info"
        )

    except KeyboardInterrupt:
        logger.info("\n⏹️  API 服務被使用者中斷")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ API 服務啟動失敗: {e}")
        sys.exit(1)


@cli.command()
@click.option('--output-dir', '-o', default='./handlers', type=click.Path(),
              help='輸出目錄')
@click.option('--force', is_flag=True, help='強制覆蓋現有檔案')
@click.option('--handler-dir', '-d', multiple=True, type=click.Path(),
              help='額外複製 handler 的來源目錄，可重複指定')
@click.option('--scripts-dir', default='./scripts', type=click.Path(),
              help='複製範例腳本的輸出目錄')
@click.option('--no-scripts', is_flag=True, help='不複製範例腳本')
def init(output_dir, force, handler_dir, scripts_dir, no_scripts):
    """
    初始化專案，複製範例 handlers

    範例:
    captcha-ocr-devkit init
    captcha-ocr-devkit init --output-dir ./my_handlers
    """
    try:
        logger.info("🛠️  初始化專案")
        logger.info(f"📦 captcha-ocr-devkit 版本: {CORE_VERSION}")

        output_path = Path(output_dir)

        # 檢查目錄是否存在
        if output_path.exists() and not force:
            if any(output_path.iterdir()):
                click.echo(f"⚠️  目錄 {output_dir} 已存在且不為空")
                if click.confirm("是否要繼續？這可能會覆蓋現有檔案"):
                    force = True
                else:
                    logger.info("取消初始化")
                    return

        # 創建目錄
        output_path.mkdir(parents=True, exist_ok=True)

        package_dir = Path(__file__).parent.parent
        examples_dir = package_dir / 'examples' / 'handlers'
        scripts_examples_dir = package_dir / 'examples' / 'scripts'
        try:
            repo_root = Path(__file__).resolve().parents[4]
        except IndexError:
            repo_root = package_dir.parent
        handler_patterns = [
            "*.py",
            "*-requirements.txt",
            "*-requirements-*.txt",
            "*.md",
            "*-config.json",
        ]
        script_patterns = ["*.sh"]

        def copy_assets(source_dir: Path, destination: Path, patterns: List[str], label: str) -> bool:
            if not source_dir.exists():
                logger.warning(f"⚠️  無法找到 {label} 來源: {source_dir}")
                return False

            destination.mkdir(parents=True, exist_ok=True)
            copied = False
            seen: Set[Path] = set()
            for pattern in patterns:
                for src_path in sorted(source_dir.glob(pattern)):
                    if src_path.name.startswith('__pycache__'):
                        continue
                    if src_path in seen:
                        continue
                    seen.add(src_path)
                    dest_file = destination / src_path.name
                    if dest_file.exists() and not force:
                        logger.info(f"⏭️  跳過 {src_path.name} (已存在)")
                        continue
                    shutil.copy2(src_path, dest_file)
                    logger.info(f"✅ 從 {label} 複製 {src_path.name}")
                    copied = True
            return copied

        logger.info(f"📂 複製範例 handlers 到 {output_dir}")
        files_copied = copy_assets(examples_dir, output_path, handler_patterns, "examples")

        extra_dirs = [Path(p) for p in handler_dir if p]
        for extra in extra_dirs:
            files_copied = copy_assets(Path(extra), output_path, handler_patterns, f"custom:{extra}") or files_copied

        if not files_copied:
            logger.info("📝 未複製到任何 handler，建立簡易骨架 basic_handler.py")
            create_basic_example_handler(output_path / 'basic_handler.py')

        if not no_scripts:
            scripts_destination = Path(scripts_dir)
            scripts_copied = copy_assets(scripts_examples_dir, scripts_destination, script_patterns, "scripts")
            if scripts_copied:
                for script_file in scripts_destination.glob("*.sh"):
                    try:
                        script_file.chmod(0o755)
                    except OSError:
                        pass

        # 創建 README
        readme_path = output_path / 'README.md'
        create_handlers_readme(readme_path)

        logger.info("\n✅ 初始化完成!")
        logger.info(f"📂 Handlers 目錄: {output_dir}")
        logger.info("\n接下來你可以:")
        logger.info("1. 編輯 handlers 實作你的邏輯")
        logger.info("2. 執行 create-handler 命令產生擴充骨架")
        logger.info("3. 執行 train 命令開始訓練")
        logger.info("4. 執行 evaluate 命令評估模型")
        logger.info("5. 執行 api 命令啟動服務")

    except Exception as e:
        logger.error(f"❌ 初始化失敗: {e}")
        sys.exit(1)








def create_basic_example_handler(output_path: Path) -> None:
    """創建基本範例 handler"""
    content = '''"""基本範例 Handler

這是一個簡單的範例，展示如何實作自己的 handlers
請根據你的需求修改和擴展這些實作
"""

from pathlib import Path
from typing import Any, List
import time

from captcha_ocr_devkit.core.handlers.base import (
    BasePreprocessHandler,
    BaseTrainHandler,
    BaseEvaluateHandler,
    BaseOCRHandler,
    HandlerResult,
    TrainingConfig,
    EvaluationResult
)


class BasicPreprocessHandler(BasePreprocessHandler):
    """基本圖片預處理 handler"""

    def process(self, image_data):
        # TODO: 實作你的圖片預處理邏輯
        # 例如: 調整大小、灰階化、去噪等
        return HandlerResult(
            success=True,
            data=image_data,  # 返回處理後的圖片
            metadata={"processed": True}
        )

    def get_supported_formats(self) -> List[str]:
        return [".png", ".jpg", ".jpeg"]

    def get_info(self):
        return {"name": "BasicPreprocessHandler", "version": "1.0"}


class BasicTrainHandler(BaseTrainHandler):
    """基本訓練 handler"""

    def train(self, config: TrainingConfig):
        # TODO: 實作你的訓練邏輯
        # 這裡只是一個示例
        print(f"開始訓練，輸入目錄: {config.input_dir}")
        print(f"輸出路徑: {config.output_path}")

        # 模擬訓練過程
        time.sleep(1)

        return HandlerResult(
            success=True,
            data={"model_path": str(config.output_path)},
            metadata={"epochs_completed": config.epochs}
        )

    def save_model(self, model_data: Any, output_path: Path) -> bool:
        # TODO: 實作模型保存邏輯
        return True

    def load_model(self, model_path: Path) -> Any:
        # TODO: 實作模型載入邏輯
        return None

    def get_info(self):
        return {"name": "BasicTrainHandler", "version": "1.0"}


class BasicEvaluateHandler(BaseEvaluateHandler):
    """基本評估 handler"""

    def evaluate(self, model_path: Path, test_data_path: Path):
        # TODO: 實作你的評估邏輯
        print(f"評估模型: {model_path}")
        print(f"測試資料: {test_data_path}")

        # 模擬評估過程
        time.sleep(0.5)

        # 模擬評估結果
        eval_result = EvaluationResult(
            accuracy=0.85,
            total_samples=100,
            correct_predictions=85,
            character_accuracy=0.92
        )

        return HandlerResult(
            success=True,
            data=eval_result
        )

    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]):
        # TODO: 實作指標計算
        total = len(predictions)
        correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)

        return EvaluationResult(
            accuracy=correct / total if total > 0 else 0,
            total_samples=total,
            correct_predictions=correct,
            character_accuracy=0.9  # 簡化
        )

    def get_info(self):
        return {"name": "BasicEvaluateHandler", "version": "1.0"}


class BasicOCRHandler(BaseOCRHandler):
    """基本 OCR handler"""

    def predict(self, processed_image: Any):
        # TODO: 實作你的 OCR 預測邏輯
        # 這裡只是返回模擬結果
        return HandlerResult(
            success=True,
            data="abcd",  # 預測的文字
            metadata={"confidence": 0.95}
        )

    def load_model(self, model_path: Path) -> bool:
        # TODO: 實作模型載入
        print(f"載入模型: {model_path}")
        return True

    def get_info(self):
        return {"name": "BasicOCRHandler", "version": "1.0"}
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def create_handlers_readme(output_path: Path) -> None:
    """創建 handlers README"""
    content = '''# Handlers 目錄

這個目錄包含了 CAPTCHA OCR 的各種 handlers。

## Handler 類型

1. **PreprocessHandler**: 圖片預處理
2. **TrainHandler**: 模型訓練
3. **EvaluateHandler**: 模型評估
4. **OCRHandler**: OCR 預測

## 使用方式

1. 編輯現有的 handler 檔案或使用 `captcha-ocr-devkit create-handler` 產生骨架
2. 確保你的 handler 繼承自對應的基類
3. 實作必要的抽象方法與 `get_info()` 描述、相依性
4. 使用 CLI 命令時系統會自動發現你的 handlers
5. 若使用 transformer handler，請先執行 `pip install -r transformer_handler-requirements.txt`

## 範例

```python
from captcha_ocr_devkit.core.handlers.base import BaseOCRHandler, HandlerResult

class MyOCRHandler(BaseOCRHandler):
    def predict(self, processed_image):
        # 你的 OCR 邏輯
        result = do_ocr(processed_image)
        return HandlerResult(success=True, data=result)

    def load_model(self, model_path):
        # 載入模型邏輯
        return True

    def get_info(self):
        return {"name": "MyOCRHandler", "version": "1.0"}
```

## 依賴管理

每個 handler 可以有自己的依賴。在 handler 檔案頂部 import 你需要的套件：

```python
# 例如使用 PyTorch
try:
    import torch
    import torchvision
except ImportError:
    print("請安裝 PyTorch: pip install torch torchvision")

# 或使用 OpenCV
try:
    import cv2
except ImportError:
    print("請安裝 OpenCV: pip install opencv-python")
```

這樣就可以讓不同的使用者選擇自己需要的技術棧！
'''

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)


def _to_camel_case(name: str) -> str:
    parts = re.split(r'[^0-9a-zA-Z]+', name)
    return ''.join(part.capitalize() for part in parts if part)


def _normalize_handler_filename(name: str) -> str:
    candidate = name.strip().lower()
    if candidate.endswith('.py'):
        candidate = candidate[:-3]
    if not candidate.endswith('_handler'):
        candidate += '_handler'
    return f"{candidate}.py"


def _generate_handler_template(
    class_prefix: str,
    handler_prefix: str,
    types: Iterable[str],
) -> str:
    header = (
        '"""Custom handler scaffold generated by captcha-ocr-devkit."""\n'
        '\n'
        'from pathlib import Path\n'
        'from typing import Any, List\n'
        '\n'
        'from captcha_ocr_devkit.core.handlers.base import (\n'
        '    BasePreprocessHandler,\n'
        '    BaseTrainHandler,\n'
        '    BaseEvaluateHandler,\n'
        '    BaseOCRHandler,\n'
        '    EvaluationResult,\n'
        '    HandlerResult,\n'
        '    TrainingConfig,\n'
        ')\n'
        '\n\n'
    )

    blocks: List[str] = []

    if 'preprocess' in types:
        blocks.append(f'''class {class_prefix}PreprocessHandler(BasePreprocessHandler):
    """圖片預處理 handler 範本。"""

    DESCRIPTION = "Describe what this preprocess handler does."
    SHORT_DESCRIPTION = "Short preprocess summary."
    HANDLER_ID = "{handler_prefix}_preprocess"
    VERSION = "0.1.0"

    def get_supported_formats(self) -> List[str]:
        return [".png", ".jpg", ".jpeg"]

    def process(self, image_data: Any) -> HandlerResult:
        # TODO: 依需求實作預處理邏輯
        return HandlerResult(success=True, data=image_data, metadata={{"processed": False}})

    def get_info(self) -> dict:
        return {{
            "name": self.name,
            "version": "0.1.0",
            "description": self.get_description(),
            "short_description": self.get_short_description(),
            "dependencies": self.get_dependencies(),
        }}


''')

    if 'train' in types:
        blocks.append(f'''class {class_prefix}TrainHandler(BaseTrainHandler):
    """模型訓練 handler 範本。"""

    DESCRIPTION = "Describe training strategy and data requirements."
    SHORT_DESCRIPTION = "Short training summary."
    HANDLER_ID = "{handler_prefix}_train"
    VERSION = "0.1.0"

    def train(self, config: TrainingConfig) -> HandlerResult:
        # TODO: 實作訓練流程
        return HandlerResult(success=True, data={{"model_path": str(config.output_path)}})

    def save_model(self, model_data: Any, output_path: Path) -> bool:
        # TODO: 儲存模型檔案
        return True

    def load_model(self, model_path: Path) -> Any:
        # TODO: 載入既有模型
        return None

    def get_info(self) -> dict:
        return {{
            "name": self.name,
            "version": "0.1.0",
            "description": self.get_description(),
            "short_description": self.get_short_description(),
            "dependencies": self.get_dependencies(),
        }}


''')

    if 'evaluate' in types:
        blocks.append(f'''class {class_prefix}EvaluateHandler(BaseEvaluateHandler):
    """模型評估 handler 範本。"""

    DESCRIPTION = "Describe evaluation metrics and datasets."
    SHORT_DESCRIPTION = "Short evaluation summary."
    HANDLER_ID = "{handler_prefix}_evaluate"
    VERSION = "0.1.0"

    def evaluate(self, model_path: Path, test_data_path: Path) -> HandlerResult:
        # TODO: 實作評估邏輯
        result = EvaluationResult(
            accuracy=0.0,
            total_samples=0,
            correct_predictions=0,
            character_accuracy=0.0,
        )
        return HandlerResult(success=True, data=result)

    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> EvaluationResult:
        # TODO: 實作客製化指標
        total = len(predictions)
        correct = sum(1 for pred, truth in zip(predictions, ground_truth) if pred == truth)
        accuracy = correct / total if total else 0.0
        return EvaluationResult(
            accuracy=accuracy,
            total_samples=total,
            correct_predictions=correct,
            character_accuracy=accuracy,
        )

    def get_info(self) -> dict:
        return {{
            "name": self.name,
            "version": "0.1.0",
            "description": self.get_description(),
            "short_description": self.get_short_description(),
            "dependencies": self.get_dependencies(),
        }}


''')

    if 'ocr' in types:
        blocks.append(f'''class {class_prefix}OCRHandler(BaseOCRHandler):
    """OCR 推論 handler 範本。"""

    DESCRIPTION = "Describe inference flow and model usage."
    SHORT_DESCRIPTION = "Short inference summary."
    HANDLER_ID = "{handler_prefix}_ocr"
    VERSION = "0.1.0"

    def load_model(self, model_path: Path) -> bool:
        # TODO: 實作模型載入
        return True

    def predict(self, processed_image: Any) -> HandlerResult:
        # TODO: 實作推論流程
        return HandlerResult(success=True, data="TODO", metadata={{"confidence": 0.0}})

    def get_info(self) -> dict:
        return {{
            "name": self.name,
            "version": "0.1.0",
            "description": self.get_description(),
            "short_description": self.get_short_description(),
            "dependencies": self.get_dependencies(),
        }}


''')

    return header + ''.join(blocks)


def _posix_relpath(target: Path, start: Path) -> str:
    return Path(os.path.relpath(target, start)).as_posix()


def _write_script(path: Path, content: str, force: bool) -> bool:
    if path.exists() and not force:
        logger.info(f"⏭️  跳過腳本 {path} (已存在)")
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    try:
        path.chmod(0o755)
    except OSError:
        pass
    logger.info(f"✅ 已建立腳本: {path}")
    return True


def _parse_handler_configs(entries: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    config_map: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        if '=' not in entry:
            raise click.BadParameter("格式需為 handler_id=path.json", param='handler_config')

        handler_id, path_str = entry.split('=', 1)
        handler_id = handler_id.strip()
        path_str = path_str.strip()

        if not handler_id:
            raise click.BadParameter("handler_id 不可為空", param='handler_config')

        config_path = Path(path_str).expanduser()
        if not config_path.is_file():
            raise click.BadParameter(f"找不到設定檔: {config_path}", param='handler_config')

        try:
            data = json.loads(config_path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as exc:
            raise click.BadParameter(f"解析 JSON 失敗 ({config_path}): {exc}", param='handler_config') from exc

        if not isinstance(data, dict):
            raise click.BadParameter(f"設定檔需為 JSON 物件: {config_path}", param='handler_config')

        config_map[handler_id] = data

    return config_map


def _build_train_script(
    handler_ids: Dict[str, str],
    config_rel_path: str,
    model_rel_path: str,
    rel_to_root_from_scripts: str,
) -> str:
    train_id = handler_ids['train']
    return (
        f"#!/bin/bash\n"
        "set -euo pipefail\n\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        f"PROJECT_ROOT=\"$(cd \"${{SCRIPT_DIR}}/{rel_to_root_from_scripts}\" && pwd)\"\n"
        "cd \"${PROJECT_ROOT}\"\n\n"
        f"CONFIG=\"{config_rel_path}\"\n"
        f"MODEL_PATH=\"{model_rel_path}\"\n\n"
        "mkdir -p \"$(dirname \"${MODEL_PATH}\")\"\n\n"
        "CONFIG_ARGS=()\n"
        "if [ -f \"${CONFIG}\" ]; then\n"
        "  while IFS= read -r token; do\n"
        "    CONFIG_ARGS+=(\"$token\")\n"
        "  done < <(python3 - <<'PY' \"${CONFIG}\"\n"
        "import json, sys\n"
        "from pathlib import Path\n\n"
        "path = Path(sys.argv[1])\n"
        "try:\n"
        "    data = json.loads(path.read_text())\n"
        "except Exception:\n"
        "    sys.exit(0)\n\n"
        "mapping = [\n"
        "    ('epochs', '--epochs'),\n"
        "    ('batch_size', '--batch-size'),\n"
        "    ('learning_rate', '--learning-rate'),\n"
        "    ('validation_split', '--validation-split'),\n"
        "    ('device', '--device'),\n"
        "    ('seed', '--seed'),\n"
        "]\n"
        "for key, flag in mapping:\n"
        "    if key in data:\n"
        "        print(flag)\n"
        "        print(str(data[key]))\n"
        "PY\n"
        "  )\n"
        "fi\n\n"
        f"# Example:\n# captcha-ocr-devkit train --input ./data --output {model_rel_path} --handler {train_id} --handler-config {train_id}={config_rel_path} --epochs 200 --batch-size 64 --learning-rate 0.001\n\n"
        "CMD_ARGS=(\n"
        "  \"--input\" \"./data\"\n"
        "  \"--output\" \"${MODEL_PATH}\"\n"
        f"  \"--handler\" \"{train_id}\"\n"
        f"  \"--handler-config\" \"{train_id}=${{CONFIG}}\"\n"
        ")\n"
        "CMD_ARGS+=(\"${CONFIG_ARGS[@]}\")\n"
        "CMD_ARGS+=(\"$@\")\n\n"
        "time captcha-ocr-devkit train \"${CMD_ARGS[@]}\"\n"
    )


def _build_evaluate_script(
    handler_ids: Dict[str, str],
    config_rel_path: str,
    model_rel_path: str,
    rel_to_root_from_scripts: str,
) -> str:
    evaluate_id = handler_ids['evaluate']
    return (
        f"#!/bin/bash\n"
        "set -euo pipefail\n\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        f"PROJECT_ROOT=\"$(cd \"${{SCRIPT_DIR}}/{rel_to_root_from_scripts}\" && pwd)\"\n"
        "cd \"${PROJECT_ROOT}\"\n\n"
        f"CONFIG=\"{config_rel_path}\"\n"
        f"MODEL_PATH=\"{model_rel_path}\"\n\n"
        f"# Example:\n# captcha-ocr-devkit evaluate --target ./data --model {model_rel_path} --handler {evaluate_id} --handler-config {evaluate_id}={config_rel_path}\n\n"
        "CMD_ARGS=(\n"
        "  \"--target\" \"./data\"\n"
        "  \"--model\" \"${MODEL_PATH}\"\n"
        f"  \"--handler\" \"{evaluate_id}\"\n"
        f"  \"--handler-config\" \"{evaluate_id}=${{CONFIG}}\"\n"
        ")\n"
        "CMD_ARGS+=(\"$@\")\n\n"
        "time captcha-ocr-devkit evaluate \"${CMD_ARGS[@]}\"\n"
    )


def _build_api_script(
    handler_ids: Dict[str, str],
    config_rel_path: str,
    model_rel_path: str,
    rel_to_root_from_scripts: str,
    include_preprocess: bool,
) -> str:
    ocr_id = handler_ids['ocr']
    preprocess_id = handler_ids.get('preprocess') if include_preprocess else None

    return (
        f"#!/bin/bash\n"
        "set -euo pipefail\n\n"
        "SCRIPT_DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        f"PROJECT_ROOT=\"$(cd \"${{SCRIPT_DIR}}/{rel_to_root_from_scripts}\" && pwd)\"\n"
        "cd \"${PROJECT_ROOT}\"\n\n"
        f"CONFIG=\"{config_rel_path}\"\n"
        f"MODEL_PATH=\"{model_rel_path}\"\n\n"
        "CLI_ARGS=()\n"
        "if [ -f \"${CONFIG}\" ]; then\n"
        "  while IFS= read -r token; do\n"
        "    CLI_ARGS+=(\"$token\")\n"
        "  done < <(python3 - <<'PY' \"${CONFIG}\"\n"
        "import json, sys\n"
        "from pathlib import Path\n\n"
        "path = Path(sys.argv[1])\n"
        "try:\n"
        "    data = json.loads(path.read_text())\n"
        "except Exception:\n"
        "    sys.exit(0)\n\n"
        "mapping = [\n"
        "    ('host', '--host'),\n"
        "    ('port', '--port'),\n"
        "    ('workers', '--workers'),\n"
        "]\n"
        "for key, flag in mapping:\n"
        "    if key in data:\n"
        "        print(flag)\n"
        "        print(str(data[key]))\n"
        "reload_value = data.get('reload')\n"
        "if isinstance(reload_value, bool) and reload_value:\n"
        "    print('--reload')\n"
        "PY\n"
        "  )\n"
        "fi\n\n"
        f"# Example:\n# captcha-ocr-devkit api --handler {ocr_id} --model {model_rel_path} --handler-config {ocr_id}={config_rel_path}\n\n"
        "CMD_ARGS=(\n"
        f"  \"--handler\" \"{ocr_id}\"\n"
        f"  \"--handler-config\" \"{ocr_id}=${{CONFIG}}\"\n"
        "  \"--model\" \"${MODEL_PATH}\"\n"
        ")\n"
        + (
            "CMD_ARGS+=(\"--preprocess-handler\" \"{0}\" \"--handler-config\" \"{0}=${{CONFIG}}\")\n".format(preprocess_id)
            if preprocess_id else ""
        )
        + "CMD_ARGS+=(\"${CLI_ARGS[@]}\")\n"
        "CMD_ARGS+=(\"$@\")\n\n"
        "captcha-ocr-devkit api \"${CMD_ARGS[@]}\"\n"
    )


def _generate_handler_readme(
    handler_name: str,
    class_prefix: str,
    handler_prefix: str,
    selected_types: Set[str],
    config_rel_path: str,
    script_paths: List[str],
) -> str:
    type_map = {
        'preprocess': '⚙️ Preprocess: 負責圖片載入、尺寸調整、資料增強與張量化。',
        'train': '🎯 Train: 負責資料切分、模型訓練流程與 checkpoint 管理。',
        'evaluate': '📊 Evaluate: 負責載入模型、跑測試資料並彙整精準率/字元率等指標。',
        'ocr': '🔍 OCR: 負責載入模型、處理輸入並回傳推論結果與額外中繼資料。',
    }
    bullet_lines = [type_map[t] for t in sorted(selected_types) if t in type_map]
    if not bullet_lines:
        bullet_lines = ['(尚未指定 handler 類型，請更新 README 說明)']
    bullet_section = '\n'.join(f'- {line}' for line in bullet_lines)

    scripts_section = '\n'.join(f'- `{path}`' for path in script_paths) if script_paths else '尚未產生腳本。使用 create-handler 時啟用腳本生成功能即可自動建立。'

    handler_ids = {
        'preprocess': f"{handler_prefix}_preprocess",
        'train': f"{handler_prefix}_train",
        'evaluate': f"{handler_prefix}_evaluate",
        'ocr': f"{handler_prefix}_ocr",
    }
    available_ids = [handler_ids[t] for t in sorted(selected_types) if t in handler_ids]
    available_ids_text = ', '.join(f'`{hid}`' for hid in available_ids) if available_ids else 'N/A'
    return (
        f"# {handler_name} Handler Blueprint\n\n"
        f"這份 README 說明 `{class_prefix}` handler 骨架的結構與擴充方式。\n\n"
        "## 架構重點\n\n"
        "- Handler 介面源自 `captcha_ocr_devkit.core.handlers.base`，可與 CLI 指令整合。\n"
        "- 預設實作會回傳成功的 `HandlerResult`，請依需求補上真實邏輯。\n"
        "- 依照資料流分層，方便將預處理 / 訓練 / 評估 / 推論分工。\n\n"
        "## 產生的元件\n\n"
        f"{bullet_section}\n\n"
        "## Handler 設定檔\n\n"
        f"- `{config_rel_path}`：預設為空 JSON，可依照 handler 需求填入參數。\n\n"
        "## 常用腳本\n\n"
        f"{scripts_section}\n\n"
        "## 下一步\n\n"
        "1. 依實際需求補齊模型載入、資料處理與錯誤處理。\n"
        "2. 若有額外相依套件，請同步編輯 `requirements.txt` 或專屬 `*-requirements.txt`。\n"
        f"3. 在 CLI 上使用 {available_ids_text} 搭配 `--handler-config <handler_id>={config_rel_path}` 測試流程，並確保 `get_info()` 內容完整。\n"
    )


@cli.command('create-handler')
@click.argument('handler_name')
@click.option('--output-dir', '-o', default='./handlers', type=click.Path(), help='輸出目錄')
@click.option('--types', '-t', default='preprocess,train,evaluate,ocr', help='指定要產生的 handler 類型 (以逗號分隔)。')
@click.option('--force', is_flag=True, help='允許覆蓋已存在檔案')
@click.option('--scripts-dir', default='./scripts', type=click.Path(), help='產生腳本的輸出目錄')
@click.option('--no-scripts', is_flag=True, help='不自動產生 helper 腳本')
def create_handler(
    handler_name: str,
    output_dir: str,
    types: str,
    force: bool,
    scripts_dir: str,
    no_scripts: bool,
) -> None:
    """建立新的 handler 骨架檔案。"""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = _normalize_handler_filename(handler_name)
        target_file = output_path / filename

        if target_file.exists() and not force:
            logger.error(f"❌ 檔案 {target_file} 已存在。使用 --force 以覆蓋。")
            sys.exit(1)

        selected_types = {t.strip().lower() for t in types.split(',') if t.strip()}
        allowed = {"preprocess", "train", "evaluate", "ocr"}
        unknown = selected_types - allowed
        if unknown:
            logger.error(f"❌ 未知的 handler 類型: {', '.join(sorted(unknown))}")
            sys.exit(1)
        if not selected_types:
            logger.error("❌ 請至少指定一種 handler 類型")
            sys.exit(1)

        class_prefix = _to_camel_case(handler_name)
        if not class_prefix:
            logger.error("❌ handler_name 無法轉換為有效類別名稱")
            sys.exit(1)

        module_stem = target_file.stem
        handler_prefix = module_stem[:-8] if module_stem.endswith('_handler') else module_stem

        content = _generate_handler_template(class_prefix, handler_prefix, selected_types)
        target_file.write_text(content, encoding='utf-8')

        config_path = target_file.with_name(f"{target_file.stem}-config.json")
        if config_path.exists() and not force:
            logger.info(f"⏭️  跳過設定檔 {config_path.name} (已存在)")
        else:
            config_path.write_text("{}\n", encoding='utf-8')
            logger.info(f"✅ 已建立設定檔: {config_path}")

        project_root = Path.cwd().resolve()
        config_rel_path = _posix_relpath(config_path.resolve(), project_root)

        script_paths: List[str] = []
        if not no_scripts:
            scripts_dir_path = Path(scripts_dir)
            scripts_dir_path.mkdir(parents=True, exist_ok=True)
            scripts_dir_resolved = scripts_dir_path.resolve()
            rel_to_root = _posix_relpath(project_root, scripts_dir_resolved)

            handler_ids = {
                'preprocess': f"{handler_prefix}_preprocess",
                'train': f"{handler_prefix}_train",
                'evaluate': f"{handler_prefix}_evaluate",
                'ocr': f"{handler_prefix}_ocr",
            }

            model_rel_path = f"model/model.{handler_prefix}"

            if 'train' in selected_types:
                script_path = scripts_dir_path / f"train_{handler_prefix}.sh"
                script_content = _build_train_script(
                    handler_ids,
                    config_rel_path,
                    model_rel_path,
                    rel_to_root,
                )
                if _write_script(script_path, script_content, force) or script_path.exists():
                    script_paths.append(_posix_relpath(script_path.resolve(), project_root))
            if 'evaluate' in selected_types:
                script_path = scripts_dir_path / f"evaluate_{handler_prefix}.sh"
                script_content = _build_evaluate_script(
                    handler_ids,
                    config_rel_path,
                    model_rel_path,
                    rel_to_root,
                )
                if _write_script(script_path, script_content, force) or script_path.exists():
                    script_paths.append(_posix_relpath(script_path.resolve(), project_root))
            if 'ocr' in selected_types:
                script_path = scripts_dir_path / f"api_{handler_prefix}.sh"
                script_content = _build_api_script(
                    handler_ids,
                    config_rel_path,
                    model_rel_path,
                    rel_to_root,
                    include_preprocess='preprocess' in selected_types,
                )
                if _write_script(script_path, script_content, force) or script_path.exists():
                    script_paths.append(_posix_relpath(script_path.resolve(), project_root))

        readme_path = target_file.with_name(f"{target_file.stem}-README.md")
        if readme_path.exists() and not force:
            logger.info(f"⏭️  跳過 README {readme_path.name} (已存在)")
        else:
            readme_content = _generate_handler_readme(
                handler_name,
                class_prefix,
                handler_prefix,
                selected_types,
                config_rel_path,
                script_paths,
            )
            readme_path.write_text(readme_content, encoding='utf-8')
            logger.info(f"✅ 已建立 README: {readme_path}")

        logger.info(f"✅ 已建立 handler: {target_file}")
    except Exception as exc:
        logger.error(f"❌ 建立 handler 失敗: {exc}")
        sys.exit(1)


if __name__ == '__main__':
    cli()
