"""
Handler 系統單元測試
"""
import pytest
from pathlib import Path

from captcha_ocr_devkit.core.handlers.registry import registry
from captcha_ocr_devkit.core.handlers.base import HandlerResult


class TestHandlerRegistry:
    """Handler 註冊系統測試"""

    def test_handler_discovery_default_directory(self, project_root: Path):
        """測試預設目錄的 handler 發現"""
        handlers_dir = project_root / "handlers"
        if not handlers_dir.exists():
            pytest.skip("No default handlers directory found")

        discovered = registry.discover_handlers(handlers_dir)

        # 檢查基本結構
        assert isinstance(discovered, dict)
        expected_types = ["preprocess", "train", "evaluate", "ocr"]
        for handler_type in expected_types:
            assert handler_type in discovered
            assert isinstance(discovered[handler_type], list)

    def test_handler_discovery_custom_directory(self, handlers_dir: Path):
        """測試自訂目錄的 handler 發現"""
        discovered = registry.discover_handlers(handlers_dir)

        # 應該發現所有類型的 handlers
        assert "preprocess" in discovered
        assert "train" in discovered
        assert "evaluate" in discovered
        assert "ocr" in discovered

        # 每個類型至少應該有 Demo handlers
        assert "demo_preprocess" in discovered["preprocess"]
        assert "demo_train" in discovered["train"]
        assert "demo_evaluate" in discovered["evaluate"]
        assert "demo_ocr" in discovered["ocr"]

    def test_handler_creation_demo_ocr(self, handlers_dir: Path):
        """測試 DemoOCRHandler 創建"""
        registry.discover_handlers(handlers_dir)
        handler = registry.create_handler("ocr", "demo_ocr")

        assert handler is not None
        assert handler.name == "demo_ocr"

        # 測試 get_info 方法
        info = handler.get_info()
        assert isinstance(info, dict)
        assert "name" in info
        assert "version" in info
        assert info["name"] == "demo_ocr"

    def test_handler_creation_demo_preprocess(self, handlers_dir: Path):
        """測試 DemoPreprocessHandler 創建"""
        registry.discover_handlers(handlers_dir)
        handler = registry.create_handler("preprocess", "demo_preprocess")

        assert handler is not None
        assert handler.name == "demo_preprocess"

        # 測試基本功能
        info = handler.get_info()
        assert info["name"] == "demo_preprocess"

    def test_handler_creation_invalid(self, handlers_dir: Path):
        """測試無效 handler 創建"""
        registry.discover_handlers(handlers_dir)

        # 無效類型
        with pytest.raises(ValueError, match="Unknown handler type"):
            registry.create_handler("invalid_type", "SomeHandler")

        # 無效名稱
        with pytest.raises(ValueError, match="Handler .* not found"):
            registry.create_handler("ocr", "NonExistentHandler")

    def test_registry_state_management(self, handlers_dir: Path):
        """測試註冊表狀態管理"""
        # 清空註冊表
        registry._handlers.clear()
        assert len(registry._handlers) == 0

        # 發現 handlers
        discovered = registry.discover_handlers(handlers_dir)
        assert len(registry._handlers) > 0

        # 再次發現應該更新狀態
        discovered2 = registry.discover_handlers(handlers_dir)
        assert set(discovered.keys()) == set(discovered2.keys())
        for key in discovered:
            assert set(discovered[key]) == set(discovered2[key])


class TestHandlerBase:
    """Handler 基礎功能測試"""

    def test_demo_ocr_handler_predict(self, handlers_dir: Path):
        """測試 DemoOCRHandler 預測功能"""
        registry.discover_handlers(handlers_dir)
        handler = registry.create_handler("ocr", "demo_ocr")
        
        # 測試預測
        fake_image = b"fake image data"
        result = handler.predict(fake_image)

        assert isinstance(result, HandlerResult)
        assert result.success is True
        assert isinstance(result.data, str)
        assert len(result.data) > 0

        # 檢查 metadata
        if result.metadata:
            assert "confidence" in result.metadata
            assert isinstance(result.metadata["confidence"], (int, float))
            assert 0 <= result.metadata["confidence"] <= 1

    def test_demo_ocr_handler_load_model(self, handlers_dir: Path, test_model_file: Path):
        """測試 DemoOCRHandler 模型載入"""
        registry.discover_handlers(handlers_dir)
        handler = registry.create_handler("ocr", "demo_ocr")

        # 測試模型載入
        success = handler.load_model(test_model_file)
        assert success is True

        # 載入後 get_info 應該包含模型資訊
        info = handler.get_info()
        assert "model_loaded" in info
        assert info["model_loaded"] is True

    def test_demo_preprocess_handler_process(self, handlers_dir: Path):
        """測試 DemoPreprocessHandler 處理功能"""
        registry.discover_handlers(handlers_dir)
        handler = registry.create_handler("preprocess", "demo_preprocess")

        # 測試處理
        fake_image = b"fake image data"
        result = handler.process(fake_image)

        assert isinstance(result, HandlerResult)
        assert result.success is True
        assert result.data is not None

    def test_handler_result_structure(self):
        """測試 HandlerResult 資料結構"""
        # 成功結果
        success_result = HandlerResult(
            success=True,
            data="test_data",
            metadata={"confidence": 0.95}
        )

        assert success_result.success is True
        assert success_result.data == "test_data"
        assert success_result.metadata["confidence"] == 0.95
        assert success_result.error is None

        # 失敗結果
        error_result = HandlerResult(
            success=False,
            error="Test error message"
        )

        assert error_result.success is False
        assert error_result.error == "Test error message"
        assert error_result.data is None


@pytest.mark.slow
class TestHandlerTraining:
    """Handler 訓練相關測試"""

    def test_demo_train_handler(self, handlers_dir: Path, test_images_dir: Path, temp_dir: Path):
        """測試 DemoTrainHandler 訓練功能"""
        registry.discover_handlers(handlers_dir)
        handler = registry.create_handler("train", "demo_train")

        from captcha_ocr_devkit.core.handlers.base import TrainingConfig

        # 準備訓練配置
        output_path = temp_dir / "trained_model.json"
        config = TrainingConfig(
            input_dir=test_images_dir,
            output_path=output_path,
            epochs=1,
            batch_size=32,
            learning_rate=0.001,
            validation_split=0.2
        )

        # 執行訓練
        result = handler.train(config)

        assert isinstance(result, HandlerResult)
        assert result.success is True
        assert output_path.exists()

        # 檢查生成的模型檔案
        import json
        with open(output_path) as f:
            model_data = json.load(f)

        assert "model_type" in model_data
        assert "training_config" in model_data
        assert "dataset_info" in model_data

    def test_demo_evaluate_handler(self, handlers_dir: Path, test_images_dir: Path, test_model_file: Path):
        """測試 DemoEvaluateHandler 評估功能"""
        registry.discover_handlers(handlers_dir)
        handler = registry.create_handler("evaluate", "demo_evaluate")

        # 執行評估
        result = handler.evaluate(test_model_file, test_images_dir)

        assert isinstance(result, HandlerResult)
        assert result.success is True

        # 檢查評估結果結構
        eval_result = result.data
        assert hasattr(eval_result, 'accuracy')
        assert hasattr(eval_result, 'character_accuracy')
        assert hasattr(eval_result, 'total_samples')
        assert hasattr(eval_result, 'correct_predictions')

        assert 0 <= eval_result.accuracy <= 1
        assert 0 <= eval_result.character_accuracy <= 1
        assert eval_result.total_samples > 0
