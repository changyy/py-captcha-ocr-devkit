"""
CLI 命令單元測試
"""
import json
import subprocess
from pathlib import Path

import pytest


class TestCLIBasic:
    """CLI 基本功能測試"""

    def test_cli_help(self, cli_path: Path):
        """測試 CLI 幫助訊息"""
        result = subprocess.run(
            [str(cli_path), "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "captcha-ocr-devkit" in result.stdout
        assert "init" in result.stdout
        assert "train" in result.stdout
        assert "evaluate" in result.stdout
        assert "api" in result.stdout

    def test_cli_version(self, cli_path: Path):
        """測試 CLI 版本訊息"""
        result = subprocess.run(
            [str(cli_path), "--version"],
            capture_output=True,
            text=True
        )

        # 版本命令可能回傳 0 或顯示版本資訊
        assert "version" in result.stdout.lower() or result.returncode == 0

    def test_invalid_command(self, cli_path: Path):
        """測試無效命令"""
        result = subprocess.run(
            [str(cli_path), "invalid_command"],
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        assert "Usage:" in result.stderr or "Error:" in result.stderr


class TestCLIInit:
    """CLI init 命令測試"""

    def test_init_command_help(self, cli_path: Path):
        """測試 init 命令幫助"""
        result = subprocess.run(
            [str(cli_path), "init", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "初始化專案" in result.stdout
        assert "--output-dir" in result.stdout
        assert "--force" in result.stdout

    def test_init_command_basic(self, cli_path: Path, temp_dir: Path):
        """測試基本 init 命令"""
        output_dir = temp_dir / "test_handlers"

        result = subprocess.run(
            [str(cli_path), "init", "--output-dir", str(output_dir)],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert output_dir.exists()

        # 檢查生成的檔案
        expected_files = [
            "demo_handler.py",
            "transformer_handler.py",
            "transformer_handler-requirements.txt",
            "README.md"
        ]

        for filename in expected_files:
            assert (output_dir / filename).exists(), f"Missing file: {filename}"

    def test_init_command_force(self, cli_path: Path, temp_dir: Path):
        """測試 init 命令的 force 參數"""
        output_dir = temp_dir / "test_handlers"
        output_dir.mkdir()

        # 創建一個現有檔案
        (output_dir / "existing_file.py").write_text("existing content")

        result = subprocess.run(
            [str(cli_path), "init", "--output-dir", str(output_dir), "--force"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert (output_dir / "demo_handler.py").exists()
        assert (output_dir / "existing_file.py").exists()  # 原有檔案保留


class TestCLITrain:
    """CLI train 命令測試"""

    def test_train_command_help(self, cli_path: Path):
        """測試 train 命令幫助"""
        result = subprocess.run(
            [str(cli_path), "train", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "訓練 CAPTCHA OCR 模型" in result.stdout
        assert "--input" in result.stdout
        assert "--output" in result.stdout
        assert "--handler" in result.stdout

    def test_train_command_missing_args(self, cli_path: Path):
        """測試 train 命令缺少參數"""
        result = subprocess.run(
            [str(cli_path), "train"],
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "missing" in result.stderr.lower()

    @pytest.mark.slow
    def test_train_command_demo(self, cli_path: Path, test_images_dir: Path, temp_dir: Path):
        """測試使用 DemoTrainHandler 的訓練命令"""
        output_model = temp_dir / "trained_model.json"

        result = subprocess.run([
            str(cli_path), "train",
            "--input", str(test_images_dir),
            "--output", str(output_model),
            "--handler", "demo_train",
            "--epochs", "1",
            "--validation-split", "0.2"
        ], capture_output=True, text=True, cwd=test_images_dir.parent)

        assert result.returncode == 0, f"Train failed: {result.stderr}"
        assert output_model.exists()

        # 檢查生成的模型檔案
        with open(output_model) as f:
            model_data = json.load(f)

        assert "model_type" in model_data
        assert "training_config" in model_data
        assert "dataset_info" in model_data
        assert model_data["training_config"]["epochs"] == 1
        assert model_data["training_config"]["validation_split"] == 0.2


class TestCLIEvaluate:
    """CLI evaluate 命令測試"""

    def test_evaluate_command_help(self, cli_path: Path):
        """測試 evaluate 命令幫助"""
        result = subprocess.run(
            [str(cli_path), "evaluate", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "評估 CAPTCHA OCR 模型" in result.stdout
        assert "--target" in result.stdout
        assert "--model" in result.stdout

    def test_evaluate_command_missing_args(self, cli_path: Path):
        """測試 evaluate 命令缺少參數"""
        result = subprocess.run(
            [str(cli_path), "evaluate"],
            capture_output=True,
            text=True
        )

        assert result.returncode != 0

    def test_evaluate_command_demo(self, cli_path: Path, test_images_dir: Path, test_model_file: Path):
        """測試使用 DemoEvaluateHandler 的評估命令"""
        result = subprocess.run([
            str(cli_path), "evaluate",
            "--target", str(test_images_dir),
            "--model", str(test_model_file),
            "--handler", "demo_evaluate"
        ], capture_output=True, text=True, cwd=test_images_dir.parent)

        assert result.returncode == 0, f"Evaluate failed: {result.stderr}"
        assert "準確率" in result.stderr or "accuracy" in result.stderr.lower()


class TestCLIAPI:
    """CLI api 命令測試"""

    def test_api_command_help(self, cli_path: Path):
        """測試 api 命令幫助"""
        result = subprocess.run(
            [str(cli_path), "api", "--help"],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert "啟動 CAPTCHA OCR API 服務" in result.stdout
        assert "--model" in result.stdout
        assert "--port" in result.stdout
        assert "--handler" in result.stdout

    def test_api_command_missing_model(self, cli_path: Path):
        """測試 api 命令缺少模型"""
        result = subprocess.run(
            [str(cli_path), "api"],
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "missing" in result.stderr.lower()

    def test_api_command_invalid_model(self, cli_path: Path):
        """測試 api 命令使用不存在的模型"""
        result = subprocess.run([
            str(cli_path), "api",
            "--model", "/nonexistent/model.json",
            "--port", "54999"
        ], capture_output=True, text=True)

        assert result.returncode != 0
        assert "not exist" in result.stderr or "找不到" in result.stderr or "does not exist" in result.stderr
