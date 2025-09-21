"""pytest 配置和共用 fixtures."""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient


# 將 src 加入 sys.path，避免必須先 pip install -e .
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from captcha_ocr_devkit import __version__ as CORE_VERSION
from handlers.demo_handler import DEMO_HANDLER_VERSION


@pytest.fixture(scope="session")
def project_root() -> Path:
    """專案根目錄"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def venv_python(project_root: Path) -> Path:
    """虛擬環境 Python 執行檔路徑"""
    venv_python = project_root / "venv" / "bin" / "python3"
    if venv_python.exists():
        return venv_python

    venv_python = project_root / "venv" / "Scripts" / "python.exe"  # Windows
    if venv_python.exists():
        return venv_python

    return Path(sys.executable)


@pytest.fixture(scope="session")
def cli_path(project_root: Path) -> Path:
    """CLI 工具路徑"""
    cli_path = project_root / "venv" / "bin" / "captcha-ocr-devkit"
    if cli_path.exists():
        return cli_path

    cli_path = project_root / "venv" / "Scripts" / "captcha-ocr-devkit.exe"  # Windows
    if cli_path.exists():
        return cli_path

    # 動態建立 CLI 包裝腳本，避免必須先安裝至虛擬環境
    wrapper_dir = project_root / ".pytest_cli"
    wrapper_dir.mkdir(exist_ok=True)
    wrapper_path = wrapper_dir / "captcha-ocr-devkit"

    if not wrapper_path.exists():
        script = textwrap.dedent(
            f"""#!/usr/bin/env python3
import sys
from pathlib import Path

project_root = Path(r"{project_root}")
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from captcha_ocr_devkit.cli.main import cli

if __name__ == "__main__":
    cli(prog_name="captcha-ocr-devkit")
"""
        )
        wrapper_path.write_text(script)
        wrapper_path.chmod(0o755)

    return wrapper_path


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """臨時目錄 fixture"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def test_images_dir(temp_dir: Path) -> Path:
    """測試圖片目錄"""
    images_dir = temp_dir / "test_images"
    images_dir.mkdir()

    # 創建假圖片檔案
    test_files = [
        "abcd_001.png", "abcd_002.png", "abcd_003.png",
        "efgh_001.png", "efgh_002.png", "efgh_003.png",
        "ijkl_001.png", "ijkl_002.png"
    ]

    for filename in test_files:
        (images_dir / filename).write_text(f"fake image data for {filename}")

    return images_dir


@pytest.fixture
def test_model_file(temp_dir: Path) -> Path:
    """測試模型檔案"""
    model_file = temp_dir / "test_model.json"
    model_data = {
        "model_type": "test",
        "version": "1.0.0",
        "demo_mode": True,
        "training_config": {
            "epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.001
        },
        "dataset_info": {
            "total_images": 8,
            "unique_labels": 3,
            "sample_labels": ["abcd", "efgh", "ijkl"]
        }
    }

    with open(model_file, 'w') as f:
        json.dump(model_data, f, indent=2)

    return model_file


@pytest.fixture
def handlers_dir(temp_dir: Path, cli_path: Path) -> Path:
    """Handler 目錄 fixture"""
    handlers_dir = temp_dir / "handlers"

    # 使用 init 命令創建 handlers
    cmd = [str(cli_path), "init", "--output-dir", str(handlers_dir), "--force"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        pytest.fail(f"Failed to initialize handlers: {result.stderr}")

    return handlers_dir


@pytest.fixture
def api_client():
    """FastAPI 測試客戶端"""
    # 需要先設定環境變數
    os.environ['CAPTCHA_MODEL_PATH'] = str(Path(__file__).parent.parent / "test_model.json")
    os.environ['CAPTCHA_OCR_HANDLER'] = "demo_ocr"
    os.environ['CAPTCHA_PREPROCESS_HANDLER'] = "demo_preprocess"

    from captcha_ocr_devkit.api.server import app

    return TestClient(app)


@pytest.fixture
def fake_image_bytes() -> bytes:
    """假的圖片 bytes 資料"""
    return b"fake image data for testing"


@pytest.fixture
def sample_ocr_response() -> dict:
    """範例 OCR 回應"""
    return {
        "status": True,
        "data": "abcd",
        "confidence": 95.5,
        "processing_time": 0.123,
        "timestamp": "2024-01-01T12:00:00.000Z",
        "method": "Handler Pipeline OCR",
        "core_version": CORE_VERSION,
        "handler_versions": {
            "ocr": DEMO_HANDLER_VERSION,
            "preprocess": DEMO_HANDLER_VERSION
        },
        "details": {
            "character_confidences": [98.1, 94.2, 96.8, 92.4],
            "character_count": 4,
            "image_size": "128x64",
            "handler_info": {
                "preprocess_handler": "demo_preprocess",
                "ocr_handler": "demo_ocr"
            },
            "warnings": [],
            "metadata_completeness": "full"
        }
    }


# 跳過需要特殊依賴的測試的標記
def pytest_configure(config):
    """Configure custom markers"""
    config.addinivalue_line(
        "markers", "pillow: mark test as requiring Pillow"
    )
    config.addinivalue_line(
        "markers", "pytorch: mark test as requiring PyTorch"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers based on test location"""
    for item in items:
        # 為 integration 目錄的測試添加 integration 標記
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # 為慢速測試添加標記
        if "slow" in item.name:
            item.add_marker(pytest.mark.slow)
