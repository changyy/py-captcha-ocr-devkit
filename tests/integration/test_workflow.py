"""
整合測試 - 完整工作流程
"""
import json
import socket
import subprocess
import time
import requests
from pathlib import Path

import pytest


@pytest.mark.integration
class TestCompleteWorkflow:
    """完整工作流程整合測試"""

    @pytest.mark.slow
    def test_full_pipeline_demo_handlers(self, cli_path: Path, temp_dir: Path):
        """測試完整的 Demo handlers 流程: init → train → evaluate → api"""

        # 1. 初始化 handlers
        handlers_dir = temp_dir / "handlers"
        result = subprocess.run([
            str(cli_path), "init",
            "--output-dir", str(handlers_dir),
            "--force"
        ], capture_output=True, text=True)

        assert result.returncode == 0
        assert handlers_dir.exists()
        assert (handlers_dir / "demo_handler.py").exists()

        # 2. 準備訓練資料
        images_dir = temp_dir / "images"
        images_dir.mkdir()

        test_files = [
            "abcd_001.png", "abcd_002.png", "abcd_003.png",
            "efgh_001.png", "efgh_002.png",
            "ijkl_001.png"
        ]

        for filename in test_files:
            (images_dir / filename).write_text(f"fake image data for {filename}")

        # 3. 執行訓練
        model_file = temp_dir / "trained_model.json"
        result = subprocess.run([
            str(cli_path), "train",
            "--input", str(images_dir),
            "--output", str(model_file),
            "--handler", "DemoTrainHandler",
            "--epochs", "1",
            "--validation-split", "0.2"
        ], capture_output=True, text=True, cwd=handlers_dir.parent)

        assert result.returncode == 0, f"Training failed: {result.stderr}"
        assert model_file.exists()

        # 檢查模型檔案
        with open(model_file) as f:
            model_data = json.load(f)
        assert "model_type" in model_data
        assert "dataset_info" in model_data

        # 4. 執行評估
        result = subprocess.run([
            str(cli_path), "evaluate",
            "--target", str(images_dir),
            "--model", str(model_file),
            "--handler", "DemoEvaluateHandler"
        ], capture_output=True, text=True, cwd=handlers_dir.parent)

        assert result.returncode == 0, f"Evaluation failed: {result.stderr}"
        assert "準確率" in result.stderr or "accuracy" in result.stderr.lower()

        # 5. 啟動 API 服務並測試
        probe_socket = None
        try:
            probe_socket = socket.socket()
            probe_socket.bind(("127.0.0.1", 0))
        except OSError:
            pytest.skip("Local TCP binding is not permitted in this environment")
        finally:
            try:
                if probe_socket:
                    probe_socket.close()
            except Exception:
                pass

        api_port = 54398
        api_process = subprocess.Popen([
            str(cli_path), "api",
            "--model", str(model_file),
            "--port", str(api_port),
            "--handler", "DemoOCRHandler",
            "--preprocess-handler", "DemoPreprocessHandler"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=handlers_dir.parent)

        try:
            # 等待 API 服務啟動
            time.sleep(5)

            # 測試健康檢查
            response = requests.get(f"http://localhost:{api_port}/api/v1/health", timeout=10)
            assert response.status_code == 200

            health_data = response.json()
            assert health_data["status"] == "healthy"
            assert health_data["model_loaded"] is True

            # 測試 OCR 端點
            test_image_data = b"fake image data for API test"
            files = {"file": ("test.png", test_image_data, "image/png")}

            response = requests.post(
                f"http://localhost:{api_port}/api/v1/ocr",
                files=files,
                timeout=10
            )

            assert response.status_code == 200
            ocr_data = response.json()

            # 檢查 OCR 回應格式
            assert "status" in ocr_data
            assert "processing_time" in ocr_data
            assert "timestamp" in ocr_data
            assert "method" in ocr_data

            if ocr_data["status"]:
                assert "data" in ocr_data
                assert "confidence" in ocr_data
                assert "details" in ocr_data

        finally:
            # 清理：關閉 API 服務
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()

    def test_handler_discovery_consistency(self, cli_path: Path, temp_dir: Path, venv_python: Path):
        """測試 handler 發現的一致性"""

        # 1. 初始化 handlers
        handlers_dir = temp_dir / "handlers"
        result = subprocess.run([
            str(cli_path), "init",
            "--output-dir", str(handlers_dir)
        ], capture_output=True, text=True)

        assert result.returncode == 0

        # 2. 使用 Python 直接測試 handler 發現
        discovery_script = f"""
from captcha_ocr_devkit.core.handlers.registry import registry
from pathlib import Path

discovered = registry.discover_handlers(Path('{handlers_dir}'))
print('Discovered handlers:', discovered)

# 測試每種類型的 handler 創建
for handler_type in ['preprocess', 'train', 'evaluate', 'ocr']:
    if f'Demo{{handler_type.title()}}Handler' in discovered[handler_type]:
        handler = registry.create_handler(handler_type, f'Demo{{handler_type.title()}}Handler')
        print(f'Created {{handler_type}} handler:', handler.name)
        print(f'Handler info:', handler.get_info())
"""

        result = subprocess.run([
            str(venv_python), "-c", discovery_script
        ], capture_output=True, text=True)

        assert result.returncode == 0, f"Handler discovery failed: {result.stderr}"
        assert "DemoPreprocessHandler" in result.stdout
        assert "DemoTrainHandler" in result.stdout
        assert "DemoEvaluateHandler" in result.stdout
        assert "DemoOCRHandler" in result.stdout

    def test_error_handling_workflow(self, cli_path: Path, temp_dir: Path):
        """測試錯誤處理工作流程"""

        # 測試不存在的輸入目錄
        result = subprocess.run([
            str(cli_path), "train",
            "--input", str(temp_dir / "nonexistent"),
            "--output", str(temp_dir / "model.json"),
            "--handler", "DemoTrainHandler"
        ], capture_output=True, text=True)

        assert result.returncode != 0

        # 測試不存在的模型檔案
        result = subprocess.run([
            str(cli_path), "evaluate",
            "--target", str(temp_dir),
            "--model", str(temp_dir / "nonexistent.json"),
            "--handler", "DemoEvaluateHandler"
        ], capture_output=True, text=True)

        assert result.returncode != 0

        # 測試不存在的模型檔案啟動 API
        result = subprocess.run([
            str(cli_path), "api",
            "--model", str(temp_dir / "nonexistent.json"),
            "--port", "54399"
        ], capture_output=True, text=True)

        assert result.returncode != 0

    @pytest.mark.slow
    def test_handler_interoperability(self, cli_path: Path, temp_dir: Path):
        """測試不同 handler 的互操作性"""

        # 準備環境
        handlers_dir = temp_dir / "handlers"
        subprocess.run([
            str(cli_path), "init",
            "--output-dir", str(handlers_dir)
        ], capture_output=True, text=True)

        images_dir = temp_dir / "images"
        images_dir.mkdir()
        for i in range(3):
            (images_dir / f"test_{i:03d}.png").write_text(f"fake image {i}")

        # 使用 DemoTrainHandler 訓練
        model_file = temp_dir / "model.json"
        result = subprocess.run([
            str(cli_path), "train",
            "--input", str(images_dir),
            "--output", str(model_file),
            "--handler", "DemoTrainHandler",
            "--epochs", "1"
        ], capture_output=True, text=True, cwd=handlers_dir.parent)

        assert result.returncode == 0

        # 使用 DemoEvaluateHandler 評估（不同的 handler）
        result = subprocess.run([
            str(cli_path), "evaluate",
            "--target", str(images_dir),
            "--model", str(model_file),
            "--handler", "DemoEvaluateHandler"
        ], capture_output=True, text=True, cwd=handlers_dir.parent)

        assert result.returncode == 0

        # 若當前環境無法開啟 TCP 連線（例如沙箱環境禁止 bind），則跳過 API 互通測試
        probe_socket = None
        try:
            probe_socket = socket.socket()
            probe_socket.bind(("127.0.0.1", 0))
        except OSError:
            pytest.skip("Local TCP binding is not permitted in this environment")
        finally:
            try:
                if probe_socket:
                    probe_socket.close()
            except Exception:
                pass

        # 使用 DemoOCRHandler 提供 API 服務（又是不同的 handler）
        api_port = 54397
        api_process = subprocess.Popen([
            str(cli_path), "api",
            "--model", str(model_file),
            "--port", str(api_port),
            "--handler", "DemoOCRHandler"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=handlers_dir.parent)

        try:
            time.sleep(3)

            # 驗證 API 可以使用訓練的模型
            response = requests.get(f"http://localhost:{api_port}/api/v1/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                assert health_data["model_loaded"] is True

        except requests.RequestException:
            # 如果連接失敗，檢查 API 進程是否還在運行
            if api_process.poll() is not None:
                # 進程已經結束，檢查錯誤
                stdout, stderr = api_process.communicate()
                pytest.fail(f"API process failed: {stderr}")

        finally:
            api_process.terminate()
            try:
                api_process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                api_process.kill()


@pytest.mark.integration
@pytest.mark.slow
class TestPerformanceIntegration:
    """效能整合測試"""

    def test_large_dataset_simulation(self, cli_path: Path, temp_dir: Path):
        """測試大資料集模擬"""

        # 初始化
        handlers_dir = temp_dir / "handlers"
        subprocess.run([
            str(cli_path), "init",
            "--output-dir", str(handlers_dir)
        ], capture_output=True, text=True)

        # 創建較大的測試資料集
        images_dir = temp_dir / "large_dataset"
        images_dir.mkdir()

        labels = ["abcd", "efgh", "ijkl", "mnop", "qrst"]
        total_images = 50

        for i in range(total_images):
            label = labels[i % len(labels)]
            filename = f"{label}_{i:04d}.png"
            (images_dir / filename).write_text(f"fake image data {i}")

        # 訓練測試
        model_file = temp_dir / "large_model.json"
        start_time = time.time()

        result = subprocess.run([
            str(cli_path), "train",
            "--input", str(images_dir),
            "--output", str(model_file),
            "--handler", "DemoTrainHandler",
            "--epochs", "1"
        ], capture_output=True, text=True, cwd=handlers_dir.parent)

        training_time = time.time() - start_time

        assert result.returncode == 0
        assert model_file.exists()

        # 訓練時間應該合理（Demo handler 應該很快）
        assert training_time < 10.0, f"Training took too long: {training_time}s"

        # 檢查模型檔案內容
        with open(model_file) as f:
            model_data = json.load(f)

        assert model_data["dataset_info"]["total_images"] == total_images
        assert len(model_data["dataset_info"]["sample_labels"]) == len(labels)

    def test_api_load_simulation(self, cli_path: Path, temp_dir: Path):
        """測試 API 負載模擬"""

        # 準備環境
        handlers_dir = temp_dir / "handlers"
        subprocess.run([
            str(cli_path), "init",
            "--output-dir", str(handlers_dir)
        ], capture_output=True, text=True)

        # 使用現有的測試模型
        project_root = temp_dir.parent.parent
        test_model = project_root / "test_model.json"

        if not test_model.exists():
            pytest.skip("Test model not available")

        # 啟動 API
        api_port = 54396
        api_process = subprocess.Popen([
            str(cli_path), "api",
            "--model", str(test_model),
            "--port", str(api_port),
            "--handler", "DemoOCRHandler"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=handlers_dir.parent)

        try:
            time.sleep(3)

            # 發送多個請求測試負載
            import concurrent.futures

            def make_request(i):
                try:
                    files = {"file": (f"test_{i}.png", b"fake image", "image/png")}
                    response = requests.post(
                        f"http://localhost:{api_port}/api/v1/ocr",
                        files=files,
                        timeout=5
                    )
                    return response.status_code == 200, response.json() if response.status_code == 200 else None
                except Exception as e:
                    return False, str(e)

            # 並發發送 10 個請求
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(make_request, i) for i in range(10)]
                results = [future.result() for future in futures]

            # 檢查結果
            successful_requests = sum(1 for success, _ in results if success)
            assert successful_requests >= 8, f"Too many failed requests: {successful_requests}/10"

        finally:
            api_process.terminate()
            try:
                api_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                api_process.kill()
