"""
API 端點單元測試
"""
import json
from io import BytesIO

import pytest

from captcha_ocr_devkit import __version__ as CORE_VERSION


class TestAPIHealth:
    """API 健康檢查端點測試"""

    def test_root_endpoint(self, api_client):
        """測試根端點"""
        response = api_client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")
        html = response.text
        assert "CAPTCHA API 測試工具" in html
        assert "識別驗證碼" in html

    def test_health_endpoint(self, api_client):
        """測試健康檢查端點"""
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data
        assert data["version"] == CORE_VERSION
        assert "uptime" in data
        assert "handler_versions" in data

        # 檢查資料類型
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime"], (int, float))
        assert isinstance(data["handler_versions"], dict)

    def test_handlers_info_endpoint(self, api_client):
        """測試 handlers 資訊端點"""
        response = api_client.get("/api/v1/handlers/info")

        # 如果 handler 未初始化，可能回傳 503
        if response.status_code == 503:
            pytest.skip("Handler not initialized, skipping test")

        assert response.status_code == 200
        data = response.json()

        assert "model_loaded" in data
        assert "pipeline_ready" in data
        assert "handlers_info" in data

        # 檢查 handlers_info 結構
        handlers_info = data["handlers_info"]
        assert "config" in handlers_info
        assert "handlers" in handlers_info

    def test_stats_endpoint(self, api_client):
        """測試統計資訊端點"""
        response = api_client.get("/api/v1/stats")
        assert response.status_code == 200

        data = response.json()
        expected_fields = [
            "total_requests",
            "ocr_requests",
            "generate_requests",
            "success_rate",
            "average_processing_time",
            "uptime",
            "requests_per_minute",
        ]

        for field in expected_fields:
            assert field in data
            assert isinstance(data[field], (int, float))

    def test_stats_reset_endpoint(self, api_client):
        """測試統計重置端點"""
        response = api_client.post("/api/v1/stats/reset")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "統計資料已重置" in data["message"]


class TestAPIErrorHandling:
    """API 錯誤處理測試"""

    def test_404_endpoint(self, api_client):
        """測試 404 錯誤處理"""
        response = api_client.get("/nonexistent")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data
        assert "error_code" in data
        assert "timestamp" in data
        assert data["error_code"] == "NOT_FOUND"

    def test_ocr_endpoint_no_file(self, api_client):
        """測試 OCR 端點無檔案"""
        response = api_client.post("/api/v1/ocr")
        assert response.status_code == 400
        data = response.json()
        assert data.get("status") is False
        assert "image" in data.get("message", "")

    def test_ocr_endpoint_invalid_file(self, api_client):
        """測試 OCR 端點無效檔案"""
        # 發送非圖片檔案
        response = api_client.post(
            "/api/v1/ocr",
            files={"file": ("test.txt", "not an image", "text/plain")}
        )

        assert response.status_code == 400  # API 回傳結構化錯誤
        data = response.json()

        assert "status" in data
        assert data["status"] is False
        assert "message" in data
        assert "processing_time" in data
        assert "timestamp" in data


class TestAPIOCR:
    """API OCR 端點測試"""

    def test_ocr_endpoint_with_fake_image(self, api_client, fake_image_bytes):
        """測試 OCR 端點使用假圖片"""
        response = api_client.post(
            "/api/v1/ocr",
            files={"file": ("test.png", fake_image_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()

        # 檢查基本 OCR 回應結構
        assert "status" in data
        assert "processing_time" in data
        assert "timestamp" in data
        assert "method" in data
        assert data.get("core_version") == CORE_VERSION

        if data["status"]:
            # 成功回應結構
            assert "data" in data
            assert "confidence" in data
            assert "handler_versions" in data
            assert isinstance(data["handler_versions"], dict)
            assert "details" in data

            # 檢查 details 結構
            details = data["details"]
            assert "character_count" in details
            assert "handler_info" in details
            assert "warnings" in details
            assert "metadata_completeness" in details

            # 檢查 handler_info
            handler_info = details["handler_info"]
            assert "ocr_handler" in handler_info

            # 檢查數值範圍
            if data["confidence"] is not None:
                assert 0 <= data["confidence"] <= 100

        else:
            # 失敗回應結構
            assert "message" in data

    def test_ocr_endpoint_empty_file(self, api_client):
        """測試 OCR 端點空檔案"""
        response = api_client.post(
            "/api/v1/ocr",
            files={"file": ("empty.png", b"", "image/png")}
        )

        assert response.status_code == 400
        data = response.json()

        assert data["status"] is False
        assert "圖片檔案為空" in data["message"]
        assert data.get("core_version") == CORE_VERSION

    def test_ocr_response_format_consistency(self, api_client, fake_image_bytes, sample_ocr_response):
        """測試 OCR 回應格式一致性"""
        response = api_client.post(
            "/api/v1/ocr",
            files={"file": ("test.png", fake_image_bytes, "image/png")}
        )

        assert response.status_code == 200
        data = response.json()

        # 檢查必需欄位存在
        required_fields = ["status", "processing_time", "timestamp", "method", "core_version"]
        for field in required_fields:
            assert field in data

        if data["status"]:
            # 成功回應額外欄位
            success_fields = ["data", "confidence", "details", "handler_versions"]
            for field in success_fields:
                if field in sample_ocr_response:
                    assert field in data

            # 檢查 details 結構與範例一致
            if "details" in data:
                details = data["details"]
                expected_detail_fields = [
                    "character_count", "handler_info",
                    "warnings", "metadata_completeness"
                ]
                for field in expected_detail_fields:
                    assert field in details

        else:
            # 失敗回應必須有 message
            assert "message" in data


class TestAPIPerformance:
    """API 效能相關測試"""

    def test_response_time_reasonable(self, api_client, fake_image_bytes):
        """測試回應時間合理性"""
        import time

        start_time = time.time()
        response = api_client.post(
            "/api/v1/ocr",
            files={"file": ("test.png", fake_image_bytes, "image/png")}
        )
        end_time = time.time()

        assert response.status_code == 200
        actual_time = end_time - start_time

        # 實際時間應該在 5 秒內（很寬鬆的限制）
        assert actual_time < 5.0

        # 檢查回應中的 processing_time
        data = response.json()
        if "processing_time" in data:
            reported_time = data["processing_time"]
            # 回報的處理時間應該 <= 實際時間
            assert reported_time <= actual_time

    def test_concurrent_requests_basic(self, api_client, fake_image_bytes):
        """測試基本並發請求"""
        import concurrent.futures
        import threading

        def make_request():
            return api_client.post(
                "/api/v1/ocr",
                files={"file": ("test.png", fake_image_bytes, "image/png")}
            )

        # 發送 3 個並發請求
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(make_request) for _ in range(3)]
            responses = [future.result() for future in futures]

        # 所有請求都應該成功
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert "status" in data
