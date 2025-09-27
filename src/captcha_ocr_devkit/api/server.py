"""
FastAPI 服務器 v2.0
基於 Handler 架構的 CAPTCHA OCR API 服務
"""

import os
import json
import time
import base64
import logging
import string
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

from captcha_ocr_devkit import __version__ as CORE_VERSION

from .schemas import (
    OCRResponse,
    HealthResponse,
    APIStatsResponse,
    HealthStatus
)

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 預設的 CAPTCHA 解析設定
DEFAULT_CAPTCHA_LENGTH = int(os.getenv("CAPTCHA_DEFAULT_LENGTH", "4"))
DEFAULT_CAPTCHA_TYPE = os.getenv("CAPTCHA_DEFAULT_TYPE", "lowercase").strip().lower()
DEFAULT_SEGMENTATION_METHOD = os.getenv("CAPTCHA_DEFAULT_SEGMENTATION", "auto").strip().lower()
ENFORCE_LOWERCASE = os.getenv("CAPTCHA_ENFORCE_LOWERCASE", "true").strip().lower() != "false"

_CAPTCHA_TYPE_ALIASES = {
    "lower": "lowercase",
    "lowercase": "lowercase",
    "letters": "alphabetic",
    "alpha": "alphabetic",
    "alphabet": "alphabetic",
    "alphabetic": "alphabetic",
    "alphanumeric": "alphanumeric",
    "mixed": "alphanumeric",
    "number": "numeric",
    "numbers": "numeric",
    "numeric": "numeric",
    "digit": "numeric",
    "digits": "numeric",
}

_CAPTCHA_TYPE_CHARSETS = {
    "lowercase": string.ascii_lowercase,
    "alphabetic": string.ascii_letters,
    "alphanumeric": string.ascii_letters + string.digits,
    "numeric": string.digits,
}

_ALLOWED_SEGMENTATION_METHODS = {"auto", "projection", "connected"}


def _parse_positive_int(value: Optional[Any]) -> Optional[int]:
    """嘗試解析正整數，若失敗則返回 None。"""

    if value is None:
        return None

    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None

    return parsed if parsed > 0 else None


def _normalize_captcha_type(raw_type: Optional[str]) -> str:
    """正規化來自請求的 captcha_type。"""

    if not raw_type:
        return DEFAULT_CAPTCHA_TYPE

    normalized = raw_type.strip().lower()
    return _CAPTCHA_TYPE_ALIASES.get(normalized, normalized if normalized in _CAPTCHA_TYPE_CHARSETS else DEFAULT_CAPTCHA_TYPE)


def _normalize_segmentation_method(raw_method: Optional[str]) -> str:
    """正規化 segmentation_method，確保落在允許範圍。"""

    if not raw_method:
        return DEFAULT_SEGMENTATION_METHOD

    normalized = raw_method.strip().lower()
    if normalized in _ALLOWED_SEGMENTATION_METHODS:
        return normalized
    return DEFAULT_SEGMENTATION_METHOD


def _resolve_captcha_settings(
    form_password: Optional[str],
    form_captcha_type: Optional[str],
    form_segmentation_method: Optional[str],
    payload: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """萃取請求欲使用的 CAPTCHA 設定，若缺漏則套用預設值。"""

    payload = payload or {}

    length_candidates = [form_password]
    for key in ("captcha_length", "captchaLength", "length", "password"):
        if key in payload:
            length_candidates.append(payload[key])
            break

    captcha_length = DEFAULT_CAPTCHA_LENGTH
    for candidate in length_candidates:
        parsed = _parse_positive_int(candidate)
        if parsed is not None:
            captcha_length = parsed
            break

    captcha_type_candidates = [form_captcha_type]
    for key in ("captcha_type", "captchaType", "type"):
        if isinstance(payload.get(key), str):
            captcha_type_candidates.append(payload[key])
            break

    captcha_type = DEFAULT_CAPTCHA_TYPE
    if not ENFORCE_LOWERCASE:
        for candidate in captcha_type_candidates:
            captcha_type = _normalize_captcha_type(candidate)
            if captcha_type:
                break

    if ENFORCE_LOWERCASE:
        captcha_type = DEFAULT_CAPTCHA_TYPE

    segmentation_candidates = [form_segmentation_method]
    for key in ("segmentation_method", "segmentationMethod", "segmentation"):
        if isinstance(payload.get(key), str):
            segmentation_candidates.append(payload[key])
            break

    segmentation_method = DEFAULT_SEGMENTATION_METHOD
    for candidate in segmentation_candidates:
        segmentation_method = _normalize_segmentation_method(candidate)
        if segmentation_method:
            break

    charset = _CAPTCHA_TYPE_CHARSETS.get(captcha_type, _CAPTCHA_TYPE_CHARSETS[DEFAULT_CAPTCHA_TYPE])

    return {
        "captcha_length": captcha_length,
        "captcha_type": captcha_type,
        "captcha_charset": charset,
        "segmentation_method": segmentation_method,
    }


def _enforce_prediction_constraints(prediction: Optional[str], settings: Dict[str, Any]) -> str:
    """若預測結果不符合設定，則觸發驗證錯誤。"""

    expected_length = settings.get("captcha_length", DEFAULT_CAPTCHA_LENGTH)
    captcha_type = settings.get("captcha_type", DEFAULT_CAPTCHA_TYPE)
    allowed_charset = settings.get("captcha_charset") or _CAPTCHA_TYPE_CHARSETS.get(DEFAULT_CAPTCHA_TYPE)

    normalized = (prediction or "").strip()
    if captcha_type == "lowercase":
        normalized = normalized.lower()

    if len(normalized) != expected_length:
        raise HTTPException(
            status_code=422,
            detail=f"預測字元數為 {len(normalized)}，與預期的 {expected_length} 不符"
        )

    disallowed = [ch for ch in normalized if allowed_charset and ch not in allowed_charset]
    if disallowed:
        raise HTTPException(
            status_code=422,
            detail="預測字元包含不支援的字元: " + ",".join(sorted(set(disallowed)))
        )

    return normalized


class APIStats:
    """統計資料管理"""

    def __init__(self):
        self.reset_stats()

    def reset_stats(self):
        """重置統計資料"""
        self.total_requests = 0
        self.ocr_requests = 0
        self.generate_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.processing_times = []
        self.start_time = time.time()

    def record_request(
        self,
        processing_time: float,
        success: bool = True,
        request_type: str = "ocr",
    ):
        """記錄請求"""
        self.total_requests += 1

        request_type = request_type.lower()
        if request_type == "ocr":
            self.ocr_requests += 1
        elif request_type == "generate":
            self.generate_requests += 1

        if success:
            self.successful_requests += 1
        else:
            self.failed_requests += 1

        self.processing_times.append(processing_time)

        # 只保留最近 1000 次記錄
        if len(self.processing_times) > 1000:
            self.processing_times = self.processing_times[-1000:]

    def get_stats(self) -> Dict[str, Any]:
        """獲取統計資料"""
        uptime = time.time() - self.start_time
        success_rate = self.successful_requests / self.total_requests if self.total_requests > 0 else 0
        avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
        requests_per_minute = (self.total_requests / uptime) * 60 if uptime > 0 else 0

        return {
            "total_requests": self.total_requests,
            "ocr_requests": self.ocr_requests,
            "generate_requests": self.generate_requests,
            "success_rate": success_rate,
            "average_processing_time": avg_processing_time,
            "uptime": uptime,
            "requests_per_minute": requests_per_minute
        }


class HandlerManager:
    """
Handler 管理器

使用新的 Handler 架構來管理 OCR 功能
"""

    def __init__(self):
        self.pipeline = None
        self.model_path = None
        self.model_loaded = False
        self.handlers_info = {}

    async def initialize(self, model_path: str, ocr_handler: str, preprocess_handler: Optional[str] = None):
        """初始化 pipeline 和 handlers"""
        try:
            logger.info(f"🚀 初始化 Handler Manager")
            logger.info(f"🤖 模型路徑: {model_path}")
            logger.info(f"🔧 OCR Handler: {ocr_handler}")
            if preprocess_handler:
                logger.info(f"🖼️ Preprocess Handler: {preprocess_handler}")

            # 載入 pipeline 模組
            from ..core.pipeline import create_pipeline_from_handlers
            from ..core.handlers.registry import registry

            # 發現 handlers
            discovered = registry.discover_handlers()
            logger.info(f"🔍 發現的 handlers: {discovered}")

            handler_configs_env = os.getenv('CAPTCHA_HANDLER_CONFIGS', '')
            handler_configs: Dict[str, Dict[str, Any]] = {}
            if handler_configs_env:
                try:
                    handler_configs = json.loads(handler_configs_env)
                except json.JSONDecodeError as exc:
                    logger.warning(f"⚠️  無法解析 CAPTCHA_HANDLER_CONFIGS: {exc}")

            # 創建 pipeline
            self.pipeline = create_pipeline_from_handlers(
                preprocess_handler=preprocess_handler,
                ocr_handler=ocr_handler,
                handler_configs=handler_configs,
            )

            # 載入 OCR 模型
            ocr_handler_instance = self.pipeline.get_handler('ocr')
            if ocr_handler_instance:
                model_load_success = ocr_handler_instance.load_model(Path(model_path))
                if not model_load_success:
                    raise Exception("OCR 模型載入失敗")

            self.model_path = model_path
            self.model_loaded = True

            # 獲取 handler 資訊
            self.handlers_info = self.pipeline.get_pipeline_info()

            logger.info("✅ Handler Manager 初始化成功")

        except Exception as e:
            logger.error(f"❌ Handler Manager 初始化失敗: {e}")
            self.model_loaded = False
            raise HTTPException(status_code=500, detail=f"Handler 初始化失敗: {str(e)}")

    def is_ready(self) -> bool:
        """檢查是否就緒"""
        return self.model_loaded and self.pipeline is not None

    async def predict_image(
        self,
        image_bytes: bytes,
        request_details: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """使用 pipeline 進行圖片預測"""
        if not self.is_ready():
            raise HTTPException(status_code=503, detail="Handler 尚未初始化")

        try:
            # 使用 pipeline 處理圖片
            result = self.pipeline.process_image(image_bytes)

            if not result.success:
                raise HTTPException(status_code=400, detail=f"圖片處理失敗: {result.error}")

            settings = request_details or {
                "captcha_length": DEFAULT_CAPTCHA_LENGTH,
                "captcha_type": DEFAULT_CAPTCHA_TYPE,
                "captcha_charset": _CAPTCHA_TYPE_CHARSETS.get(DEFAULT_CAPTCHA_TYPE),
                "segmentation_method": DEFAULT_SEGMENTATION_METHOD,
            }

            enforced_text = _enforce_prediction_constraints(result.data, settings)

            # 基於 ocr_4_chars 格式的回應結構
            warnings = []

            # 基本數據
            confidence_raw = result.metadata.get("confidence", 0.0) if result.metadata else 0.0
            # 轉換信心度到 0-100 範圍 (如果原本是 0-1)
            confidence = confidence_raw * 100 if confidence_raw <= 1.0 else confidence_raw

            processing_time = result.metadata.get("processing_time", 0.0) if result.metadata else 0.0
            character_confidences = []
            image_size = None

            # 檢查並提取 metadata
            if result.metadata:
                if "image_size" in result.metadata:
                    image_size = result.metadata["image_size"]
                else:
                    warnings.append("圖片尺寸信息缺失")

                if "character_confidences" in result.metadata:
                    char_confs = result.metadata["character_confidences"]
                    # 確保字元信心度也在 0-100 範圍
                    character_confidences = [c * 100 if c <= 1.0 else c for c in char_confs]
                else:
                    warnings.append("字元信心度信息缺失")
            else:
                warnings.extend([
                    "圖片尺寸信息缺失",
                    "字元信心度信息缺失",
                    "預處理信息缺失"
                ])

            # 決定元數據完整性狀態
            if len(warnings) == 0:
                metadata_completeness = "full"
            elif len(warnings) <= 2:
                metadata_completeness = "partial"
            else:
                metadata_completeness = "minimal"

            # 生成時間戳
            timestamp = datetime.now().isoformat()

            # 取得 pipeline 信息
            pipeline_info = self.pipeline.get_pipeline_info()
            handler_versions = {}
            for handler_type, meta in pipeline_info.get("handlers", {}).items():
                info = meta.get("info") or {}
                version = info.get("version")
                if version:
                    handler_versions[handler_type] = str(version)

            # 構建成功回應 (參考 api_server.py 格式)
            response_data = {
                "status": True,
                "data": enforced_text,
                "confidence": float(confidence),
                "processing_time": float(processing_time),
                "timestamp": timestamp,
                "method": "Handler Pipeline OCR",
                "core_version": CORE_VERSION,
                "handler_versions": handler_versions,
                "details": {
                    "character_confidences": [float(c) for c in character_confidences],
                    "character_count": len(enforced_text),
                    "image_size": image_size,
                    "handler_info": {
                        "preprocess_handler": pipeline_info.get("config", {}).get("preprocess_handler"),
                        "ocr_handler": pipeline_info.get("config", {}).get("ocr_handler")
                    },
                    "handler_versions": handler_versions,
                    "warnings": warnings,
                    "metadata_completeness": metadata_completeness
                }
            }

            details: Dict[str, Any] = response_data.get("details") or {}
            if request_details:
                for key, value in request_details.items():
                    if value is None:
                        continue
                    details.setdefault(key, value)
            response_data["details"] = details

            return response_data

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"圖片預測失敗: {e}")
            raise HTTPException(status_code=400, detail=f"圖片處理失敗: {str(e)}")

    def get_info(self) -> Dict[str, Any]:
        """獲取 manager 資訊"""
        handler_versions: Dict[str, str] = {}
        if self.pipeline:
            for handler_type, handler in self.pipeline.handlers.items():
                info = handler.get_info() or {}
                version = info.get("version")
                if version:
                    handler_versions[handler_type] = str(version)

        return {
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "handlers_info": self.handlers_info,
            "pipeline_ready": self.is_ready(),
            "handler_versions": handler_versions
        }


# 創建全域實例
handler_manager = HandlerManager()
api_stats = APIStats()

_HTML_TEMPLATE_PATH = Path(__file__).with_name("demo_index.html")

try:
    INDEX_HTML = _HTML_TEMPLATE_PATH.read_text(encoding="utf-8")
except FileNotFoundError:
    INDEX_HTML = """<!DOCTYPE html><html><head><meta charset='utf-8'><title>CAPTCHA API</title></head><body><h1>CAPTCHA OCR API</h1><p>缺少 demo_index.html 模板文件。</p></body></html>"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown without deprecated events."""

    logger.info("🎆 啟動 CAPTCHA OCR API v2.0 服務...")

    model_path = os.getenv("CAPTCHA_MODEL_PATH")
    ocr_handler = os.getenv("CAPTCHA_OCR_HANDLER")
    preprocess_handler = os.getenv("CAPTCHA_PREPROCESS_HANDLER")

    should_initialize = True
    if not model_path:
        logger.warning("⚠️ 未設定 CAPTCHA_MODEL_PATH 環境變數")
        should_initialize = False
    elif not os.path.exists(model_path):
        logger.warning(f"⚠️ 模型檔案不存在: {model_path}")
        should_initialize = False

    if not ocr_handler:
        logger.warning("⚠️ 未設定 CAPTCHA_OCR_HANDLER 環境變數")
        should_initialize = False

    if should_initialize:
        try:
            await handler_manager.initialize(model_path, ocr_handler, preprocess_handler)
            logger.info("🎉 API 服務啟動成功")
        except Exception as exc:
            logger.error(f"❌ API 服務啟動失敗: {exc}")

    try:
        yield
    finally:
        logger.info("📤 關閉 CAPTCHA OCR API v2.0 服務...")


# 創建 FastAPI 應用
app = FastAPI(
    title="CAPTCHA OCR API v2.0",
    description="基於 Handler 架構的插件化 CAPTCHA OCR API 服務",
    version=CORE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# 添加 CORS 中間件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 依賴函數
async def get_handler_manager() -> HandlerManager:
    """獲取 Handler 管理器"""
    return handler_manager


async def record_api_call(processing_time: float, success: bool = True, request_type: str = "ocr"):
    """記錄 API 調用"""
    api_stats.record_request(processing_time, success, request_type=request_type)


# API 端點

@app.get("/", response_class=HTMLResponse)
async def root():
    """根端點 - 提供 Demo 網頁"""
    return HTMLResponse(content=INDEX_HTML)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health_check():
    """健康檢查端點"""
    manager = handler_manager

    uptime = time.time() - api_stats.start_time
    manager_info = manager.get_info()

    return HealthResponse(
        status=HealthStatus.healthy if manager.is_ready() else HealthStatus.unhealthy,
        model_loaded=manager.model_loaded,
        version=CORE_VERSION,
        handler_versions=manager_info.get("handler_versions", {}),
        uptime=uptime,
        model_info=manager_info.get("handlers_info", {})
    )


def _health_like_response(manager: HandlerManager) -> OCRResponse:
    manager_info = manager.get_info()
    return OCRResponse(
        status=True,
        data=None,
        confidence=None,
        processing_time=0.0,
        timestamp=datetime.now().isoformat(),
        method="Handler Pipeline OCR",
        core_version=CORE_VERSION,
        handler_versions=manager_info.get("handler_versions"),
        details={
            "handlers": manager_info.get("handlers_info"),
            "status": "healthy" if manager.is_ready() else "initializing",
        },
        message="GET request received. Returning health-style response.",
    )


@app.api_route("/api/v1/ocr", methods=["GET", "POST"], response_model=OCRResponse)
async def ocr_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: Optional[UploadFile] = File(None),
    password: Optional[str] = Form(None),
    captcha_type: Optional[str] = Form(None),
    segmentation_method: Optional[str] = Form(None)
):
    """
    OCR 圖片辨識端點

    上傳圖片檔案，使用配置的 handlers 進行辨識
    """
    manager = handler_manager

    if request.method.upper() == "GET":
        return _health_like_response(manager)

    start_time = time.time()

    try:
        image_bytes: Optional[bytes] = None
        payload: Optional[Dict[str, Any]] = None
        if file is None:
            content_type = request.headers.get("content-type", "").lower()
            if "application/json" in content_type:
                try:
                    payload = await request.json()
                except Exception:
                    payload = None

        if file is None and payload:
            image_base64 = payload.get("image_base64") or payload.get("image")
            if not image_base64:
                processing_time = time.time() - start_time
                background_tasks.add_task(record_api_call, processing_time, False)
                return JSONResponse(
                    status_code=400,
                    content=OCRResponse(
                        status=False,
                        message="JSON 請求需要 image 或 image_base64 欄位",
                        processing_time=processing_time,
                        timestamp=datetime.now().isoformat(),
                        method="Handler Pipeline OCR",
                        core_version=CORE_VERSION
                    ).model_dump()
                )
            try:
                image_bytes = base64.b64decode(image_base64)
            except Exception:
                processing_time = time.time() - start_time
                background_tasks.add_task(record_api_call, processing_time, False)
                return JSONResponse(
                    status_code=400,
                    content=OCRResponse(
                        status=False,
                        message="image_base64 解碼失敗",
                        processing_time=processing_time,
                        timestamp=datetime.now().isoformat(),
                        method="Handler Pipeline OCR",
                        core_version=CORE_VERSION
                    ).model_dump()
                )
        elif file is not None:
            if not file.content_type or not file.content_type.startswith('image/'):
                processing_time = time.time() - start_time
                background_tasks.add_task(record_api_call, processing_time, False)
                return JSONResponse(
                    status_code=400,
                    content=OCRResponse(
                        status=False,
                        message="請上傳圖片檔案",
                        processing_time=processing_time,
                        timestamp=datetime.now().isoformat(),
                        method="Handler Pipeline OCR",
                        core_version=CORE_VERSION
                    ).model_dump()
                )
            image_bytes = await file.read()
            if len(image_bytes) == 0:
                processing_time = time.time() - start_time
                background_tasks.add_task(record_api_call, processing_time, False)
                return JSONResponse(
                    status_code=400,
                    content=OCRResponse(
                        status=False,
                        message="圖片檔案為空",
                        processing_time=processing_time,
                        timestamp=datetime.now().isoformat(),
                        method="Handler Pipeline OCR",
                        core_version=CORE_VERSION
                    ).model_dump()
                )
        else:
            processing_time = time.time() - start_time
            background_tasks.add_task(record_api_call, processing_time, False)
            return JSONResponse(
                status_code=400,
                content=OCRResponse(
                    status=False,
                    message="請透過 multipart/form-data 上傳圖片或提供 image/image_base64",
                    processing_time=processing_time,
                    timestamp=datetime.now().isoformat(),
                    method="Handler Pipeline OCR",
                    core_version=CORE_VERSION
                ).model_dump()
            )

        request_settings = _resolve_captcha_settings(
            form_password=password,
            form_captcha_type=captcha_type,
            form_segmentation_method=segmentation_method,
            payload=payload
        )

        result = await manager.predict_image(
            image_bytes,
            request_details=request_settings
        )

        processing_time = time.time() - start_time
        background_tasks.add_task(record_api_call, processing_time, True)

        return OCRResponse(**result)

    except HTTPException as e:
        # 處理 HTTP 異常
        processing_time = time.time() - start_time
        background_tasks.add_task(record_api_call, processing_time, False)

        return OCRResponse(
            status=False,
            message=str(e.detail),
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            method="Handler Pipeline OCR",
            core_version=CORE_VERSION
        )

    except Exception as e:
        # 處理一般異常
        processing_time = time.time() - start_time
        background_tasks.add_task(record_api_call, processing_time, False)
        logger.error(f"OCR 處理失敗: {e}")

        return OCRResponse(
            status=False,
            message=f"圖片處理失敗: {str(e)}",
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            method="Handler Pipeline OCR",
            core_version=CORE_VERSION
        )


@app.get("/api/v1/handlers/info")
async def get_handlers_info():
    """獲取 Handler 資訊端點"""
    manager = handler_manager

    if not manager.is_ready():
        raise HTTPException(status_code=503, detail="Handler 尚未初始化")

    return manager.get_info()


@app.get("/api/v1/stats", response_model=APIStatsResponse)
async def get_api_stats():
    """獲取 API 統計資訊端點"""
    stats = api_stats.get_stats()
    return APIStatsResponse(**stats)


@app.post("/api/v1/stats/reset")
async def reset_api_stats():
    """重置 API 統計資訊端點"""
    api_stats.reset_stats()
    return {"message": "統計資料已重置"}


# 錯誤處理器
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 錯誤處理"""
    return JSONResponse(
        status_code=404,
        content={
            "detail": "端點未找到",
            "error_code": "NOT_FOUND",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(422)
async def validation_exception_handler(request, exc):
    """驗證錯誤處理"""
    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.detail,
            "error_code": "VALIDATION_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    """500 錯誤處理"""
    return JSONResponse(
        status_code=500,
        content={
            "detail": "內部服務器錯誤",
            "error_code": "INTERNAL_ERROR",
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=54321,
        reload=True,
        log_level="info"
    )
