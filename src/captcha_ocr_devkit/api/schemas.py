"""
API 資料結構定義
定義 FastAPI 的請求和回應模型
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CaptchaStyle(str, Enum):
    """CAPTCHA 樣式枚舉"""

    simple = "simple"
    standard = "standard"
    complex = "complex"


class HealthStatus(str, Enum):
    """健康狀態枚舉"""

    healthy = "healthy"
    unhealthy = "unhealthy"


# OCR 相關模型
class OCRRequest(BaseModel):
    """OCR 請求模型"""
    # 圖片將通過 multipart/form-data 上傳，這裡不需要定義


class OCRResponse(BaseModel):
    """OCR 回應模型 - 基於 ocr_4_chars API 格式"""

    status: bool = Field(..., description="處理狀態 (true=成功, false=失敗)")
    data: Optional[str] = Field(default=None, description="辨識出的文字")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=100.0, description="平均信心度 (0-100)")
    processing_time: float = Field(..., ge=0.0, description="處理時間（秒）")
    timestamp: str = Field(..., description="處理時間戳")
    method: str = Field(default="Handler Pipeline OCR", description="處理方法")
    core_version: Optional[str] = Field(default=None, description="captcha_ocr_devkit 核心版本")
    handler_versions: Optional[Dict[str, str]] = Field(default=None, description="各 handler 版本資訊")
    details: Optional[Dict[str, Any]] = Field(default=None, description="詳細信息")
    message: Optional[str] = Field(default=None, description="錯誤訊息")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": True,
                "data": "abcd",
                "confidence": 95.3,
                "processing_time": 0.123,
                "timestamp": "2024-01-01T12:00:00.000Z",
                "method": "Handler Pipeline OCR",
                "core_version": "1.20250919.1645",
                "handler_versions": {
                    "ocr": "1.20250919.1640"
                },
                "details": {
                    "character_confidences": [98.1, 94.2, 96.8, 92.4],
                    "character_count": 4,
                    "image_size": "128x64",
                    "handler_info": {
                        "preprocess_handler": "demo_preprocess",
                        "ocr_handler": "demo_ocr",
                    },
                    "warnings": [],
                    "metadata_completeness": "full",
                },
            }
        }
    )


class OCRErrorResponse(BaseModel):
    """OCR 錯誤回應模型"""

    error: str = Field(..., description="錯誤訊息")
    error_code: str = Field(..., description="錯誤代碼")
    processing_time: float = Field(..., ge=0.0, description="處理時間（秒）")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "無法讀取圖片檔案",
                "error_code": "INVALID_IMAGE",
                "processing_time": 0.001,
            }
        }
    )


# 圖片生成相關模型
class GenerateRequest(BaseModel):
    """圖片生成請求模型"""

    text: str = Field(..., min_length=1, max_length=10, description="要生成的文字")
    style: CaptchaStyle = Field(default=CaptchaStyle.standard, description="生成樣式")
    width: Optional[int] = Field(default=128, ge=64, le=512, description="圖片寬度")
    height: Optional[int] = Field(default=64, ge=32, le=256, description="圖片高度")
    font_size: Optional[int] = Field(default=24, ge=12, le=72, description="字體大小")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("文字不能為空")
        if not value.isalnum():
            raise ValueError("文字只能包含字母和數字")
        return value.strip()

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "abcd",
                "style": "standard",
                "width": 128,
                "height": 64,
                "font_size": 24,
            }
        }
    )


class GenerateResponse(BaseModel):
    """圖片生成回應模型"""

    image_base64: str = Field(..., description="Base64 編碼的圖片")
    text: str = Field(..., description="生成的文字")
    style: str = Field(..., description="使用的樣式")
    generation_time: float = Field(..., ge=0.0, description="生成時間（秒）")
    image_size: str = Field(..., description="圖片尺寸")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "text": "abcd",
                "style": "standard",
                "generation_time": 0.056,
                "image_size": "128x64",
            }
        }
    )


class GenerateErrorResponse(BaseModel):
    """圖片生成錯誤回應模型"""

    error: str = Field(..., description="錯誤訊息")
    error_code: str = Field(..., description="錯誤代碼")
    generation_time: float = Field(..., ge=0.0, description="處理時間（秒）")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "文字長度超出限制",
                "error_code": "INVALID_TEXT_LENGTH",
                "generation_time": 0.001,
            }
        }
    )


# 批次生成相關模型
class BatchGenerateRequest(BaseModel):
    """批次圖片生成請求模型"""

    count: int = Field(..., ge=1, le=100, description="生成數量")
    style: CaptchaStyle = Field(default=CaptchaStyle.standard, description="生成樣式")
    text_list: Optional[List[str]] = Field(default=None, description="指定文字列表（可選）")
    width: Optional[int] = Field(default=128, ge=64, le=512, description="圖片寬度")
    height: Optional[int] = Field(default=64, ge=32, le=256, description="圖片高度")
    font_size: Optional[int] = Field(default=24, ge=12, le=72, description="字體大小")

    @model_validator(mode="after")
    def validate_text_list(self) -> "BatchGenerateRequest":
        if self.text_list is not None:
            if len(self.text_list) != self.count:
                raise ValueError("文字列表長度必須等於生成數量")
            for text in self.text_list:
                if not text or not text.strip() or not text.isalnum():
                    raise ValueError("文字只能包含字母和數字且不能為空")
        return self

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "count": 5,
                "style": "standard",
                "text_list": ["abcd", "efgh", "ijkl", "mnop", "qrst"],
                "width": 128,
                "height": 64,
                "font_size": 24,
            }
        }
    )


class BatchGenerateResponse(BaseModel):
    """批次圖片生成回應模型"""

    images: List[Dict[str, Union[str, float]]] = Field(..., description="生成的圖片列表")
    total_count: int = Field(..., description="總生成數量")
    total_generation_time: float = Field(..., ge=0.0, description="總生成時間（秒）")
    average_generation_time: float = Field(..., ge=0.0, description="平均生成時間（秒）")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "images": [
                    {
                        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                        "text": "abcd",
                        "generation_time": 0.056,
                    },
                    {
                        "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                        "text": "efgh",
                        "generation_time": 0.062,
                    },
                ],
                "total_count": 2,
                "total_generation_time": 0.118,
                "average_generation_time": 0.059,
            }
        }
    )


# 健康檢查相關模型
class HealthResponse(BaseModel):
    """健康檢查回應模型"""

    status: HealthStatus = Field(..., description="服務狀態")
    model_loaded: bool = Field(..., description="模型是否已載入")
    version: str = Field(..., description="核心版本號")
    handler_versions: Dict[str, str] = Field(default_factory=dict, description="各 handler 版本資訊")
    uptime: float = Field(..., ge=0.0, description="運行時間（秒）")
    model_info: Optional[Dict[str, Any]] = Field(default=None, description="模型資訊")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "version": "1.0.0",
                "handler_versions": {
                    "ocr": "1.20250919.1640",
                    "preprocess": "1.20250919.1640"
                },
                "uptime": 3600.0,
                "model_info": {
                    "vocab_size": 62,
                    "max_length": 4,
                    "total_params": 1234567,
                },
            }
        }
    )


# 模型資訊相關模型
class ModelInfoResponse(BaseModel):
    """模型資訊回應模型"""

    model_name: str = Field(..., description="模型名稱")
    model_version: str = Field(..., description="模型版本")
    model_size: str = Field(..., description="模型大小")
    vocab_size: int = Field(..., description="詞彙表大小")
    max_length: int = Field(..., description="最大序列長度")
    total_params: int = Field(..., description="總參數數量")
    trainable_params: int = Field(..., description="可訓練參數數量")
    alphabet: str = Field(..., description="支援的字元集")
    input_size: str = Field(..., description="輸入圖片尺寸")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "model_name": "CaptchaOCRModel",
                "model_version": "1.0.0",
                "model_size": "4.7 MB",
                "vocab_size": 62,
                "max_length": 4,
                "total_params": 1234567,
                "trainable_params": 1234567,
                "alphabet": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                "input_size": "128x64",
            }
        }
    )


# 統計資訊相關模型
class APIStatsResponse(BaseModel):
    """API 統計資訊回應模型"""

    total_requests: int = Field(..., description="總請求數")
    ocr_requests: int = Field(..., description="OCR 請求數")
    generate_requests: int = Field(..., description="生成請求數")
    success_rate: float = Field(..., ge=0.0, le=1.0, description="成功率")
    average_processing_time: float = Field(..., ge=0.0, description="平均處理時間（秒）")
    uptime: float = Field(..., ge=0.0, description="運行時間（秒）")
    requests_per_minute: float = Field(..., ge=0.0, description="每分鐘請求數")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "total_requests": 1000,
                "ocr_requests": 600,
                "generate_requests": 400,
                "success_rate": 0.98,
                "average_processing_time": 0.15,
                "uptime": 7200.0,
                "requests_per_minute": 8.33,
            }
        }
    )


# 錯誤回應基礎模型
class BaseErrorResponse(BaseModel):
    """基礎錯誤回應模型"""

    detail: str = Field(..., description="錯誤詳細資訊")
    error_code: Optional[str] = Field(default=None, description="錯誤代碼")
    timestamp: str = Field(..., description="錯誤時間戳")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "內部服務器錯誤",
                "error_code": "INTERNAL_ERROR",
                "timestamp": "2024-01-01T12:00:00Z",
            }
        }
    )


# 驗證錯誤回應
class ValidationErrorResponse(BaseModel):
    """驗證錯誤回應模型"""

    detail: List[Dict[str, Any]] = Field(..., description="驗證錯誤詳細資訊")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": [
                    {
                        "loc": ["body", "text"],
                        "msg": "文字長度必須在 1-10 之間",
                        "type": "value_error",
                    }
                ]
            }
        }
    )
