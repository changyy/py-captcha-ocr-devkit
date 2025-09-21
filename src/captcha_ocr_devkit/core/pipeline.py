"""
Handler Pipeline 系統

管理不同 handler 之間的組合和資料流
"""

from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import time
import logging
from dataclasses import dataclass

from captcha_ocr_devkit import __version__ as CORE_VERSION

from .handlers.base import (
    BaseHandler,
    BasePreprocessHandler,
    BaseTrainHandler,
    BaseEvaluateHandler,
    BaseOCRHandler,
    HandlerResult,
    TrainingConfig,
    EvaluationResult
)
from .handlers.registry import registry


@dataclass
class PipelineConfig:
    """流水線配置"""
    preprocess_handler: Optional[str] = None
    train_handler: Optional[str] = None
    evaluate_handler: Optional[str] = None
    ocr_handler: Optional[str] = None
    handler_configs: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.handler_configs is None:
            self.handler_configs = {}


class HandlerPipeline:
    """
    Handler 流水線系統
    
    組合不同類型的 handler，實現完整的處理流程
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.handlers: Dict[str, BaseHandler] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialize_handlers()
    
    def _initialize_handlers(self):
        """初始化 handlers"""
        handler_types = ['preprocess', 'train', 'evaluate', 'ocr']
        
        for handler_type in handler_types:
            handler_name = getattr(self.config, f"{handler_type}_handler")
            if handler_name:
                try:
                    handler_config = self.config.handler_configs.get(handler_name, {})
                    handler = registry.create_handler(handler_type, handler_name, handler_config)
                    self.handlers[handler_type] = handler
                    self.logger.info(f"Initialized {handler_type} handler: {handler_name}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize {handler_type} handler '{handler_name}': {e}")
                    raise
    
    def get_handler(self, handler_type: str) -> Optional[BaseHandler]:
        """取得指定類型的 handler"""
        return self.handlers.get(handler_type)
    
    def has_handler(self, handler_type: str) -> bool:
        """檢查是否有指定類型的 handler"""
        return handler_type in self.handlers
    
    def process_image(self, image_data: Union[bytes, str, Path]) -> HandlerResult:
        """
        圖片處理流程: 預處理 -> OCR
        
        用於 API 服務中的圖片識別
        """
        start_time = time.time()
        
        try:
            processed_data = image_data
            metadata = {'pipeline_start': start_time}
            
            # 預處理步驟
            if self.has_handler('preprocess'):
                preprocess_handler = self.get_handler('preprocess')
                preprocess_result = preprocess_handler.process(image_data)
                
                if not preprocess_result.success:
                    return HandlerResult(
                        success=False,
                        error=f"Preprocessing failed: {preprocess_result.error}",
                        metadata=metadata
                    )
                
                processed_data = preprocess_result.data
                metadata.update(preprocess_result.metadata or {})
                self.logger.info("Image preprocessing completed")
            
            # OCR 識別步驟
            if not self.has_handler('ocr'):
                return HandlerResult(
                    success=False,
                    error="No OCR handler configured",
                    metadata=metadata
                )
            
            ocr_handler = self.get_handler('ocr')
            ocr_result = ocr_handler.predict(processed_data)
            
            if not ocr_result.success:
                return HandlerResult(
                    success=False,
                    error=f"OCR failed: {ocr_result.error}",
                    metadata=metadata
                )
            
            # 結合結果
            processing_time = time.time() - start_time
            metadata.update({
                'processing_time': processing_time,
                'pipeline_end': time.time()
            })
            metadata.update(ocr_result.metadata or {})
            
            self.logger.info(f"Image processing completed in {processing_time:.3f}s")
            
            return HandlerResult(
                success=True,
                data=ocr_result.data,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}")
            return HandlerResult(
                success=False,
                error=str(e),
                metadata={'processing_time': time.time() - start_time}
            )
    
    def train_model(self, training_config: TrainingConfig) -> HandlerResult:
        """
        模型訓練流程
        """
        start_time = time.time()
        
        if not self.has_handler('train'):
            return HandlerResult(
                success=False,
                error="No training handler configured"
            )
        
        try:
            train_handler = self.get_handler('train')

            self.logger.info(f"Starting model training with {train_handler.name}")
            self.logger.info(f"Input: {training_config.input_dir}")
            self.logger.info(f"Output: {training_config.output_path}")

            handler_id = getattr(train_handler.__class__, 'HANDLER_ID', train_handler.__class__.__name__)
            handler_version = None
            try:
                handler_info = train_handler.get_info()
                handler_version = handler_info.get('version') or handler_info.get('handler_version')
            except Exception as exc:  # pragma: no cover
                self.logger.warning(f"Unable to fetch handler info for {handler_id}: {exc}")

            version_msg = f"core_version={CORE_VERSION}"
            if handler_version:
                version_msg += f", handler_version={handler_version}"
            version_msg += f", handler_id={handler_id}"
            self.logger.info(f"Training metadata: {version_msg}")

            result = train_handler.train(training_config)
            
            training_time = time.time() - start_time
            if result.metadata is None:
                result.metadata = {}
            result.metadata['training_time'] = training_time
            
            if result.success:
                self.logger.info(f"Training completed in {training_time:.2f}s")
            else:
                self.logger.error(f"Training failed: {result.error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Training pipeline error: {e}")
            return HandlerResult(
                success=False,
                error=str(e),
                metadata={'training_time': time.time() - start_time}
            )
    
    def evaluate_model(self, model_path: Path, test_data_path: Path) -> HandlerResult:
        """
        模型評估流程
        """
        start_time = time.time()
        
        if not self.has_handler('evaluate'):
            return HandlerResult(
                success=False,
                error="No evaluation handler configured"
            )
        
        try:
            evaluate_handler = self.get_handler('evaluate')
            
            self.logger.info(f"Starting model evaluation with {evaluate_handler.name}")
            self.logger.info(f"Model: {model_path}")
            self.logger.info(f"Test data: {test_data_path}")
            
            result = evaluate_handler.evaluate(model_path, test_data_path)
            
            evaluation_time = time.time() - start_time
            if result.metadata is None:
                result.metadata = {}
            result.metadata['evaluation_time'] = evaluation_time
            
            if result.success:
                self.logger.info(f"Evaluation completed in {evaluation_time:.2f}s")
                if isinstance(result.data, EvaluationResult):
                    eval_result = result.data
                    self.logger.info(f"Accuracy: {eval_result.accuracy:.4f}")
                    self.logger.info(f"Character accuracy: {eval_result.character_accuracy:.4f}")
            else:
                self.logger.error(f"Evaluation failed: {result.error}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Evaluation pipeline error: {e}")
            return HandlerResult(
                success=False,
                error=str(e),
                metadata={'evaluation_time': time.time() - start_time}
            )
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """取得流水線資訊"""
        info = {
            'config': {
                'preprocess_handler': self.config.preprocess_handler,
                'train_handler': self.config.train_handler,
                'evaluate_handler': self.config.evaluate_handler,
                'ocr_handler': self.config.ocr_handler
            },
            'handlers': {}
        }
        
        for handler_type, handler in self.handlers.items():
            info['handlers'][handler_type] = {
                'name': handler.name,
                'class': handler.__class__.__name__,
                'info': handler.get_info()
            }
        
        return info


def create_pipeline_from_handlers(preprocess_handler: Optional[str] = None,
                                train_handler: Optional[str] = None,
                                evaluate_handler: Optional[str] = None,
                                ocr_handler: Optional[str] = None,
                                handler_configs: Optional[Dict[str, Dict[str, Any]]] = None) -> HandlerPipeline:
    """
    從 handler 名稱創建 pipeline
    
    這是 CLI 使用的主要函數
    """
    config = PipelineConfig(
        preprocess_handler=preprocess_handler,
        train_handler=train_handler,
        evaluate_handler=evaluate_handler,
        ocr_handler=ocr_handler,
        handler_configs=handler_configs or {}
    )
    
    return HandlerPipeline(config)
