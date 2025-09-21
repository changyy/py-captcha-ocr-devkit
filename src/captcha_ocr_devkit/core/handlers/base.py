"""
CAPTCHA OCR Handler 抽象基類

提供插件化架構，支援使用者自訂處理邏輯
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import importlib
import importlib.util
import json
from datetime import datetime


@dataclass
class HandlerResult:
    """Handler 處理結果"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TrainingConfig:
    """訓練配置"""
    input_dir: Path
    output_path: Path
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    validation_split: float = 0.2
    seed: Optional[int] = None
    device: str = 'auto'
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        return cls(
            input_dir=Path(config_dict['input_dir']),
            output_path=Path(config_dict['output_path']),
            epochs=config_dict.get('epochs', 100),
            batch_size=config_dict.get('batch_size', 32),
            learning_rate=config_dict.get('learning_rate', 0.001),
            validation_split=config_dict.get('validation_split', 0.2),
            seed=config_dict.get('seed'),
            device=config_dict.get('device', 'auto')
        )


@dataclass
class EvaluationResult:
    """評估結果"""
    accuracy: float
    total_samples: int
    correct_predictions: int
    character_accuracy: float
    confusion_matrix: Optional[Dict] = None
    per_class_metrics: Optional[Dict] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class BaseHandler(ABC):
    """所有 Handler 的基類"""

    HANDLER_ID: Optional[str] = None

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        self.name = name
        self.config = config or {}
        self.metadata = {
            'handler_type': self.__class__.__name__,
            'created_at': datetime.now().isoformat()
        }

    @classmethod
    def get_handler_id(cls) -> str:
        handler_id = getattr(cls, 'HANDLER_ID', None)
        if handler_id:
            return str(handler_id)
        return cls.__name__

    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """回傳 Handler 資訊"""
        pass

    def validate_config(self) -> bool:
        """驗證配置是否正確"""
        return True

    # ------------------------------------------------------------------
    # 描述與相依性
    # ------------------------------------------------------------------

    def get_description(self) -> str:
        """取得 handler 詳細描述，可用於健康檢查或文件輸出。"""
        if 'description' in self.config:
            return str(self.config['description'])
        description = getattr(self, 'DESCRIPTION', None)
        if description:
            return str(description)
        doc = self.__doc__ or ''
        return ' '.join(doc.strip().split())

    def get_short_description(self) -> str:
        """取得 handler 簡短描述，預設採用完整描述的第一句。"""
        if 'short_description' in self.config:
            return str(self.config['short_description'])
        short_desc = getattr(self, 'SHORT_DESCRIPTION', None)
        if short_desc:
            return str(short_desc)
        description = self.get_description()
        if not description:
            return ''
        first_sentence = description.split('.')
        return first_sentence[0].strip()

    def get_dependencies(self) -> List[str]:
        """回傳 handler 執行所需套件列表。"""
        deps = self.config.get('dependencies')
        if deps:
            return list(deps)
        required = getattr(self, 'REQUIRED_DEPENDENCIES', None)
        if required:
            return list(required)
        return []

    def _normalize_dependency_name(self, dependency: str) -> str:
        token = dependency.strip()
        for delimiter in ['>=', '<=', '==', '!=', '~=', '>','<']:
            if delimiter in token:
                token = token.split(delimiter)[0]
                break
        token = token.split()[0].split('[')[0].replace('-', '_')
        dependency_aliases = {
            'pillow': 'PIL',
            'opencv_python': 'cv2',
            'opencv_python_headless': 'cv2',
        }
        return dependency_aliases.get(token.lower(), token)

    def get_dependency_status(self) -> Dict[str, bool]:
        """檢查依賴是否可被載入，回傳 {dependency: 是否存在}。"""
        status: Dict[str, bool] = {}
        for dep in self.get_dependencies():
            module_name = self._normalize_dependency_name(dep)
            try:
                status[dep] = importlib.util.find_spec(module_name) is not None
            except Exception:  # pragma: no cover - defensive safety
                status[dep] = False
        return status

    def get_missing_dependencies(self) -> List[str]:
        """取得缺少的依賴列表。"""
        return [dep for dep, ok in self.get_dependency_status().items() if not ok]

    def save_config(self, path: Path) -> None:
        """儲存配置到檔案"""
        config_data = {
            'handler_name': self.name,
            'handler_type': self.__class__.__name__,
            'config': self.config,
            'metadata': self.metadata
        }
        with open(path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
    
    @classmethod
    def from_config_file(cls, path: Path) -> 'BaseHandler':
        """從配置檔案載入 Handler"""
        with open(path, 'r') as f:
            config_data = json.load(f)
        
        return cls(
            name=config_data['handler_name'],
            config=config_data['config']
        )


class BasePreprocessHandler(BaseHandler):
    """圖片預處理 Handler 基類"""
    
    @abstractmethod
    def process(self, image_data: Union[bytes, str, Path]) -> HandlerResult:
        """
        處理圖片資料
        
        Args:
            image_data: 圖片資料 (bytes/路徑)
            
        Returns:
            HandlerResult: 處理結果，data 包含處理後的圖片資料
        """
        pass
    
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """回傳支援的圖片格式"""
        pass


class BaseTrainHandler(BaseHandler):
    """模型訓練 Handler 基類"""
    
    @abstractmethod
    def train(self, config: TrainingConfig) -> HandlerResult:
        """
        訓練模型
        
        Args:
            config: 訓練配置
            
        Returns:
            HandlerResult: 訓練結果，包含模型路徑和訓練統計
        """
        pass
    
    @abstractmethod
    def save_model(self, model_data: Any, output_path: Path) -> bool:
        """儲存模型"""
        pass
    
    @abstractmethod
    def load_model(self, model_path: Path) -> Any:
        """載入模型"""
        pass
    
    def parse_labels_from_filenames(self, image_paths: List[Path]) -> List[str]:
        """
        從檔名解析標籤 (預設實作)
        例如: abcd_001.png -> "abcd"
        """
        labels = []
        for path in image_paths:
            filename = path.stem  # 移除副檔名
            label = filename.split('_')[0]  # 以底線分割，取第一部分
            labels.append(label)
        return labels


class BaseEvaluateHandler(BaseHandler):
    """模型評估 Handler 基類"""
    
    @abstractmethod
    def evaluate(self, model_path: Path, test_data_path: Path) -> HandlerResult:
        """
        評估模型
        
        Args:
            model_path: 模型檔案路徑
            test_data_path: 測試資料路徑
            
        Returns:
            HandlerResult: 評估結果
        """
        pass
    
    @abstractmethod
    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> EvaluationResult:
        """計算評估指標"""
        pass


class BaseOCRHandler(BaseHandler):
    """OCR 預測 Handler 基類"""
    
    @abstractmethod
    def predict(self, processed_image: Any) -> HandlerResult:
        """
        對處理後的圖片進行 OCR 預測
        
        Args:
            processed_image: 預處理後的圖片資料
            
        Returns:
            HandlerResult: 預測結果，data 包含識別的文字和信心度
        """
        pass
    
    @abstractmethod
    def load_model(self, model_path: Path) -> bool:
        """載入 OCR 模型"""
        pass
    
    def get_prediction_confidence(self, result: HandlerResult) -> float:
        """
        從預測結果中提取信心度
        """
        if result.success and result.metadata:
            return result.metadata.get('confidence', 0.0)
        return 0.0
