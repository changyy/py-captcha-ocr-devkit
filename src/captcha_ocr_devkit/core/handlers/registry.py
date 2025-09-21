"""
Handler 動態註冊與發現機制

參考 yt-dlp 的插件架構，實作可擴展的 handler 系統
"""

import os
import sys
import importlib
import importlib.util
from pathlib import Path
from typing import Dict, List, Type, Optional, Any, Union
import inspect
import json
from .base import (
    BaseHandler,
    BasePreprocessHandler,
    BaseTrainHandler,
    BaseEvaluateHandler,
    BaseOCRHandler
)


def _is_relative_to(path: Path, other: Path) -> bool:
    try:
        path.relative_to(other)
        return True
    except ValueError:
        return False


class HandlerRegistry:
    """
    Handler 註冊中心
    
    支援動態發現、註冊和管理各種 handler
    """
    
    def __init__(self):
        self._supported_types = {'preprocess', 'train', 'evaluate', 'ocr'}
        self._handlers: Dict[str, Dict[str, Type[BaseHandler]]] = {
            handler_type: {} for handler_type in self._supported_types
        }
        self._handler_sources: Dict[str, Dict[str, Path]] = {
            handler_type: {} for handler_type in self._supported_types
        }
        self.handler_instances: Dict[str, BaseHandler] = {}
        self.search_paths: List[Path] = []

        # 內建搜尋路徑：套件範例 handlers 以及專案根目錄 handlers
        module_root = Path(__file__).resolve().parents[2]
        default_paths = [
            module_root / 'examples' / 'handlers',
            Path(__file__).resolve().parents[4] / 'handlers'
        ]

        # 支援環境變數覆寫，多個路徑以 os.pathsep 分隔
        env_paths = os.getenv('CAPTCHA_OCR_HANDLER_PATHS')
        if env_paths:
            default_paths.extend(Path(p).expanduser() for p in env_paths.split(os.pathsep) if p)

        for candidate in default_paths:
            self.add_search_path(candidate)

    @property
    def handlers(self) -> Dict[str, Dict[str, Type[BaseHandler]]]:
        """向後相容的 handlers 訪問介面"""
        for handler_type in self._supported_types:
            self._handlers.setdefault(handler_type, {})
            self._handler_sources.setdefault(handler_type, {})
        return self._handlers
        
    def add_search_path(self, path: Union[str, Path]) -> None:
        """新增 handler 搜尋路徑"""
        path = Path(path)
        if path.exists() and path not in self.search_paths:
            self.search_paths.append(path)
            
    def discover_handlers(self, search_dir: Optional[Path] = None) -> Dict[str, List[str]]:
        """
        自動發現 handlers
        
        Args:
            search_dir: 搜尋目錄，預設為當前目錄的 'handlers' 子目錄
            
        Returns:
            發現的 handlers 字典，以類型分類
        """
        discovered = {handler_type: [] for handler_type in self._supported_types}

        candidate_dirs: List[Path] = []
        if search_dir is not None:
            candidate_dirs.append(Path(search_dir))
        else:
            candidate_dirs.append(Path.cwd() / 'handlers')

        candidate_dirs.extend(self.search_paths)

        seen: set[Path] = set()
        for directory in candidate_dirs:
            directory = directory.resolve()
            if directory in seen or not directory.exists() or not directory.is_dir():
                continue
            seen.add(directory)

            for py_file in directory.glob('*.py'):
                if py_file.name.startswith('_'):
                    continue

                try:
                    handlers = self._load_handlers_from_file(py_file)
                    for handler_type, handler_classes in handlers.items():
                        discovered[handler_type].extend(handler_classes)
                except Exception as e:
                    print(f"Warning: Failed to load handlers from {py_file}: {e}")

        for handler_type, handler_list in discovered.items():
            if handler_list:
                discovered[handler_type] = list(dict.fromkeys(handler_list))

        return discovered
    
    def _load_handlers_from_file(self, py_file: Path) -> Dict[str, List[str]]:
        """從單一 Python 檔案載入 handlers"""
        spec = importlib.util.spec_from_file_location(py_file.stem, py_file)
        if spec is None or spec.loader is None:
            return {'preprocess': [], 'train': [], 'evaluate': [], 'ocr': []}
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        handlers = {
            'preprocess': [],
            'train': [],
            'evaluate': [],
            'ocr': []
        }
        
        # 檢查模組中的所有類別
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if self._is_handler_class(obj):
                handler_type = self._get_handler_type(obj)
                if handler_type:
                    handler_id = obj.get_handler_id() if hasattr(obj, 'get_handler_id') else name
                    if self.register_handler(handler_type, handler_id, obj, source_path=py_file):
                        handlers[handler_type].append(handler_id)
                    
        return handlers
    
    def _is_handler_class(self, cls: Type) -> bool:
        """檢查是否為有效的 handler 類別"""
        if not inspect.isclass(cls):
            return False
            
        # 檢查是否繼承自 BaseHandler 子類
        base_classes = [
            BasePreprocessHandler,
            BaseTrainHandler, 
            BaseEvaluateHandler,
            BaseOCRHandler
        ]
        
        for base_class in base_classes:
            if issubclass(cls, base_class) and cls != base_class:
                return True
                
        return False
    
    def _get_handler_type(self, cls: Type) -> Optional[str]:
        """取得 handler 類型"""
        if issubclass(cls, BasePreprocessHandler):
            return 'preprocess'
        elif issubclass(cls, BaseTrainHandler):
            return 'train'
        elif issubclass(cls, BaseEvaluateHandler):
            return 'evaluate'
        elif issubclass(cls, BaseOCRHandler):
            return 'ocr'
        return None
    
    def register_handler(
        self,
        handler_type: str,
        handler_id: str,
        handler_class: Type[BaseHandler],
        source_path: Optional[Path] = None,
    ) -> bool:
        """註冊 handler"""
        if handler_type not in self._supported_types:
            raise ValueError(f"Unknown handler type: {handler_type}")

        if handler_type not in self._handlers:
            self._handlers[handler_type] = {}
            self._handler_sources[handler_type] = {}

        existing = self._handlers[handler_type].get(handler_id)
        if existing:
            existing_source = self._handler_sources[handler_type].get(handler_id)
            if self._should_override(existing_source, source_path):
                self._handlers[handler_type][handler_id] = handler_class
                if source_path is not None:
                    self._handler_sources[handler_type][handler_id] = source_path
                return True
            return False

        self._handlers[handler_type][handler_id] = handler_class
        if source_path is not None:
            self._handler_sources[handler_type][handler_id] = source_path
        return True
        
    def get_available_handlers(self, handler_type: Optional[str] = None) -> Dict[str, List[str]]:
        """取得可用的 handlers"""
        if handler_type:
            if handler_type not in self._handlers:
                return {}
            return {handler_type: list(self._handlers[handler_type].keys())}

        return {k: list(v.keys()) for k, v in self._handlers.items()}
    
    def create_handler(self, handler_type: str, handler_name: str, 
                      config: Optional[Dict[str, Any]] = None) -> BaseHandler:
        """創建 handler 實例"""
        if handler_type not in self._handlers:
            raise ValueError(f"Unknown handler type: {handler_type}")
            
        if handler_name not in self._handlers[handler_type]:
            available = list(self._handlers[handler_type].keys())
            raise ValueError(f"Handler '{handler_name}' not found in {handler_type}. Available: {available}")
        
        handler_class = self._handlers[handler_type][handler_name]
        instance = handler_class(name=handler_name, config=config)
        
        # 儲存實例供後續使用
        instance_key = f"{handler_type}.{handler_name}"
        self.handler_instances[instance_key] = instance
        
        return instance

    def _should_override(self, existing: Optional[Path], incoming: Optional[Path]) -> bool:
        """判斷是否以新的 handler 覆蓋既有註冊。"""
        if incoming is None:
            return False
        if existing is None:
            return True

        incoming = incoming.resolve()
        existing = existing.resolve()

        cwd_handlers = Path.cwd() / 'handlers'
        if _is_relative_to(incoming, cwd_handlers) and not _is_relative_to(existing, cwd_handlers):
            return True
        if _is_relative_to(incoming, cwd_handlers) and _is_relative_to(existing, cwd_handlers):
            try:
                return incoming.stat().st_mtime >= existing.stat().st_mtime
            except OSError:
                return False

        return False
    
    def get_handler_instance(self, handler_type: str, handler_name: str) -> Optional[BaseHandler]:
        """取得已創建的 handler 實例"""
        instance_key = f"{handler_type}.{handler_name}"
        return self.handler_instances.get(instance_key)
    
    def list_handlers_interactive(self, handler_type: str) -> Optional[str]:
        """
        交互式選擇 handler
        
        當有多個 handler 時，讓使用者選擇
        """
        available = self._handlers.get(handler_type, {})
        
        if not available:
            print(f"No {handler_type} handlers found.")
            return None
            
        if len(available) == 1:
            handler_name = list(available.keys())[0]
            print(f"Using {handler_type} handler: {handler_name}")
            return handler_name
            
        # 多個選擇時顯示列表
        print(f"\nAvailable {handler_type} handlers:")
        handler_list = list(available.keys())
        for i, name in enumerate(handler_list, 1):
            handler_class = available[name]
            print(f"  {i}. {name} ({handler_class.__name__})")
        
        while True:
            try:
                choice = input(f"\nSelect {handler_type} handler (1-{len(handler_list)}) or 'q' to quit: ").strip()
                
                if choice.lower() == 'q':
                    return None
                    
                idx = int(choice) - 1
                if 0 <= idx < len(handler_list):
                    selected = handler_list[idx]
                    print(f"Selected: {selected}")
                    return selected
                else:
                    print(f"Please enter a number between 1 and {len(handler_list)}")
                    
            except (ValueError, KeyboardInterrupt):
                print("\nCancelled.")
                return None
    
    def save_registry_info(self, output_path: Path) -> None:
        """儲存 registry 狀態到檔案"""
        registry_info = {
            'handlers': {},
            'search_paths': [str(p) for p in self.search_paths],
            'instances': list(self.handler_instances.keys())
        }
        
        for handler_type, handlers in self.handlers.items():
            registry_info['handlers'][handler_type] = {
                name: {
                    'class_name': cls.__name__,
                    'module': cls.__module__,
                    'file': inspect.getfile(cls) if inspect.getfile(cls) else 'unknown'
                }
                for name, cls in handlers.items()
            }
        
        with open(output_path, 'w') as f:
            json.dump(registry_info, f, indent=2)
    
    def load_handlers_from_config(self, config_path: Path) -> None:
        """從配置檔案載入 handlers"""
        if not config_path.exists():
            return
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 新增搜尋路徑
        for path_str in config.get('search_paths', []):
            self.add_search_path(Path(path_str))
        
        # 重新發現 handlers
        for search_path in self.search_paths:
            self.discover_handlers(search_path)


# 全域 registry 實例
registry = HandlerRegistry()


def auto_discover_and_select(handler_type: str,
                            handler_name: Optional[str] = None,
                            search_dir: Optional[Path] = None,
                            interactive: bool = True) -> Optional[str]:
    """
    自動發現並選擇 handler
    
    這是 CLI 使用的主要函數
    """
    # 發現 handlers
    discovered = registry.discover_handlers(search_dir)
    
    if handler_name:
        # 指定了 handler 名稱
        if handler_name in registry.handlers.get(handler_type, {}):
            return handler_name
        else:
            available = list(registry.handlers.get(handler_type, {}).keys())
            print(f"Handler '{handler_name}' not found. Available {handler_type} handlers: {available}")
            return None

    # 沒有指定 handler，進行選擇
    available = registry.handlers.get(handler_type, {})

    if not interactive:
        if not available:
            print(f"No {handler_type} handlers found.")
            return None

        preferred_defaults = {
            'preprocess': 'demo_preprocess',
            'train': 'demo_train',
            'evaluate': 'demo_evaluate',
            'ocr': 'demo_ocr'
        }

        preferred = preferred_defaults.get(handler_type)
        if preferred and preferred in available:
            return preferred

        # 回退到第一個可用 handler（依名稱排序保持穩定）
        return sorted(available.keys())[0]

    if interactive:
        return registry.list_handlers_interactive(handler_type)
    else:
        if len(available) == 1:
            return list(available.keys())[0]
        elif len(available) > 1:
            print(f"Multiple {handler_type} handlers found. Please specify with --handler option:")
            for name in available.keys():
                print(f"  --handler {name}")
            return None
        else:
            print(f"No {handler_type} handlers found.")
            return None
