"""Shared utilities for captcha OCR handlers."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    np = None  # type: ignore
    NUMPY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    Image = None  # type: ignore
    PIL_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset, random_split

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    TORCH_AVAILABLE = False
    torch = None  # type: ignore
    nn = None  # type: ignore
    optim = None  # type: ignore
    DataLoader = None  # type: ignore
    Dataset = object  # type: ignore
    random_split = None  # type: ignore

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DEFAULT_INSTALL_FALLBACK = "pip install torch torchvision pillow numpy"


def format_dependency_error(missing: Sequence[str], install_hint: str = DEFAULT_INSTALL_FALLBACK) -> str:
    missing_str = ", ".join(missing)
    return f"缺少必要套件: {missing_str}. 請先執行 {install_hint}。"

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False
if LOGGER.getEffectiveLevel() > logging.INFO:
    LOGGER.setLevel(logging.INFO)


class TorchHandlerDependencyMixin:
    """Mixin providing dependency helpers for torch-based handlers."""

    REQUIREMENTS_FILE: Optional[Union[str, Path]] = None
    INSTALL_FALLBACK: str = DEFAULT_INSTALL_FALLBACK
    config: Dict[str, Any]

    def _requirements_override(self) -> Optional[Union[str, Path]]:
        return self.config.get("requirements_file") if isinstance(self.config, dict) else None

    def _module_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _requirements_file_path(self) -> Path:
        override = self._requirements_override()
        module_dir = self._module_dir()
        if override:
            path = Path(override)
            if not path.is_absolute():
                path = module_dir / path
            return path
        if self.REQUIREMENTS_FILE:
            return module_dir / Path(self.REQUIREMENTS_FILE)
        return module_dir / Path("torch_handler-requirements.txt")

    def _install_hint(self) -> str:
        req_path = self._requirements_file_path()
        if req_path.exists():
            try:
                display_path = req_path.relative_to(Path.cwd())
            except ValueError:  # pragma: no cover - path outside cwd
                display_path = req_path
            return f"pip install -r {display_path}"
        return self.INSTALL_FALLBACK

    def _dependency_error_message(self, missing: Sequence[str]) -> str:
        return format_dependency_error(missing, self._install_hint())


def _missing_dependencies(require_torch: bool = True) -> List[str]:
    missing: List[str] = []
    if require_torch and not TORCH_AVAILABLE:
        missing.extend(["torch", "torchvision"])
    if not NUMPY_AVAILABLE:
        missing.append("numpy")
    if not PIL_AVAILABLE:
        missing.append("Pillow")
    return missing


def set_seed(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    if NUMPY_AVAILABLE:
        np.random.seed(seed)  # type: ignore[attr-defined]
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)  # type: ignore[union-attr]


def parse_label_from_filename(path: Path) -> str:
    return path.stem.split("_")[0]


class OCRDataset(Dataset):
    """Generic captcha OCR dataset that resizes and normalizes images."""

    def __init__(
        self,
        root: Union[str, Path],
        img_h: int,
        img_w: int,
        requirements_override: Optional[Union[str, Path]] = None,
        extensions: Optional[Iterable[str]] = None,
    ):
        missing = _missing_dependencies()
        if missing:
            raise RuntimeError(format_dependency_error(missing))
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        self.img_h = img_h
        self.img_w = img_w
        exts = {ext.lower() for ext in (extensions or SUPPORTED_EXTENSIONS)}
        self.samples: List[Tuple[Path, str]] = []
        for path in sorted(self.root.iterdir()):
            if path.suffix.lower() in exts:
                label = parse_label_from_filename(path)
                if label:
                    self.samples.append((path, label))
        if not self.samples:
            raise RuntimeError(f"No supported images found in {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> Image.Image:
        if not PIL_AVAILABLE:
            raise RuntimeError("Pillow is required to load images. 請先執行 pip install pillow")
        return Image.open(path).convert("L")  # type: ignore[union-attr]

    def _resize_pad(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        scale = self.img_h / float(h)
        new_w = max(1, int(w * scale))
        img = img.resize((new_w, self.img_h), Image.BILINEAR)
        if new_w > self.img_w:
            img = img.crop((0, 0, self.img_w, self.img_h))
            new_w = self.img_w
        canvas = Image.new("L", (self.img_w, self.img_h), color=255)
        canvas.paste(img, (0, 0))
        return canvas

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", str, Path]:  # type: ignore[override]
        if not NUMPY_AVAILABLE or not TORCH_AVAILABLE:
            raise RuntimeError("NumPy and PyTorch are required to use OCRDataset")
        path, label = self.samples[idx]
        img = self._load_image(path)
        img = self._resize_pad(img)
        tensor = torch.from_numpy(np.array(img)).float().unsqueeze(0) / 255.0  # type: ignore[arg-type]
        return tensor, label, path


TransformerOCRDataset = OCRDataset


def collate_batch(batch: List[Tuple["torch.Tensor", str, Path]]) -> Tuple["torch.Tensor", List[str], List[Path]]:
    images, labels, paths = zip(*batch)
    stacked = torch.stack(images, dim=0)
    return stacked, list(labels), list(paths)


class Charset:
    """Simple character set helper mirroring the reference script."""

    BLANK_SYMBOL = "<blank>"

    def __init__(self, itos: List[str]):
        if not itos:
            raise ValueError("Charset cannot be empty")
        if itos[0] != self.BLANK_SYMBOL:
            raise ValueError("First entry of charset must be '<blank>'")
        self.itos = itos
        self.stoi = {ch: idx for idx, ch in enumerate(itos)}
        self.blank_idx = 0

    @classmethod
    def from_characters(cls, chars: Sequence[str]) -> "Charset":
        unique = sorted(set(chars))
        return cls([cls.BLANK_SYMBOL] + unique)

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode_greedy(self, logits: "torch.Tensor") -> str:
        indices = logits.argmax(dim=-1).tolist()
        output: List[str] = []
        prev = None
        for idx in indices:
            if idx != self.blank_idx and idx != prev:
                output.append(self.itos[idx])
            prev = idx
        return "".join(output)


if TORCH_AVAILABLE:

    class ConvFeatureExtractor(nn.Module):
        def __init__(self, in_channels: int = 1, out_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1), (2, 1)),
            )
            self.proj = nn.Linear(256, out_dim)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            feat = self.net(x)
            feat = feat.mean(dim=2, keepdim=True)
            feat = feat.squeeze(2)
            feat = feat.permute(0, 2, 1)
            feat = self.proj(feat)
            return feat

else:  # pragma: no cover - fallback when torch missing

    class ConvFeatureExtractor:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for ConvFeatureExtractor. Please install torch and torchvision.")


def build_charset_from_dataset(dataset: OCRDataset) -> Charset:
    chars: List[str] = []
    for _, label in dataset.samples:
        chars.extend(label)
    if not chars:
        raise RuntimeError("Unable to build charset from dataset labels")
    return Charset.from_characters(chars)


def resolve_device(requested: Optional[str]) -> "torch.device":
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for OCR handlers. Please install torch and torchvision.")
    if requested and requested not in {"auto", ""}:
        return torch.device(requested)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():  # pragma: no cover - depends on environment
        return torch.device("cuda")
    return torch.device("cpu")


def labels_to_targets(labels: Sequence[str], charset: Charset) -> Tuple["torch.Tensor", "torch.Tensor"]:
    targets = [charset.encode(label) for label in labels]
    lengths = torch.tensor([len(seq) for seq in targets], dtype=torch.long)
    if lengths.sum().item() == 0:
        targets = [[1] for _ in targets]
        lengths = torch.ones(len(targets), dtype=torch.long)
    flat = torch.tensor([idx for seq in targets for idx in seq], dtype=torch.long)
    return flat, lengths


def greedy_decode_batch(logits: "torch.Tensor", charset: Charset) -> List[str]:
    return [charset.decode_greedy(sequence) for sequence in logits]


def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n < m:
        a, b = b, a
        n, m = m, n
    previous = list(range(m + 1))
    for i in range(1, n + 1):
        current = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )
        previous = current
    return previous[m]


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    charset: Charset,
    device: "torch.device",
) -> Tuple[float, float, List[Tuple[Path, str, str]]]:
    model.eval()
    total = 0
    correct = 0
    cer_numer = 0
    cer_denom = 0
    records: List[Tuple[Path, str, str]] = []
    with torch.no_grad():
        for inputs, labels, paths in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            preds = greedy_decode_batch(logits, charset)
            for path, label, pred in zip(paths, labels, preds):
                total += 1
                if pred == label:
                    correct += 1
                cer_numer += levenshtein(pred, label)
                cer_denom += max(1, len(label))
                records.append((path, label, pred))
    accuracy = correct / max(1, total)
    cer = cer_numer / max(1, cer_denom)
    return accuracy, cer, records


__all__ = [
    "Charset",
    "ConvFeatureExtractor",
    "OCRDataset",
    "TorchHandlerDependencyMixin",
    "TransformerOCRDataset",
    "collate_batch",
    "format_dependency_error",
    "evaluate_model",
    "greedy_decode_batch",
    "labels_to_targets",
    "levenshtein",
    "parse_label_from_filename",
    "resolve_device",
    "set_seed",
    "build_charset_from_dataset",
    "_missing_dependencies",
    "SUPPORTED_EXTENSIONS",
    "NUMPY_AVAILABLE",
    "PIL_AVAILABLE",
    "TORCH_AVAILABLE",
    "torch",
    "nn",
    "optim",
    "DataLoader",
    "Dataset",
    "random_split",
    "np",
    "Image",
]
