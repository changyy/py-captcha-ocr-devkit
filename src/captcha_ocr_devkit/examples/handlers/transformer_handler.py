"""Transformer-based OCR handlers for 4-character captchas."""

from __future__ import annotations

import io
import random
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from captcha_ocr_devkit.core.handlers.base import (
    BaseEvaluateHandler,
    BaseOCRHandler,
    BasePreprocessHandler,
    BaseTrainHandler,
    EvaluationResult,
    HandlerResult,
    TrainingConfig,
)

from captcha_ocr_devkit.examples.handlers.ocr_common import (
    Charset,
    ConvFeatureExtractor,
    OCRDataset,
    TorchHandlerDependencyMixin,
    _missing_dependencies,
    build_charset_from_dataset,
    collate_batch,
    evaluate_model,
    greedy_decode_batch,
    labels_to_targets,
    levenshtein,
    resolve_device,
    set_seed,
    Image,
    np,
    torch,
    nn,
    optim,
    DataLoader,
    random_split,
    NUMPY_AVAILABLE,
    PIL_AVAILABLE,
    TORCH_AVAILABLE,
)

# ---------------------------------------------------------------------------
# Dependency helpers
# ---------------------------------------------------------------------------

TRANSFORMER_HANDLER_VERSION = "1.20250919.1700"
TRANSFORMER_DEPENDENCIES = ["torch", "torchvision", "pillow", "numpy"]
TRANSFORMER_REQUIREMENTS_FILE = "transformer_handler-requirements.txt"
TRANSFORMER_INSTALL_FALLBACK = "pip install torch torchvision pillow numpy"

class TransformerDependencyMixin(TorchHandlerDependencyMixin):
    REQUIREMENTS_FILE = TRANSFORMER_REQUIREMENTS_FILE
    INSTALL_FALLBACK = TRANSFORMER_INSTALL_FALLBACK


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False
if LOGGER.getEffectiveLevel() > logging.INFO:
    LOGGER.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class PositionalEncoding(nn.Module):
        def __init__(self, d_model: int, max_len: int = 2000):
            super().__init__()
            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            length = x.size(1)
            return x + self.pe[:, :length]


    class OCRModel(nn.Module):
        def __init__(self, num_classes: int, d_model: int = 256, num_layers: int = 2):
            super().__init__()
            self.backbone = ConvFeatureExtractor(out_dim=d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=512)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.positional_encoding = PositionalEncoding(d_model)
            self.classifier = nn.Linear(d_model, num_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            feats = self.backbone(x)
            feats = self.positional_encoding(feats)
            feats = feats.permute(1, 0, 2)
            encoded = self.encoder(feats)
            logits = self.classifier(encoded)
            return logits.permute(1, 0, 2)

else:  # pragma: no cover - fallback when torch missing

    class PositionalEncoding:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for transformer handlers. Please install torch and torchvision.")

    class OCRModel:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for transformer handlers. Please install torch and torchvision.")


# ---------------------------------------------------------------------------
# Training utilities adapted from reference script
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: OCRModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    charset: Charset,
    device: torch.device,
) -> float:
    log_interval = getattr(loader, "_log_interval", 0)  # type: ignore[attr-defined]
    epoch_index = getattr(loader, "_epoch_index", None)  # type: ignore[attr-defined]
    total_epochs = getattr(loader, "_total_epochs", None)  # type: ignore[attr-defined]
    global_step = 0
    model.train()
    running_loss = 0.0
    for batch_index, (inputs, labels, _) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        logits = model(inputs)
        batch_size, time_steps, _ = logits.shape
        log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
        input_lengths = torch.full((batch_size,), time_steps, dtype=torch.long, device=device)
        targets, target_lengths = labels_to_targets(labels, charset)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)
        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_size

        global_step += batch_size
        if log_interval and batch_index % log_interval == 0 and LOGGER.isEnabledFor(logging.INFO):
            epoch_msg = ""
            if epoch_index is not None and total_epochs is not None:
                epoch_msg = f" (epoch {epoch_index}/{total_epochs})"
            LOGGER.info(
                "Transformer training%s - batch %d/%d avg_loss=%.4f",
                epoch_msg,
                batch_index,
                len(loader),
                running_loss / global_step,
            )
    return running_loss / max(1, len(loader.dataset))


def greedy_decode_batch(logits: torch.Tensor, charset: Charset) -> List[str]:
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
    model: OCRModel,
    loader: DataLoader,
    charset: Charset,
    device: torch.device,
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


# ---------------------------------------------------------------------------
# Preprocess Handler
# ---------------------------------------------------------------------------


class TransformerPreprocessHandler(TransformerDependencyMixin, BasePreprocessHandler):
    """Resize, normalize, and tensorize CAPTCHA images for the transformer."""

    DESCRIPTION = "Resize captcha images, normalize pixel intensities, and pack tensors for the transformer OCR pipeline."
    SHORT_DESCRIPTION = "Prepare 4-char captcha images for transformer OCR."
    REQUIRED_DEPENDENCIES = TRANSFORMER_DEPENDENCIES
    HANDLER_ID = "transformer_preprocess"

    DEFAULT_IMG_HEIGHT = 32
    DEFAULT_IMG_WIDTH = 256

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.img_h = int(cfg.get("img_height", self.DEFAULT_IMG_HEIGHT))
        self.img_w = int(cfg.get("img_width", self.DEFAULT_IMG_WIDTH))

    def get_supported_formats(self) -> List[str]:
        return [".png", ".jpg", ".jpeg", ".bmp"]

    def _load_image(self, image_data: Union[bytes, str, Path]) -> Image.Image:
        if not PIL_AVAILABLE:
            raise RuntimeError(self._dependency_error_message(["Pillow"]))
        if isinstance(image_data, bytes):
            return Image.open(io.BytesIO(image_data)).convert("L")
        if isinstance(image_data, (str, Path)):
            return Image.open(str(image_data)).convert("L")
        raise TypeError("Unsupported image_data type")

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

    def process(self, image_data: Union[bytes, str, Path]) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=self._dependency_error_message(missing))
        try:
            pil_image = self._load_image(image_data)
            original_size = pil_image.size
            processed_img = self._resize_pad(pil_image)
            tensor = torch.from_numpy(np.array(processed_img)).float().unsqueeze(0).unsqueeze(0) / 255.0  # type: ignore[arg-type]
            metadata = {
                "preprocess_handler": self.name,
                "img_height": self.img_h,
                "img_width": self.img_w,
                "image_size": {
                    "original": {"width": original_size[0], "height": original_size[1]},
                    "processed": {"width": processed_img.size[0], "height": processed_img.size[1]},
                },
            }
            return HandlerResult(success=True, data=tensor, metadata=metadata)
        except Exception as exc:  # pragma: no cover - defensive branch
            return HandlerResult(success=False, error=str(exc))

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": TRANSFORMER_HANDLER_VERSION,
            "description": self.get_description(),
            "short_description": self.get_short_description(),
            "dependencies": self.get_dependencies(),
            "dependency_status": self.get_dependency_status(),
            "missing_dependencies": self.get_missing_dependencies(),
            "requirements_file": str(self._requirements_file_path()),
            "install_hint": self._install_hint(),
            "img_height": self.img_h,
            "img_width": self.img_w,
        }


# ---------------------------------------------------------------------------
# Training Handler
# ---------------------------------------------------------------------------


class TransformerTrainHandler(TransformerDependencyMixin, BaseTrainHandler):
    """Train the transformer OCR model using repository conventions."""

    DESCRIPTION = "Train the transformer-based OCR model on 4-character captcha datasets with CTC loss."
    SHORT_DESCRIPTION = "Train transformer OCR for 4-char captchas."
    REQUIRED_DEPENDENCIES = TRANSFORMER_DEPENDENCIES
    HANDLER_ID = "transformer_train"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.img_h = int(cfg.get("img_height", TransformerPreprocessHandler.DEFAULT_IMG_HEIGHT))
        self.img_w = int(cfg.get("img_width", TransformerPreprocessHandler.DEFAULT_IMG_WIDTH))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))
        self.num_workers = int(cfg.get("num_workers", 0))
        self.device_name = cfg.get("device", "auto")
        self.log_interval = max(0, int(cfg.get("log_interval", 50)))

    def _ensure_torch(self) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(self._dependency_error_message(["torch", "torchvision"]))

    def train(self, config: TrainingConfig) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=self._dependency_error_message(missing))

        self._ensure_torch()

        input_dir = config.input_dir
        if not input_dir.exists():
            return HandlerResult(success=False, error=f"Training data directory not found: {input_dir}")

        set_seed(config.seed)
        device = resolve_device(config.device if config.device != "auto" else self.device_name)

        try:
            dataset = OCRDataset(
                input_dir,
                self.img_h,
                self.img_w,
                requirements_override=self._requirements_override(),
            )
            charset = build_charset_from_dataset(dataset)
        except Exception as exc:
            return HandlerResult(success=False, error=str(exc))

        val_split = float(config.validation_split)
        total_samples = len(dataset)
        val_size = 0
        if total_samples > 1 and val_split > 0:
            val_size = max(1, int(total_samples * val_split))
            if val_size >= total_samples:
                val_size = max(1, total_samples // 5)
        train_size = total_samples - val_size
        if train_size <= 0:
            train_size = max(1, total_samples - 1)
            val_size = total_samples - train_size

        if val_size > 0 and random_split is not None:
            train_ds, val_ds = random_split(dataset, [train_size, val_size])
        else:
            train_ds, val_ds = dataset, None

        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )
        if self.log_interval:
            setattr(train_loader, "_log_interval", self.log_interval)
            setattr(train_loader, "_total_epochs", config.epochs)

        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_batch,
            )

        model = OCRModel(num_classes=charset.size)
        model.to(device)
        criterion = nn.CTCLoss(blank=charset.blank_idx, reduction="mean", zero_infinity=True)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=self.weight_decay)

        history: List[Dict[str, Any]] = []
        best_acc = -1.0
        best_cer = float("inf")

        LOGGER.info(
            "Transformer training configured: version=%s epochs=%d, batches=%d, device=%s, log_interval=%d",
            TRANSFORMER_HANDLER_VERSION,
            config.epochs,
            len(train_loader),
            device,
            self.log_interval,
        )

        for epoch in range(1, config.epochs + 1):
            if self.log_interval:
                setattr(train_loader, "_epoch_index", epoch)
            LOGGER.info("Epoch %d/%d started", epoch, config.epochs)
            print(
                f"[TransformerTrainHandler] epoch {epoch}/{config.epochs} started (version {TRANSFORMER_HANDLER_VERSION})",
                flush=True,
            )
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, charset, device)
            val_acc = None
            val_cer = None
            if val_loader is not None:
                val_acc, val_cer, _ = evaluate_model(model, val_loader, charset, device)
            LOGGER.info(
                "Epoch %d/%d finished: loss=%.4f%s",
                epoch,
                config.epochs,
                train_loss,
                f", val_acc={val_acc:.4f}, val_cer={val_cer:.4f}" if val_acc is not None else "",
            )
            extra = ""
            if val_acc is not None:
                extra = f", val_acc={val_acc:.4f}, val_cer={val_cer:.4f}"
            print(
                f"[TransformerTrainHandler] epoch {epoch}/{config.epochs} finished loss={train_loss:.4f}{extra}",
                flush=True,
            )
            history.append(
                {
                    "epoch": epoch,
                    "loss": train_loss,
                    "val_accuracy": val_acc,
                    "val_cer": val_cer,
                }
            )

            should_save = val_loader is None or (val_acc is not None and val_acc >= best_acc)

            if should_save:
                if val_acc is not None:
                    best_acc = max(best_acc, val_acc)
                if val_cer is not None:
                    best_cer = min(best_cer, val_cer)
                checkpoint = {
                    "model": model.state_dict(),
                    "charset": charset.itos,
                    "img_h": self.img_h,
                    "img_w": self.img_w,
                    "handler_version": TRANSFORMER_HANDLER_VERSION,
                }
                if not self.save_model(checkpoint, config.output_path):
                    return HandlerResult(success=False, error="Failed to save checkpoint")

        metadata = {
            "device": str(device),
            "charset_size": charset.size,
            "total_samples": total_samples,
            "handler_version": TRANSFORMER_HANDLER_VERSION,
        }

        result_data = {
            "model_path": str(config.output_path),
            "history": history,
            "best_val_accuracy": best_acc if best_acc >= 0 else None,
            "best_val_cer": best_cer if best_cer != float("inf") else None,
        }

        return HandlerResult(success=True, data=result_data, metadata=metadata)

    def save_model(self, model_data: Any, output_path: Path) -> bool:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_data, str(output_path))
            return True
        except Exception:
            return False

    def load_model(self, model_path: Path) -> Any:
        return torch.load(str(model_path), map_location="cpu")

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": TRANSFORMER_HANDLER_VERSION,
            "description": self.get_description(),
            "short_description": self.get_short_description(),
            "dependencies": self.get_dependencies(),
            "dependency_status": self.get_dependency_status(),
            "missing_dependencies": self.get_missing_dependencies(),
            "requirements_file": str(self._requirements_file_path()),
            "install_hint": self._install_hint(),
            "img_height": self.img_h,
            "img_width": self.img_w,
            "device": self.device_name,
            "weight_decay": self.weight_decay,
            "num_workers": self.num_workers,
            "log_interval": self.log_interval,
        }


# ---------------------------------------------------------------------------
# Evaluation Handler
# ---------------------------------------------------------------------------


class TransformerEvaluateHandler(TransformerDependencyMixin, BaseEvaluateHandler):
    """Evaluate the transformer OCR model on labeled datasets."""

    DESCRIPTION = "Evaluate transformer OCR checkpoints against labeled captcha datasets and report accuracy metrics."
    SHORT_DESCRIPTION = "Evaluate transformer OCR checkpoints."
    REQUIRED_DEPENDENCIES = TRANSFORMER_DEPENDENCIES
    HANDLER_ID = "transformer_evaluate"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.batch_size = int(cfg.get("batch_size", 16))
        self.device_name = cfg.get("device", "auto")

    def _ensure_torch(self) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(self._dependency_error_message(["torch", "torchvision"]))

    def evaluate(self, model_path: Path, test_data_path: Path) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=self._dependency_error_message(missing))

        self._ensure_torch()

        if not model_path.exists():
            return HandlerResult(success=False, error=f"Checkpoint not found: {model_path}")
        if not test_data_path.exists():
            return HandlerResult(success=False, error=f"Test data directory not found: {test_data_path}")

        checkpoint = torch.load(str(model_path), map_location="cpu")
        charset_list = checkpoint.get("charset")
        if not charset_list:
            return HandlerResult(success=False, error="Checkpoint missing charset information")
        charset = Charset(charset_list)
        img_h = int(checkpoint.get("img_h", TransformerPreprocessHandler.DEFAULT_IMG_HEIGHT))
        img_w = int(checkpoint.get("img_w", TransformerPreprocessHandler.DEFAULT_IMG_WIDTH))

        try:
            dataset = OCRDataset(
                test_data_path,
                img_h,
                img_w,
                requirements_override=self._requirements_override(),
            )
        except Exception as exc:
            return HandlerResult(success=False, error=str(exc))

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch,
        )

        device = resolve_device(self.device_name)
        model = OCRModel(num_classes=charset.size)
        model.load_state_dict(checkpoint["model"])
        model.to(device)

        accuracy, cer, records = evaluate_model(model, loader, charset, device)
        ground_truth = [label for _, label, _ in records]
        predictions = [pred for _, _, pred in records]
        metrics = self.calculate_metrics(predictions, ground_truth)
        metrics.accuracy = accuracy
        metrics.character_accuracy = 1.0 - cer
        metrics.total_samples = len(dataset)
        metrics.correct_predictions = int(round(accuracy * len(dataset)))

        LOGGER.info(
            "Transformer evaluation processed %d samples: accuracy=%.4f, char_accuracy=%.4f (%d correct)",
            metrics.total_samples,
            metrics.accuracy,
            metrics.character_accuracy,
            metrics.correct_predictions,
        )

        metadata = {
            "device": str(device),
            "handler_version": TRANSFORMER_HANDLER_VERSION,
        }

        data = {
            "model_path": str(model_path),
            "test_data_path": str(test_data_path),
            "accuracy": metrics.accuracy,
            "character_accuracy": metrics.character_accuracy,
            "predictions": [
                {
                    "path": str(path),
                    "label": label,
                    "prediction": pred,
                    "correct": pred == label,
                }
                for path, label, pred in records
            ],
        }

        return HandlerResult(success=True, data=data, metadata=metadata)

    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> EvaluationResult:
        total = len(ground_truth)
        correct = sum(1 for pred, truth in zip(predictions, ground_truth) if pred == truth)
        total_chars = sum(len(truth) for truth in ground_truth)
        char_errors = sum(levenshtein(pred, truth) for pred, truth in zip(predictions, ground_truth))
        char_accuracy = (total_chars - char_errors) / max(1, total_chars)
        return EvaluationResult(
            accuracy=correct / max(1, total),
            total_samples=total,
            correct_predictions=correct,
            character_accuracy=char_accuracy,
        )

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": TRANSFORMER_HANDLER_VERSION,
            "description": self.get_description(),
            "short_description": self.get_short_description(),
            "dependencies": self.get_dependencies(),
            "dependency_status": self.get_dependency_status(),
            "missing_dependencies": self.get_missing_dependencies(),
            "requirements_file": str(self._requirements_file_path()),
            "install_hint": self._install_hint(),
            "batch_size": self.batch_size,
            "device": self.device_name,
        }


# ---------------------------------------------------------------------------
# OCR Handler
# ---------------------------------------------------------------------------


class TransformerOCRHandler(TransformerDependencyMixin, BaseOCRHandler):
    """Inference handler that wraps the transformer OCR model."""

    DESCRIPTION = "Run transformer OCR inference on preprocessed captcha images and decode 4-character predictions."
    SHORT_DESCRIPTION = "Inference for transformer captcha OCR."
    REQUIRED_DEPENDENCIES = TRANSFORMER_DEPENDENCIES
    HANDLER_ID = "transformer_ocr"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.device_name = cfg.get("device", "auto")
        self.charset: Optional[Charset] = None
        self.model: Optional[OCRModel] = None
        self.device: Optional[torch.device] = None
        self.img_h = int(cfg.get("img_height", TransformerPreprocessHandler.DEFAULT_IMG_HEIGHT))
        self.img_w = int(cfg.get("img_width", TransformerPreprocessHandler.DEFAULT_IMG_WIDTH))

    def _ensure_torch(self) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(self._dependency_error_message(["torch", "torchvision"]))

    def load_model(self, model_path: Path) -> bool:
        missing = _missing_dependencies()
        if missing:
            raise RuntimeError(self._dependency_error_message(missing))
        try:
            self._ensure_torch()
            ckpt = torch.load(str(model_path), map_location="cpu")
            charset_list = ckpt.get("charset")
            if not charset_list:
                raise ValueError("Checkpoint missing 'charset'")
            self.charset = Charset(charset_list)
            self.model = OCRModel(num_classes=self.charset.size)
            self.model.load_state_dict(ckpt["model"])
            self.device = resolve_device(self.device_name)
            self.model.to(self.device)
            self.model.eval()
            self.img_h = int(ckpt.get("img_h", self.img_h))
            self.img_w = int(ckpt.get("img_w", self.img_w))
            return True
        except Exception as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(f"Failed to load transformer OCR checkpoint: {exc}")

    def predict(self, processed_image: Any) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=self._dependency_error_message(missing))
        if self.model is None or self.charset is None:
            return HandlerResult(success=False, error="Model not loaded. Call load_model() first.")

        try:
            preprocess_metadata: Dict[str, Any] = {}
            if isinstance(processed_image, torch.Tensor):
                tensor = processed_image
            else:
                preprocess = TransformerPreprocessHandler(
                    "temp",
                    {
                        "img_height": self.img_h,
                        "img_width": self.img_w,
                        "requirements_file": self._requirements_override(),
                    },
                )
                tensor_result = preprocess.process(processed_image)
                if not tensor_result.success:
                    return tensor_result
                tensor = tensor_result.data
                preprocess_metadata = tensor_result.metadata or {}

            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self.device)

            start = time.time()
            with torch.no_grad():
                logits = self.model(tensor)
            inference_time = time.time() - start

            probs = torch.softmax(logits, dim=-1)
            avg_conf = probs.max(dim=-1)[0].mean().item()
            decoded = self.charset.decode_greedy(logits[0].cpu())

            character_confidences: List[float] = []
            if decoded:
                prev_idx = self.charset.blank_idx
                for timestep in probs[0]:
                    conf, idx = timestep.max(dim=-1)
                    idx_item = idx.item()
                    conf_item = conf.item()
                    if idx_item != self.charset.blank_idx and idx_item != prev_idx:
                        character_confidences.append(conf_item)
                    prev_idx = idx_item

            metadata = {
                "inference_time": inference_time,
                "confidence": float(avg_conf),
                "handler_version": TRANSFORMER_HANDLER_VERSION,
                "alphabet_size": self.charset.size,
                "character_confidences": character_confidences,
                "character_count": len(decoded),
            }
            if preprocess_metadata.get("image_size"):
                metadata["image_size"] = preprocess_metadata["image_size"]
            elif self.img_h and self.img_w:
                metadata["image_size"] = {
                    "processed": {"width": self.img_w, "height": self.img_h}
                }
            return HandlerResult(success=True, data=decoded, metadata=metadata)
        except Exception as exc:  # pragma: no cover
            return HandlerResult(success=False, error=str(exc))

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": TRANSFORMER_HANDLER_VERSION,
            "description": self.get_description(),
            "short_description": self.get_short_description(),
            "dependencies": self.get_dependencies(),
            "dependency_status": self.get_dependency_status(),
            "missing_dependencies": self.get_missing_dependencies(),
            "requirements_file": str(self._requirements_file_path()),
            "install_hint": self._install_hint(),
            "device": self.device_name,
            "model_loaded": self.model is not None,
            "img_height": self.img_h,
            "img_width": self.img_w,
        }


__all__ = [
    "TRANSFORMER_HANDLER_VERSION",
    "TRANSFORMER_DEPENDENCIES",
    "TRANSFORMER_REQUIREMENTS_FILE",
    "Charset",
    "TransformerPreprocessHandler",
    "TransformerTrainHandler",
    "TransformerEvaluateHandler",
    "TransformerOCRHandler",
]
