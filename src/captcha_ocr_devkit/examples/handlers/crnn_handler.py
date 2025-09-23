"""CRNN-based OCR handlers for 4-character lowercase captchas."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from captcha_ocr_devkit.core.handlers.base import (
    BaseEvaluateHandler,
    BaseOCRHandler,
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
    build_charset_from_dataset,
    collate_batch,
    evaluate_model,
    format_dependency_error,
    greedy_decode_batch,
    labels_to_targets,
    levenshtein,
    resolve_device,
    set_seed,
    _missing_dependencies,
    TORCH_AVAILABLE,
    torch,
    nn,
    optim,
    DataLoader,
    random_split,
)
from captcha_ocr_devkit.examples.handlers.transformer_handler import TransformerPreprocessHandler

CRNN_HANDLER_VERSION = "1.20251001.0000"
CRNN_DEPENDENCIES = ["torch", "torchvision", "pillow", "numpy"]
CRNN_REQUIREMENTS_FILE = "crnn_handler-requirements.txt"
DEFAULT_IMG_HEIGHT = TransformerPreprocessHandler.DEFAULT_IMG_HEIGHT
DEFAULT_IMG_WIDTH = TransformerPreprocessHandler.DEFAULT_IMG_WIDTH


class CRNNDependencyMixin(TorchHandlerDependencyMixin):
    """Override requirements path defaults for CRNN handlers."""

    REQUIREMENTS_FILE = CRNN_REQUIREMENTS_FILE


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False
if LOGGER.getEffectiveLevel() > logging.INFO:
    LOGGER.setLevel(logging.INFO)


def _ensure_torch_available() -> None:
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for CRNN handlers. Please install torch and torchvision.")


def _as_bool(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


class CRNNPreprocessHandler(CRNNDependencyMixin, TransformerPreprocessHandler):
    """Resize and normalize captcha images for the CRNN pipeline."""

    DESCRIPTION = "Resize captcha images and normalize them for CRNN OCR workflows."
    SHORT_DESCRIPTION = "Preprocess captcha images for CRNN OCR."
    REQUIRED_DEPENDENCIES = CRNN_DEPENDENCIES
    HANDLER_ID = "crnn_preprocess"

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["version"] = CRNN_HANDLER_VERSION
        return info


class CRNNOCRModel(nn.Module):
    """Convolutional-recurrent architecture for captcha OCR with CTC decoding."""

    def __init__(
        self,
        num_classes: int,
        cnn_out_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.cnn = ConvFeatureExtractor(out_dim=cnn_out_dim)
        self.rnn = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        rnn_output_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Linear(rnn_output_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        rnn_out, _ = self.rnn(features)
        logits = self.classifier(rnn_out)
        return logits


def train_crnn_one_epoch(
    model: CRNNOCRModel,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    charset: Charset,
    device: torch.device,
) -> float:
    log_interval = getattr(loader, "_log_interval", 0)
    epoch_index = getattr(loader, "_epoch_index", None)
    total_epochs = getattr(loader, "_total_epochs", None)
    model.train()
    running_loss = 0.0
    total_items = 0
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
        total_items += batch_size

        if log_interval and batch_index % log_interval == 0 and LOGGER.isEnabledFor(logging.INFO):
            epoch_msg = ""
            if epoch_index is not None and total_epochs is not None:
                epoch_msg = f" (epoch {epoch_index}/{total_epochs})"
            LOGGER.info(
                "CRNN training%s - batch %d/%d avg_loss=%.4f",
                epoch_msg,
                batch_index,
                len(loader),
                running_loss / max(1, total_items),
            )
    return running_loss / max(1, len(loader.dataset))


class CRNNTrainHandler(CRNNDependencyMixin, BaseTrainHandler):
    """Train the CRNN OCR model using CTC loss."""

    DESCRIPTION = "Train a CRNN (CNN + bidirectional LSTM) captcha OCR model with CTC decoding."
    SHORT_DESCRIPTION = "Train CRNN OCR for 4-char captchas."
    REQUIRED_DEPENDENCIES = CRNN_DEPENDENCIES
    HANDLER_ID = "crnn_train"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.img_h = int(cfg.get("img_height", DEFAULT_IMG_HEIGHT))
        self.img_w = int(cfg.get("img_width", DEFAULT_IMG_WIDTH))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))
        self.num_workers = int(cfg.get("num_workers", 0))
        self.device_name = cfg.get("device", "auto")
        self.log_interval = max(0, int(cfg.get("log_interval", 50)))
        self.hidden_size = int(cfg.get("hidden_size", 256))
        self.num_layers = int(cfg.get("num_layers", 2))
        self.bidirectional = _as_bool(cfg.get("bidirectional", True))
        self.dropout = float(cfg.get("dropout", 0.1))

    def train(self, config: TrainingConfig) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=format_dependency_error(missing, self._install_hint()))

        _ensure_torch_available()

        if not config.input_dir.exists():
            return HandlerResult(success=False, error=f"Training data directory not found: {config.input_dir}")

        set_seed(config.seed)
        device = resolve_device(config.device if config.device != "auto" else self.device_name)

        try:
            dataset = OCRDataset(
                config.input_dir,
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

        model = CRNNOCRModel(
            num_classes=charset.size,
            cnn_out_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional=self.bidirectional,
            dropout=self.dropout,
        )
        model.to(device)
        criterion = nn.CTCLoss(blank=charset.blank_idx, reduction="mean", zero_infinity=True)
        optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=self.weight_decay)

        history: List[Dict[str, Any]] = []
        best_acc = -1.0
        best_cer = float("inf")

        LOGGER.info(
            "CRNN training configured: version=%s epochs=%d, batches=%d, device=%s, log_interval=%d",
            CRNN_HANDLER_VERSION,
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
                f"[CRNNTrainHandler] epoch {epoch}/{config.epochs} started (version {CRNN_HANDLER_VERSION})",
                flush=True,
            )

            train_loss = train_crnn_one_epoch(model, train_loader, optimizer, criterion, charset, device)
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
                f"[CRNNTrainHandler] epoch {epoch}/{config.epochs} finished loss={train_loss:.4f}{extra}",
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
                    "handler_version": CRNN_HANDLER_VERSION,
                    "model_config": {
                        "hidden_size": self.hidden_size,
                        "num_layers": self.num_layers,
                        "bidirectional": self.bidirectional,
                        "dropout": self.dropout,
                    },
                }
                if not self.save_model(checkpoint, config.output_path):
                    return HandlerResult(success=False, error="Failed to save CRNN checkpoint")

        metadata = {
            "device": str(device),
            "charset_size": charset.size,
            "total_samples": total_samples,
            "handler_version": CRNN_HANDLER_VERSION,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        }
        data = {
            "model_path": str(config.output_path),
            "history": history,
            "best_val_accuracy": best_acc if best_acc >= 0 else None,
            "best_val_cer": best_cer if best_cer != float("inf") else None,
        }

        return HandlerResult(success=True, data=data, metadata=metadata)

    def save_model(self, model_data: Any, output_path: Path) -> bool:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_data, str(output_path))
            return True
        except Exception:
            return False

    def load_model(self, model_path: Path) -> Any:
        _ensure_torch_available()
        return torch.load(str(model_path), map_location="cpu")

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": CRNN_HANDLER_VERSION,
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
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        }


class CRNNEvaluateHandler(CRNNDependencyMixin, BaseEvaluateHandler):
    """Evaluate CRNN OCR checkpoints on labeled datasets."""

    DESCRIPTION = "Evaluate CRNN OCR checkpoints with captcha- and character-level metrics."
    SHORT_DESCRIPTION = "Evaluate CRNN captcha OCR checkpoints."
    REQUIRED_DEPENDENCIES = CRNN_DEPENDENCIES
    HANDLER_ID = "crnn_evaluate"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.batch_size = int(cfg.get("batch_size", 16))
        self.device_name = cfg.get("device", "auto")
        self.num_workers = int(cfg.get("num_workers", 0))

    def evaluate(self, model_path: Path, test_data_path: Path) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=format_dependency_error(missing, self._install_hint()))

        _ensure_torch_available()

        if not model_path.exists():
            return HandlerResult(success=False, error=f"Checkpoint not found: {model_path}")
        if not test_data_path.exists():
            return HandlerResult(success=False, error=f"Test data directory not found: {test_data_path}")

        checkpoint = torch.load(str(model_path), map_location="cpu")
        charset_list = checkpoint.get("charset")
        if not charset_list:
            return HandlerResult(success=False, error="Checkpoint missing charset information")
        charset = Charset(charset_list)
        img_h = int(checkpoint.get("img_h", DEFAULT_IMG_HEIGHT))
        img_w = int(checkpoint.get("img_w", DEFAULT_IMG_WIDTH))
        model_cfg = checkpoint.get("model_config", {})

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
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )

        device = resolve_device(self.device_name)
        model = CRNNOCRModel(
            num_classes=charset.size,
            cnn_out_dim=int(model_cfg.get("hidden_size", 256)),
            hidden_size=int(model_cfg.get("hidden_size", 256)),
            num_layers=int(model_cfg.get("num_layers", 2)),
            bidirectional=_as_bool(model_cfg.get("bidirectional", True)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        accuracy, cer, records = evaluate_model(model, loader, charset, device)
        ground_truth = [label for _, label, _ in records]
        predictions = [pred for _, _, pred in records]
        metrics = self.calculate_metrics(predictions, ground_truth)
        metrics.accuracy = accuracy
        metrics.character_accuracy = 1.0 - cer
        metrics.total_samples = len(dataset)
        metrics.correct_predictions = int(round(accuracy * len(dataset)))

        LOGGER.info(
            "CRNN evaluation processed %d samples: accuracy=%.4f, char_accuracy=%.4f (%d correct)",
            metrics.total_samples,
            metrics.accuracy,
            metrics.character_accuracy,
            metrics.correct_predictions,
        )

        metadata = {
            "device": str(device),
            "handler_version": CRNN_HANDLER_VERSION,
            "model_config": model_cfg,
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
            "version": CRNN_HANDLER_VERSION,
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


class CRNNOCRHandler(CRNNDependencyMixin, BaseOCRHandler):
    """Inference handler that wraps the CRNN OCR model."""

    DESCRIPTION = "Predict 4-character captchas using a CRNN with CTC decoding."
    SHORT_DESCRIPTION = "Inference for CRNN captcha OCR."
    REQUIRED_DEPENDENCIES = CRNN_DEPENDENCIES
    HANDLER_ID = "crnn_ocr"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.device_name = cfg.get("device", "auto")
        self.img_h = int(cfg.get("img_height", DEFAULT_IMG_HEIGHT))
        self.img_w = int(cfg.get("img_width", DEFAULT_IMG_WIDTH))
        self.hidden_size = int(cfg.get("hidden_size", 256))
        self.num_layers = int(cfg.get("num_layers", 2))
        self.bidirectional = _as_bool(cfg.get("bidirectional", True))
        self.dropout = float(cfg.get("dropout", 0.1))
        self.charset: Optional[Charset] = None
        self.model: Optional[CRNNOCRModel] = None
        self.device: Optional[torch.device] = None

    def load_model(self, model_path: Path) -> bool:
        missing = _missing_dependencies()
        if missing:
            raise RuntimeError(format_dependency_error(missing, self._install_hint()))

        _ensure_torch_available()

        try:
            checkpoint = torch.load(str(model_path), map_location="cpu")
            charset_list = checkpoint.get("charset")
            if not charset_list:
                raise ValueError("Checkpoint missing 'charset'")
            self.charset = Charset(charset_list)
            self.img_h = int(checkpoint.get("img_h", self.img_h))
            self.img_w = int(checkpoint.get("img_w", self.img_w))
            cfg = checkpoint.get("model_config", {})
            self.hidden_size = int(cfg.get("hidden_size", self.hidden_size))
            self.num_layers = int(cfg.get("num_layers", self.num_layers))
            self.bidirectional = _as_bool(cfg.get("bidirectional", self.bidirectional))
            self.dropout = float(cfg.get("dropout", self.dropout))

            self.model = CRNNOCRModel(
                num_classes=self.charset.size,
                cnn_out_dim=self.hidden_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                bidirectional=self.bidirectional,
                dropout=self.dropout,
            )
            self.model.load_state_dict(checkpoint["model"])
            self.device = resolve_device(self.device_name)
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(f"Failed to load CRNN OCR checkpoint: {exc}")

    def predict(self, processed_image: Any) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=format_dependency_error(missing, self._install_hint()))
        if self.model is None or self.charset is None or self.device is None:
            return HandlerResult(success=False, error="Model not loaded. Call load_model() first.")

        try:
            if isinstance(processed_image, torch.Tensor):
                tensor = processed_image
                metadata: Dict[str, Any] = {}
            else:
                preprocess = CRNNPreprocessHandler(
                    "temp",
                    {
                        "img_height": self.img_h,
                        "img_width": self.img_w,
                        "requirements_file": self._requirements_override(),
                    },
                )
                preprocess_result = preprocess.process(processed_image)
                if not preprocess_result.success:
                    return preprocess_result
                tensor = preprocess_result.data
                metadata = preprocess_result.metadata or {}

            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
            tensor = tensor.to(self.device)

            logits = self.model(tensor)
            prediction = greedy_decode_batch(logits, self.charset)[0]

            metadata.update(
                {
                    "handler_version": CRNN_HANDLER_VERSION,
                    "charset_size": self.charset.size,
                    "hidden_size": self.hidden_size,
                    "num_layers": self.num_layers,
                    "bidirectional": self.bidirectional,
                }
            )
            return HandlerResult(success=True, data=prediction, metadata=metadata)
        except Exception as exc:  # pragma: no cover - defensive branch
            return HandlerResult(success=False, error=str(exc))

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": CRNN_HANDLER_VERSION,
            "description": self.get_description(),
            "short_description": self.get_short_description(),
            "dependencies": self.get_dependencies(),
            "dependency_status": self.get_dependency_status(),
            "missing_dependencies": self.get_missing_dependencies(),
            "requirements_file": str(self._requirements_file_path()),
            "install_hint": self._install_hint(),
            "device": self.device_name,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        }


__all__ = [
    "CRNNPreprocessHandler",
    "CRNNTrainHandler",
    "CRNNEvaluateHandler",
    "CRNNOCRHandler",
]
