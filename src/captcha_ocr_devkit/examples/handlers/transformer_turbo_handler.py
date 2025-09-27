"""Transformer Turbo OCR handlers - Enhanced transformer with increased model capacity.

🚀 Transformer Turbo represents the next evolution of transformer-based OCR, featuring:
- Enhanced model capacity (d_model=384, 4 layers, 8 attention heads)
- Advanced training strategies (cosine annealing, improved regularization)
- Optimized architecture for 96%+ accuracy targets
- Full backward compatibility with standard transformer workflow
"""

from __future__ import annotations

import io
import math
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

TRANSFORMER_TURBO_HANDLER_VERSION = "1.20250925.2000"  # Turbo version
TRANSFORMER_TURBO_DEPENDENCIES = ["torch", "torchvision", "pillow", "numpy"]
TRANSFORMER_TURBO_REQUIREMENTS_FILE = "transformer_turbo_handler-requirements.txt"
TRANSFORMER_TURBO_INSTALL_FALLBACK = "pip install torch torchvision pillow numpy"

class TransformerTurboDependencyMixin(TorchHandlerDependencyMixin):
    REQUIREMENTS_FILE = TRANSFORMER_TURBO_REQUIREMENTS_FILE
    INSTALL_FALLBACK = TRANSFORMER_TURBO_INSTALL_FALLBACK


LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False
if LOGGER.getEffectiveLevel() > logging.INFO:
    LOGGER.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _coerce_positive_int(value: Any, default: int) -> int:
    """Safely parse a positive integer from configs or fall back to default."""

    try:
        parsed = int(value)
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def _coerce_bool(value: Any, default: bool) -> bool:
    """Convert various truthy/falsy representations into a boolean."""

    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
        return default
    return bool(value)


def _default_charset_char(charset: Charset) -> str:
    """Pick a non-blank fallback character from the charset."""

    for ch in charset.itos:
        if ch != Charset.BLANK_SYMBOL:
            return ch
    return ""


def _decode_fixed_length(
    logits: "torch.Tensor",
    charset: Charset,
    expected_length: int,
    *,
    force_lowercase: bool = True,
) -> Tuple[str, List[float]]:
    """Decode logits into a fixed-length string using simple segment aggregation."""

    if expected_length <= 0:
        text = charset.decode_greedy(logits)
        return (text.lower() if force_lowercase else text), []

    probs = torch.softmax(logits, dim=-1)
    time_steps = probs.size(0)
    if time_steps == 0:
        fallback_char = _default_charset_char(charset) or ""
        text = fallback_char * expected_length
        return (text.lower() if force_lowercase else text), [0.0] * expected_length

    segment_edges = torch.linspace(0, time_steps, expected_length + 1)
    blank_idx = charset.blank_idx
    fallback_char = _default_charset_char(charset) or ""

    chars: List[str] = []
    confidences: List[float] = []

    for idx in range(expected_length):
        start = int(segment_edges[idx].item())
        end = int(segment_edges[idx + 1].item())
        if end <= start:
            end = min(start + 1, time_steps)
        segment = probs[start:end]
        if segment.numel() == 0:
            segment = probs[start:start + 1]

        segment_mean = segment.mean(dim=0)
        masked = segment_mean.clone()
        if masked.numel() > blank_idx:
            masked[blank_idx] = 0.0

        best_conf, best_idx = torch.max(masked, dim=0)
        if best_conf.item() <= 0.0:
            best_conf, best_idx = torch.max(segment_mean, dim=0)

        char = charset.itos[best_idx.item()] if best_idx.item() < len(charset.itos) else fallback_char
        if char == Charset.BLANK_SYMBOL or char == "":
            char = fallback_char

        chars.append(char)
        confidences.append(float(best_conf.item() * 100.0))

    text = "".join(chars)
    if force_lowercase:
        text = text.lower()

    if len(text) != expected_length:
        text = (text + fallback_char * expected_length)[:expected_length]
    if len(confidences) < expected_length:
        confidences.extend([0.0] * (expected_length - len(confidences)))
    elif len(confidences) > expected_length:
        confidences = confidences[:expected_length]

    return text, confidences


def _extract_character_confidences(
    probs: "torch.Tensor",
    charset: Charset,
    expected_length: int,
) -> List[float]:
    """Collect confidence per character from greedy decoding order."""

    confidences: List[float] = []
    prev_idx = charset.blank_idx
    for timestep in probs:
        conf, idx = timestep.max(dim=-1)
        idx_item = idx.item()
        if idx_item != charset.blank_idx and idx_item != prev_idx:
            confidences.append(float(conf.item() * 100.0))
        prev_idx = idx_item
        if len(confidences) >= expected_length:
            break

    if len(confidences) < expected_length:
        confidences.extend([0.0] * (expected_length - len(confidences)))
    elif len(confidences) > expected_length:
        confidences = confidences[:expected_length]

    return confidences


# ---------------------------------------------------------------------------
# Enhanced Model components for Turbo
# ---------------------------------------------------------------------------

if TORCH_AVAILABLE:

    class EnhancedPositionalEncoding(nn.Module):
        """Enhanced positional encoding with learnable parameters."""

        def __init__(self, d_model: int, max_len: int = 2000, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)

            position = torch.arange(0, max_len).float().unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

            pe = torch.zeros(max_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)

            self.register_buffer("pe", pe)

            # Learnable scaling factor
            self.scale = nn.Parameter(torch.ones(1))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            length = x.size(1)
            x = x + self.scale * self.pe[:, :length]
            return self.dropout(x)

    class TurboOCRModel(nn.Module):
        """Enhanced transformer model with increased capacity."""

        def __init__(
            self,
            num_classes: int,
            d_model: int = 384,      # Increased from 256
            num_layers: int = 4,     # Increased from 2
            nhead: int = 8,          # Increased from 4
            dim_feedforward: int = 1024,  # Increased from 512
            dropout: float = 0.1,
            activation: str = "gelu"  # Using GELU instead of ReLU
        ):
            super().__init__()

            # Enhanced backbone
            self.backbone = ConvFeatureExtractor(out_dim=d_model)

            # Enhanced positional encoding with dropout
            self.positional_encoding = EnhancedPositionalEncoding(d_model, dropout=dropout)

            # Enhanced transformer encoder
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                batch_first=False,  # Keep consistent with original
                norm_first=True     # Pre-layer normalization for better training
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

            # Enhanced classifier with residual connection
            self.pre_classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(d_model, num_classes)

            # Initialize parameters
            self._init_parameters()

            # Log model info
            total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            LOGGER.info(f"🚀 TurboOCRModel initialized: {total_params:,} parameters")
            LOGGER.info(f"📐 Architecture: d_model={d_model}, layers={num_layers}, heads={nhead}")

        def _init_parameters(self):
            """Initialize parameters using Xavier/Glorot initialization."""
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Extract features
            feats = self.backbone(x)  # (batch, seq_len, d_model)

            # Add positional encoding
            feats = self.positional_encoding(feats)

            # Transformer expects (seq_len, batch, d_model)
            feats = feats.permute(1, 0, 2)

            # Transformer encoding
            encoded = self.encoder(feats)  # (seq_len, batch, d_model)

            # Enhanced classification with residual connection
            pre_logits = self.pre_classifier(encoded)  # Enhanced processing
            residual_encoded = encoded + pre_logits    # Residual connection
            logits = self.classifier(residual_encoded)

            # Return as (batch, seq_len, num_classes)
            return logits.permute(1, 0, 2)

else:  # pragma: no cover - fallback when torch missing

    class EnhancedPositionalEncoding:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for transformer turbo handlers. Please install torch and torchvision.")

    class TurboOCRModel:  # type: ignore[override]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            raise RuntimeError("PyTorch is required for transformer turbo handlers. Please install torch and torchvision.")


# ---------------------------------------------------------------------------
# Enhanced Training utilities with advanced strategies
# ---------------------------------------------------------------------------

def train_one_epoch_turbo(
    model: TurboOCRModel,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    charset: Charset,
    device: torch.device,
    scheduler: Optional[Any] = None,
    grad_clip: float = 1.0,
) -> float:
    """Enhanced training loop with gradient clipping and scheduling."""
    model.train()
    running_loss = 0.0
    num_batches = len(train_loader)

    for batch_idx, (images, labels, _) in enumerate(train_loader):
        images = images.to(device)
        targets, target_lengths = labels_to_targets(labels, charset)
        targets = targets.to(device)
        target_lengths = target_lengths.to(device)

        # Forward pass
        optimizer.zero_grad()
        logits = model(images)

        # CTC loss computation
        log_probs = nn.functional.log_softmax(logits, dim=2)
        batch_size, seq_len, num_classes = log_probs.size()
        input_lengths = torch.full((batch_size,), seq_len, dtype=torch.long, device=device)

        loss = criterion(log_probs.permute(1, 0, 2), targets, input_lengths, target_lengths)

        # Backward pass with gradient clipping
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Scheduler step (if using per-step scheduling)
        if scheduler is not None and hasattr(scheduler, 'step') and len(scheduler.state_dict().get('_step_count', [])) == 0:
            scheduler.step()

        running_loss += loss.item()

        # Log progress
        if hasattr(train_loader, '_log_interval') and hasattr(train_loader, '_epoch_index'):
            log_interval = getattr(train_loader, '_log_interval', 0)
            epoch_index = getattr(train_loader, '_epoch_index', 1)
            if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
                avg_loss = running_loss / (batch_idx + 1)
                current_lr = optimizer.param_groups[0]['lr']
                LOGGER.info(f"🚀 Turbo Epoch {epoch_index}, Batch {batch_idx + 1}/{num_batches}: loss={avg_loss:.4f}, lr={current_lr:.2e}")

    return running_loss / num_batches


# ---------------------------------------------------------------------------
# Turbo Handlers
# ---------------------------------------------------------------------------

class TransformerTurboPreprocessHandler(TransformerTurboDependencyMixin, BasePreprocessHandler):
    """Enhanced preprocessing handler for Transformer Turbo."""

    DESCRIPTION = "Preprocess captcha images for enhanced transformer turbo OCR training and inference."
    SHORT_DESCRIPTION = "Turbo preprocess captcha images for transformer OCR."
    REQUIRED_DEPENDENCIES = TRANSFORMER_TURBO_DEPENDENCIES
    HANDLER_ID = "transformer_turbo_preprocess"

    # Enhanced default dimensions
    DEFAULT_IMG_HEIGHT = 64
    DEFAULT_IMG_WIDTH = 192

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.img_height = int(cfg.get("img_height", self.DEFAULT_IMG_HEIGHT))
        self.img_width = int(cfg.get("img_width", self.DEFAULT_IMG_WIDTH))

    def process(self, input_data: Any) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=self._dependency_error_message(missing))

        if not PIL_AVAILABLE:
            return HandlerResult(success=False, error="Pillow is required for image preprocessing")

        try:
            if isinstance(input_data, (str, Path)):
                image = Image.open(input_data)
            else:
                image = input_data

            if not hasattr(image, 'size'):
                return HandlerResult(success=False, error="Invalid image input")

            # Enhanced preprocessing
            image = image.convert('L')  # Grayscale
            image = image.resize((self.img_width, self.img_height), Image.Resampling.LANCZOS)

            if NUMPY_AVAILABLE and TORCH_AVAILABLE:
                img_array = np.array(image, dtype=np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)

                return HandlerResult(
                    success=True,
                    data=img_tensor,
                    metadata={
                        "processed_size": f"{self.img_width}x{self.img_height}",
                        "format": "tensor",
                        "version": TRANSFORMER_TURBO_HANDLER_VERSION
                    }
                )
            else:
                return HandlerResult(success=True, data=image)

        except Exception as exc:
            return HandlerResult(success=False, error=f"Preprocessing failed: {exc}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": TRANSFORMER_TURBO_HANDLER_VERSION,
            "description": self.DESCRIPTION,
            "handler_id": self.HANDLER_ID,
            "dependencies": TRANSFORMER_TURBO_DEPENDENCIES,
            "default_size": f"{self.img_width}x{self.img_height}",
        }


class TransformerTurboTrainHandler(TransformerTurboDependencyMixin, BaseTrainHandler):
    """Enhanced training handler with advanced strategies."""

    DESCRIPTION = "Train enhanced transformer turbo model with increased capacity and advanced training strategies."
    SHORT_DESCRIPTION = "Train transformer turbo OCR for 4-char captchas."
    REQUIRED_DEPENDENCIES = TRANSFORMER_TURBO_DEPENDENCIES
    HANDLER_ID = "transformer_turbo_train"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}

        # Enhanced model parameters
        self.d_model = int(cfg.get("d_model", 384))
        self.num_layers = int(cfg.get("num_layers", 4))
        self.nhead = int(cfg.get("nhead", 8))
        self.dim_feedforward = int(cfg.get("dim_feedforward", 1024))
        self.dropout = float(cfg.get("dropout", 0.1))

        # Enhanced training parameters
        self.weight_decay = float(cfg.get("weight_decay", 0.0001))
        self.grad_clip = float(cfg.get("grad_clip", 1.0))
        self.use_cosine_annealing = cfg.get("cosine_annealing", True)
        self.warmup_epochs = int(cfg.get("warmup_epochs", 10))

        # Standard parameters
        self.img_h = int(cfg.get("img_height", TransformerTurboPreprocessHandler.DEFAULT_IMG_HEIGHT))
        self.img_w = int(cfg.get("img_width", TransformerTurboPreprocessHandler.DEFAULT_IMG_WIDTH))
        self.num_workers = int(cfg.get("num_workers", 0))
        self.device_name = cfg.get("device", "auto")
        self.log_interval = max(0, int(cfg.get("log_interval", 50)))

    def train(self, config: TrainingConfig) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=self._dependency_error_message(missing))

        if not TORCH_AVAILABLE:
            return HandlerResult(success=False, error="PyTorch is required for transformer turbo training")

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

        # Enhanced data splitting
        val_split = float(config.validation_split)
        total_samples = len(dataset)
        val_size = max(1, int(total_samples * val_split)) if val_split > 0 else 0
        train_size = total_samples - val_size

        if val_size > 0:
            train_ds, val_ds = random_split(dataset, [train_size, val_size])
        else:
            train_ds, val_ds = dataset, None

        # Enhanced data loaders
        train_loader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
            pin_memory=True if device.type == 'cuda' else False,
        )

        val_loader = None
        if val_ds is not None:
            val_loader = DataLoader(
                val_ds,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_batch,
                pin_memory=True if device.type == 'cuda' else False,
            )

        # Enhanced model with custom parameters
        model = TurboOCRModel(
            num_classes=charset.size,
            d_model=self.d_model,
            num_layers=self.num_layers,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )
        model.to(device)

        # Enhanced optimizer and criterion
        criterion = nn.CTCLoss(blank=charset.blank_idx, reduction="mean", zero_infinity=True)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )

        # Enhanced learning rate scheduling
        scheduler = None
        if self.use_cosine_annealing:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
            LOGGER.info("🚀 Using Cosine Annealing LR Scheduler")

        # Training setup logging
        if self.log_interval:
            setattr(train_loader, "_log_interval", self.log_interval)
            setattr(train_loader, "_total_epochs", config.epochs)

        LOGGER.info(
            "🚀 Transformer Turbo training configured: version=%s, epochs=%d, batches=%d, device=%s",
            TRANSFORMER_TURBO_HANDLER_VERSION,
            config.epochs,
            len(train_loader),
            device,
        )
        LOGGER.info(f"🏗️  Enhanced Architecture: d_model={self.d_model}, layers={self.num_layers}, heads={self.nhead}")
        LOGGER.info(f"⚡ Training Enhancements: grad_clip={self.grad_clip}, weight_decay={self.weight_decay}")

        # Enhanced training loop
        history: List[Dict[str, Any]] = []
        best_acc = -1.0
        best_cer = float("inf")

        for epoch in range(1, config.epochs + 1):
            if self.log_interval:
                setattr(train_loader, "_epoch_index", epoch)

            LOGGER.info("🚀 Turbo Epoch %d/%d started", epoch, config.epochs)
            print(f"[TransformerTurboTrainHandler] epoch {epoch}/{config.epochs} started (version {TRANSFORMER_TURBO_HANDLER_VERSION})")

            # Enhanced training with advanced features
            train_loss = train_one_epoch_turbo(
                model, train_loader, optimizer, criterion, charset, device,
                scheduler=None,  # Per-epoch scheduling only
                grad_clip=self.grad_clip
            )

            # Validation
            val_acc = None
            val_cer = None
            if val_loader is not None:
                val_acc, val_cer, _ = evaluate_model(model, val_loader, charset, device)

            # Learning rate scheduling (per epoch)
            if scheduler is not None:
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                LOGGER.info(f"📊 Learning rate updated: {current_lr:.2e}")

            # Logging
            LOGGER.info(
                "🚀 Turbo Epoch %d/%d finished: loss=%.4f%s",
                epoch,
                config.epochs,
                train_loss,
                f", val_acc={val_acc:.4f}, val_cer={val_cer:.4f}" if val_acc is not None else "",
            )

            extra = ""
            if val_acc is not None:
                extra = f", val_acc={val_acc:.4f}, val_cer={val_cer:.4f}"
            print(f"[TransformerTurboTrainHandler] epoch {epoch}/{config.epochs} finished loss={train_loss:.4f}{extra}")

            # History tracking
            history.append({
                "epoch": epoch,
                "loss": train_loss,
                "val_accuracy": val_acc,
                "val_cer": val_cer,
                "learning_rate": optimizer.param_groups[0]['lr'] if scheduler else config.learning_rate,
            })

            # Enhanced model checkpointing
            should_save = False
            if val_acc is not None and val_acc > best_acc:
                best_acc = val_acc
                should_save = True
            if val_cer is not None and val_cer < best_cer:
                best_cer = val_cer
                should_save = True
            if val_acc is None and val_cer is None:  # No validation
                should_save = True

            if should_save:
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "charset": charset,
                    "config": {
                        "d_model": self.d_model,
                        "num_layers": self.num_layers,
                        "nhead": self.nhead,
                        "dim_feedforward": self.dim_feedforward,
                        "dropout": self.dropout,
                        "img_height": self.img_h,
                        "img_width": self.img_w,
                    },
                    "history": history,
                    "version": TRANSFORMER_TURBO_HANDLER_VERSION,
                }
                if not self.save_model(checkpoint, config.output_path):
                    return HandlerResult(success=False, error="Failed to save turbo model checkpoint")

        return HandlerResult(
            success=True,
            data={
                "best_accuracy": best_acc if best_acc > -1 else None,
                "best_cer": best_cer if best_cer < float("inf") else None,
                "final_loss": history[-1]["loss"] if history else None,
                "epochs_trained": len(history),
                "model_path": str(config.output_path),
                "architecture": f"d_model={self.d_model}, layers={self.num_layers}, heads={self.nhead}",
                "version": TRANSFORMER_TURBO_HANDLER_VERSION,
            }
        )

    def save_model(self, model_data: Any, output_path: Path) -> bool:
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model_data, str(output_path))
            return True
        except Exception as e:
            LOGGER.error(f"Failed to save model: {e}")
            return False

    def load_model(self, model_path: Path) -> Any:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for transformer turbo training")
        return torch.load(str(model_path), map_location="cpu", weights_only=False)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": TRANSFORMER_TURBO_HANDLER_VERSION,
            "description": self.DESCRIPTION,
            "handler_id": self.HANDLER_ID,
            "dependencies": TRANSFORMER_TURBO_DEPENDENCIES,
            "architecture": f"Enhanced Transformer: d_model={self.d_model}, layers={self.num_layers}, heads={self.nhead}",
            "features": [
                "Enhanced model capacity (384d, 4 layers, 8 heads)",
                "Advanced training strategies (cosine annealing, gradient clipping)",
                "Pre-layer normalization for better training stability",
                "Enhanced positional encoding with learnable scaling",
                "Residual connections in classifier head",
                "GELU activation for improved performance"
            ]
        }


class TransformerTurboEvaluateHandler(TransformerTurboDependencyMixin, BaseEvaluateHandler):
    """Enhanced evaluation handler for Transformer Turbo models."""

    DESCRIPTION = "Evaluate transformer turbo model performance on test datasets."
    SHORT_DESCRIPTION = "Evaluate transformer turbo OCR performance."
    REQUIRED_DEPENDENCIES = TRANSFORMER_TURBO_DEPENDENCIES
    HANDLER_ID = "transformer_turbo_evaluate"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.device_name = cfg.get("device", "auto")
        self.num_workers = int(cfg.get("num_workers", 0))

    def evaluate(self, model_path: Path, test_data_path: Path) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=self._dependency_error_message(missing))

        if not TORCH_AVAILABLE:
            return HandlerResult(success=False, error="PyTorch is required for transformer turbo evaluation")

        try:
            # Load enhanced model checkpoint
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
            charset = checkpoint["charset"]
            config = checkpoint.get("config", {})

            # Extract model configuration
            d_model = config.get("d_model", 384)
            num_layers = config.get("num_layers", 4)
            nhead = config.get("nhead", 8)
            dim_feedforward = config.get("dim_feedforward", 1024)
            dropout = config.get("dropout", 0.1)
            img_h = config.get("img_height", 64)
            img_w = config.get("img_width", 192)

            # Load test dataset
            try:
                dataset = OCRDataset(
                    test_data_path,
                    img_h,
                    img_w,
                    requirements_override=self._requirements_override(),
                )
            except Exception as exc:
                return HandlerResult(success=False, error=f"Failed to load test dataset: {exc}")

            # Create data loader
            loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_batch,
            )

            # Initialize and load model
            device = resolve_device(self.device_name)
            model = TurboOCRModel(
                num_classes=charset.size,
                d_model=d_model,
                num_layers=num_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            # Enhanced evaluation
            accuracy, cer, all_predictions = evaluate_model(model, loader, charset, device)

            # Calculate detailed metrics
            total_samples = len(dataset)
            correct_samples = int(accuracy * total_samples)

            metrics = EvaluationResult(
                accuracy=accuracy,
                character_accuracy=1.0 - cer,  # Convert CER to character accuracy
                total_samples=total_samples,
                correct_predictions=correct_samples,
            )

            LOGGER.info(
                "🚀 Transformer Turbo evaluation completed: %.2f%% accuracy, %.2f%% character accuracy (%d correct out of %d)",
                accuracy * 100, (1.0 - cer) * 100, correct_samples, total_samples
            )

            return HandlerResult(
                success=True,
                data=metrics,
                metadata={
                    "version": TRANSFORMER_TURBO_HANDLER_VERSION,
                    "architecture": f"d_model={d_model}, layers={num_layers}, heads={nhead}",
                    "model_path": str(model_path),
                }
            )

        except Exception as exc:
            LOGGER.error("🚀 Transformer Turbo evaluation failed: %s", exc, exc_info=True)
            return HandlerResult(success=False, error=f"Evaluation failed: {exc}")

    def calculate_metrics(self, predictions: List[str], ground_truth: List[str]) -> EvaluationResult:
        """Calculate evaluation metrics for transformer turbo model."""
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
            "version": TRANSFORMER_TURBO_HANDLER_VERSION,
            "description": self.DESCRIPTION,
            "handler_id": self.HANDLER_ID,
            "dependencies": TRANSFORMER_TURBO_DEPENDENCIES,
        }


class TransformerTurboOCRHandler(TransformerTurboDependencyMixin, BaseOCRHandler):
    """Enhanced OCR inference handler for production use."""

    DESCRIPTION = "Transformer Turbo OCR inference with enhanced model capacity."
    SHORT_DESCRIPTION = "Transformer Turbo OCR inference."
    REQUIRED_DEPENDENCIES = TRANSFORMER_TURBO_DEPENDENCIES
    HANDLER_ID = "transformer_turbo_ocr"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.device_name = cfg.get("device", "auto")
        self.charset: Optional[Charset] = None
        self.model: Optional[TurboOCRModel] = None
        self.device: Optional[torch.device] = None
        self.expected_length = _coerce_positive_int(cfg.get("expected_length", cfg.get("captcha_length", 4)), 4)
        self.force_lowercase = _coerce_bool(cfg.get("force_lowercase", True), True)

    def load_model(self, model_path: Path) -> bool:
        missing = _missing_dependencies()
        if missing:
            raise RuntimeError(self._dependency_error_message(missing))

        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for transformer turbo OCR")

        try:
            ckpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
            self.charset = ckpt["charset"]
            config = ckpt.get("config", {})

            # Extract enhanced model configuration
            d_model = config.get("d_model", 384)
            num_layers = config.get("num_layers", 4)
            nhead = config.get("nhead", 8)
            dim_feedforward = config.get("dim_feedforward", 1024)
            dropout = config.get("dropout", 0.1)

            self.model = TurboOCRModel(
                num_classes=self.charset.size,
                d_model=d_model,
                num_layers=num_layers,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.device = resolve_device(self.device_name)
            self.model.to(self.device)
            self.model.eval()

            checkpoint_length = _coerce_positive_int(config.get("target_length", config.get("captcha_length")), self.expected_length)
            if checkpoint_length:
                self.expected_length = checkpoint_length
            if "force_lowercase" in config:
                self.force_lowercase = _coerce_bool(config.get("force_lowercase"), self.force_lowercase)

            LOGGER.info(f"🚀 Transformer Turbo OCR model loaded: d_model={d_model}, layers={num_layers}, heads={nhead}")
            return True

        except Exception as exc:
            raise RuntimeError(f"Failed to load transformer turbo OCR model: {exc}")

    def predict(self, processed_image: Any) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=self._dependency_error_message(missing))

        if self.model is None or self.charset is None or self.device is None:
            return HandlerResult(success=False, error="Model not loaded. Call load_model() first.")

        if not TORCH_AVAILABLE:
            return HandlerResult(success=False, error="PyTorch is required for transformer turbo OCR")

        try:
            if torch.is_tensor(processed_image):
                image_tensor = processed_image.to(self.device)
            else:
                image_tensor = torch.tensor(processed_image, dtype=torch.float32).to(self.device)

            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():
                logits = self.model(image_tensor)

            if logits.size(0) == 0:
                return HandlerResult(success=False, error="No logits produced by the model")

            probs = torch.softmax(logits[0], dim=-1)
            max_probs = torch.max(probs, dim=-1)[0]
            avg_confidence = float(max_probs.mean().item() * 100.0)

            predictions = greedy_decode_batch(logits, self.charset)
            raw_prediction = predictions[0] if predictions else ""
            normalized = raw_prediction or ""
            if self.force_lowercase:
                normalized = normalized.lower()

            fallback_applied = False
            char_confidences: List[float] = []

            if len(normalized) != self.expected_length:
                fallback_applied = True
                normalized, char_confidences = _decode_fixed_length(
                    logits[0].detach().cpu(),
                    self.charset,
                    self.expected_length,
                    force_lowercase=self.force_lowercase,
                )
            else:
                char_confidences = _extract_character_confidences(
                    probs,
                    self.charset,
                    self.expected_length,
                )

            if len(normalized) != self.expected_length:
                fallback_applied = True
                normalized, char_confidences = _decode_fixed_length(
                    logits[0].detach().cpu(),
                    self.charset,
                    self.expected_length,
                    force_lowercase=self.force_lowercase,
                )

            if not normalized:
                fallback_applied = True
                fallback_char = _default_charset_char(self.charset)
                normalized = (fallback_char.lower() if self.force_lowercase else fallback_char) * self.expected_length
                char_confidences = [0.0] * self.expected_length

            metadata = {
                "confidence": avg_confidence,
                "model_version": TRANSFORMER_TURBO_HANDLER_VERSION,
                "architecture": "Enhanced Transformer Turbo",
                "sequence_length": logits.size(1),
                "expected_length": self.expected_length,
                "force_lowercase": self.force_lowercase,
                "fallback_applied": fallback_applied,
                "raw_prediction": raw_prediction.lower() if self.force_lowercase else raw_prediction,
                "character_confidences": [float(c) for c in char_confidences],
                "character_count": len(normalized),
            }

            return HandlerResult(
                success=True,
                data=normalized,
                metadata=metadata,
            )

        except Exception as exc:
            return HandlerResult(success=False, error=f"Prediction failed: {exc}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": TRANSFORMER_TURBO_HANDLER_VERSION,
            "description": self.DESCRIPTION,
            "handler_id": self.HANDLER_ID,
            "dependencies": TRANSFORMER_TURBO_DEPENDENCIES,
            "model_loaded": self.model is not None,
            "charset_size": self.charset.size if self.charset else None,
            "architecture": "Enhanced Transformer Turbo (384d, 4L, 8H)",
        }

