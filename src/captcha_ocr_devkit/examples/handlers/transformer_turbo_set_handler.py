"""Set-based Transformer Turbo OCR handler.

Provides an order-invariant decoder that selects the most probable characters
from the Transformer Turbo logits without relying on CTC-style collapsing.
Useful for captchas where only the multiset of characters matters.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from captcha_ocr_devkit.core.handlers.base import BaseOCRHandler, HandlerResult
from captcha_ocr_devkit.examples.handlers.ocr_common import (
    Charset,
    _missing_dependencies,
    resolve_device,
    torch,
)

try:
    from .transformer_turbo_handler import (
        TRANSFORMER_TURBO_DEPENDENCIES,
        TRANSFORMER_TURBO_HANDLER_VERSION,
        TransformerTurboDependencyMixin,
        TurboOCRModel,
        _coerce_bool,
        _coerce_positive_int,
        _default_charset_char,
    )
except ImportError:  # pragma: no cover - allow running outside package context
    from captcha_ocr_devkit.examples.handlers.transformer_turbo_handler import (
        TRANSFORMER_TURBO_DEPENDENCIES,
        TRANSFORMER_TURBO_HANDLER_VERSION,
        TransformerTurboDependencyMixin,
        TurboOCRModel,
        _coerce_bool,
        _coerce_positive_int,
        _default_charset_char,
    )

LOGGER = logging.getLogger(__name__)
if not LOGGER.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    LOGGER.addHandler(handler)
    LOGGER.propagate = False
if LOGGER.getEffectiveLevel() > logging.INFO:
    LOGGER.setLevel(logging.INFO)


class TransformerTurboSetOCRHandler(TransformerTurboDependencyMixin, BaseOCRHandler):
    """Set-based OCR inference that ignores character ordering and picks top-k letters."""

    DESCRIPTION = "Transformer Turbo OCR inference (order-invariant, top-k characters)."
    SHORT_DESCRIPTION = "Set-based transformer turbo OCR inference."
    REQUIRED_DEPENDENCIES = TRANSFORMER_TURBO_DEPENDENCIES
    HANDLER_ID = "transformer_turbo_set_ocr"

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

        if not torch:  # pragma: no cover - defensive check
            raise RuntimeError("PyTorch is required for transformer turbo set OCR")

        try:
            checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
            self.charset = checkpoint["charset"]
            config = checkpoint.get("config", {})

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
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.device = resolve_device(self.device_name)
            self.model.to(self.device)
            self.model.eval()

            checkpoint_length = _coerce_positive_int(config.get("target_length", config.get("captcha_length")), self.expected_length)
            if checkpoint_length:
                self.expected_length = checkpoint_length
            if "force_lowercase" in config:
                self.force_lowercase = _coerce_bool(config.get("force_lowercase"), self.force_lowercase)

            LOGGER.info(
                "🚀 Transformer Turbo Set OCR model loaded: d_model=%s, layers=%s, heads=%s, expected_length=%s",
                d_model,
                num_layers,
                nhead,
                self.expected_length,
            )
            return True

        except Exception as exc:  # pragma: no cover - defensive branch
            raise RuntimeError(f"Failed to load transformer turbo set OCR model: {exc}")

    def predict(self, processed_image: Any) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=self._dependency_error_message(missing))

        if self.model is None or self.charset is None or self.device is None:
            return HandlerResult(success=False, error="Model not loaded. Call load_model() first.")

        if not torch:  # pragma: no cover - defensive check
            return HandlerResult(success=False, error="PyTorch is required for transformer turbo set OCR")

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
            blank_idx = self.charset.blank_idx

            if probs.size(-1) <= blank_idx:
                return HandlerResult(success=False, error="Invalid charset configuration for set decoding")

            scores = probs.clone()
            if blank_idx < scores.size(-1):
                scores[:, blank_idx] = 0.0

            num_classes = scores.size(-1)
            flat_scores = scores.reshape(-1)
            sorted_indices = torch.argsort(flat_scores, descending=True)

            chars: List[str] = []
            confidences: List[float] = []
            for flat_index in sorted_indices.tolist():
                class_idx = flat_index % num_classes
                if class_idx == blank_idx:
                    continue
                char = self.charset.itos[class_idx]
                if not char:
                    continue
                char_out = char.lower() if self.force_lowercase else char
                chars.append(char_out)
                confidences.append(float(flat_scores[flat_index].item() * 100.0))
                if len(chars) == self.expected_length:
                    break

            if len(chars) < self.expected_length:
                fallback_char = _default_charset_char(self.charset)
                filler = fallback_char.lower() if self.force_lowercase else fallback_char
                while len(chars) < self.expected_length:
                    chars.append(filler)
                    confidences.append(0.0)

            prediction = "".join(chars[: self.expected_length])
            avg_conf = float(sum(confidences[: self.expected_length]) / max(1, self.expected_length))

            metadata = {
                "confidence": avg_conf,
                "model_version": TRANSFORMER_TURBO_HANDLER_VERSION,
                "architecture": "Enhanced Transformer Turbo (set mode)",
                "character_confidences": confidences[: self.expected_length],
                "character_count": len(prediction),
                "aggregator": "topk_time_char",
            }

            return HandlerResult(success=True, data=prediction, metadata=metadata)

        except Exception as exc:
            return HandlerResult(success=False, error=f"Set prediction failed: {exc}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": TRANSFORMER_TURBO_HANDLER_VERSION,
            "description": self.DESCRIPTION,
            "handler_id": self.HANDLER_ID,
            "dependencies": TRANSFORMER_TURBO_DEPENDENCIES,
            "model_loaded": self.model is not None,
            "charset_size": self.charset.size if self.charset else None,
            "architecture": "Enhanced Transformer Turbo (set mode)",
            "expected_length": self.expected_length,
            "force_lowercase": self.force_lowercase,
        }

__all__ = ["TransformerTurboSetOCRHandler"]
