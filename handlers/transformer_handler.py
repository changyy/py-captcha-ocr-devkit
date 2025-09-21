"""Transformer handler re-export.

This module stays at repository root so legacy discovery paths keep working.
Actual implementations live in `captcha_ocr_devkit.examples.handlers.transformer_handler`.
"""

from captcha_ocr_devkit.examples.handlers.transformer_handler import (  # noqa: F401
    TRANSFORMER_DEPENDENCIES,
    TRANSFORMER_REQUIREMENTS_FILE,
    TRANSFORMER_HANDLER_VERSION,
    Charset,
    TransformerEvaluateHandler,
    TransformerOCRHandler,
    TransformerPreprocessHandler,
    TransformerTrainHandler,
)

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
