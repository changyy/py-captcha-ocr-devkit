"""Ultra-CNN OCR handlers for 4-character lowercase captchas.

🔥 Ultra-optimized CNN with ResNet+CBAM+FPN architecture targeting 95% accuracy.
This handler represents the state-of-the-art CNN implementation for CAPTCHA OCR,
challenging Transformer's dominance with advanced deep learning techniques.

Key Features:
- ResNet residual architecture (4 layers: 96→192→384→768 channels)
- CBAM attention modules (SE-Block + Spatial Attention)
- Feature Pyramid Network (FPN) for multi-scale feature fusion
- UltraOCRDataset with comprehensive data augmentation
- Mixed precision training and enhanced optimizations
"""

from __future__ import annotations

import logging
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from captcha_ocr_devkit.core.handlers.base import (
    BaseEvaluateHandler,
    BaseOCRHandler,
    BaseTrainHandler,
    EvaluationResult,
    HandlerResult,
    TrainingConfig,
)

from captcha_ocr_devkit.examples.handlers.ocr_common import (
    TorchHandlerDependencyMixin,
    OCRDataset,
    collate_batch,
    format_dependency_error,
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

# Import torchvision transforms for advanced data augmentation
if TORCH_AVAILABLE:
    try:
        import torchvision.transforms as transforms
        TORCHVISION_AVAILABLE = True
    except ImportError:
        TORCHVISION_AVAILABLE = False
else:
    TORCHVISION_AVAILABLE = False

from captcha_ocr_devkit.examples.handlers.transformer_handler import TransformerPreprocessHandler

ULTRA_CNN_HANDLER_VERSION = "1.20250924.0000"  # Ultra-CNN for 95% accuracy target
ULTRA_CNN_DEPENDENCIES = ["torch", "torchvision", "pillow", "numpy"]
ULTRA_CNN_REQUIREMENTS_FILE = "ultra_cnn_handler-requirements.txt"
DEFAULT_NUM_CHARACTERS = 4
DEFAULT_ALPHABET = list(string.ascii_lowercase)
DEFAULT_IMG_HEIGHT = TransformerPreprocessHandler.DEFAULT_IMG_HEIGHT
DEFAULT_IMG_WIDTH = TransformerPreprocessHandler.DEFAULT_IMG_WIDTH


class UltraCNNDependencyMixin(TorchHandlerDependencyMixin):
    """Override requirements file defaults for Ultra-CNN handlers."""

    REQUIREMENTS_FILE = ULTRA_CNN_REQUIREMENTS_FILE


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
        raise RuntimeError("PyTorch is required for Ultra-CNN handlers. Please install torch and torchvision.")


def _normalize_alphabet(alphabet: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(alphabet, str):
        candidates = list(alphabet)
    else:
        candidates = [str(ch) for ch in alphabet if str(ch)]
    normalized: List[str] = []
    seen = set()
    for item in candidates:
        symbol = item[0]
        if symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    return normalized or DEFAULT_ALPHABET.copy()


def _is_valid_label(label: str, alphabet_set: set[str], num_characters: int) -> bool:
    return len(label) == num_characters and all(ch in alphabet_set for ch in label)


def _filter_dataset_samples(
    dataset: OCRDataset,
    alphabet_set: set[str],
    num_characters: int,
) -> Tuple[int, int]:
    original = len(dataset.samples)
    dataset.samples = [
        (path, label)
        for path, label in dataset.samples
        if _is_valid_label(label, alphabet_set, num_characters)
    ]
    return original, len(dataset.samples)


class UltraOCRDataset(OCRDataset):
    """Enhanced OCRDataset with advanced data augmentation for Ultra-CNN."""

    def __init__(self, *args, use_augmentation: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_augmentation = use_augmentation

        if use_augmentation and TORCHVISION_AVAILABLE:
            # Advanced augmentation pipeline for 95% accuracy target
            self.augment_transform = transforms.Compose([
                # Geometric transformations
                transforms.RandomRotation(degrees=(-5, 5), fill=0),  # Slight rotation
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.05, 0.05),  # Small translation
                    scale=(0.95, 1.05),      # Slight scaling
                    shear=(-2, 2),           # Minor shear
                    fill=0
                ),

                # Photometric transformations
                transforms.ColorJitter(
                    brightness=0.2,          # Brightness variation
                    contrast=0.2,            # Contrast variation
                    saturation=0.1,          # Saturation variation
                    hue=0.05                 # Hue variation
                ),

                # Noise and blur
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),  # Gaussian blur

                # Random horizontal flip (light chance)
                transforms.RandomHorizontalFlip(p=0.1),

                # Random erasing (cutout) for robustness
                transforms.RandomErasing(
                    p=0.1,                   # 10% probability
                    scale=(0.02, 0.08),      # Small patches
                    ratio=(0.3, 3.0),        # Aspect ratio range
                    value=0                  # Fill with black
                ),
            ])
            LOGGER.info("🎨 UltraOCRDataset: Advanced data augmentation enabled")
        else:
            self.augment_transform = None
            LOGGER.info("📋 UltraOCRDataset: Standard dataset mode")

    def __getitem__(self, idx):
        result = super().__getitem__(idx)

        # Handle both 2-value and 3-value returns from parent class
        if len(result) == 3:
            image, label, path = result
        else:
            image, label = result
            path = None

        # Apply augmentation if enabled
        if self.use_augmentation and self.augment_transform is not None:
            # Convert to PIL for augmentation if needed
            if not hasattr(image, 'mode'):  # If it's a tensor
                image = transforms.ToPILImage()(image)

            # Apply augmentation
            image = self.augment_transform(image)

            # Convert back to tensor if needed
            if not torch.is_tensor(image):
                image = transforms.ToTensor()(image)

        # Return in the same format as the parent class
        if path is not None:
            return image, label, path
        else:
            return image, label


class UltraCNNPreprocessHandler(UltraCNNDependencyMixin, TransformerPreprocessHandler):
    """Resize and normalize captcha images for the Ultra-CNN pipeline."""

    DESCRIPTION = "Resize captcha images and normalize them for Ultra-CNN OCR training and inference."
    SHORT_DESCRIPTION = "Preprocess captcha images for Ultra-CNN OCR."
    REQUIRED_DEPENDENCIES = ULTRA_CNN_DEPENDENCIES
    HANDLER_ID = "ultra_cnn_preprocess"

    def get_info(self) -> Dict[str, Any]:
        info = super().get_info()
        info["version"] = ULTRA_CNN_HANDLER_VERSION
        return info


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module combining channel and spatial attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Enhanced channel attention
        self.ca = SEBlock(channels, reduction)

        # Enhanced spatial attention
        self.sa = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        x = self.ca(x)

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        spatial_weight = self.sa(spatial_input)
        x = x * spatial_weight

        return x


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion."""

    def __init__(self, in_channels_list: List[int], out_channels: int = 256):
        super().__init__()
        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            inner_block = nn.Conv2d(in_channels, out_channels, 1)
            layer_block = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.inner_blocks.append(inner_block)
            self.layer_blocks.append(layer_block)

    def forward(self, feature_maps: List[torch.Tensor]) -> List[torch.Tensor]:
        results = []
        last_inner = self.inner_blocks[-1](feature_maps[-1])
        results.append(self.layer_blocks[-1](last_inner))

        for idx in range(len(feature_maps) - 2, -1, -1):
            inner_lateral = self.inner_blocks[idx](feature_maps[idx])
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = torch.nn.functional.interpolate(last_inner, size=feat_shape, mode='nearest')
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.layer_blocks[idx](last_inner))

        return results


class ResidualBlock(nn.Module):
    """Enhanced Residual block with CBAM attention and dropout."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        # CBAM attention module
        self.cbam = CBAM(out_channels)

        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.skip(x)

        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        # Apply CBAM attention
        out = self.cbam(out)

        out += residual
        return torch.relu(out)


class UltraCNNClassifier(nn.Module):
    """Ultra-optimized CNN with ResNet+CBAM+FPN for 95% accuracy target."""

    def __init__(self, num_classes: int, num_characters: int, in_channels: int = 1, dropout: float = 0.2):
        super().__init__()
        self.num_characters = num_characters
        self.num_classes = num_classes

        # Enhanced stem with dual 3x3 convolutions
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1)
        )

        # Deeper residual layers with progressive widening and CBAM attention
        self.layer1 = self._make_layer(64, 96, 3, stride=1, dropout=dropout)
        self.layer2 = self._make_layer(96, 192, 4, stride=2, dropout=dropout)
        self.layer3 = self._make_layer(192, 384, 6, stride=2, dropout=dropout)
        self.layer4 = self._make_layer(384, 768, 3, stride=2, dropout=dropout)

        # Feature Pyramid Network for multi-scale fusion
        self.fpn = FeaturePyramidNetwork([96, 192, 384, 768], 256)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Enhanced classifier head with more capacity
        feature_dim = 256 * 4  # 4 FPN levels
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Multi-head character classifiers with cross-attention
        self.character_heads = nn.ModuleList([
            nn.Linear(512, num_classes) for _ in range(num_characters)
        ])

        # Cross-character attention for context modeling
        self.cross_attention = nn.MultiheadAttention(512, 8, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(512)

        LOGGER.info(f"🔥 UltraCNNClassifier: Initialized with {self._count_parameters():,} parameters")

    def _make_layer(self, in_channels: int, out_channels: int, blocks: int, stride: int = 1, dropout: float = 0.1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, dropout))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, dropout=dropout))
        return nn.Sequential(*layers)

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # Stem feature extraction
        x = self.stem(x)

        # Multi-scale feature extraction
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        # Feature pyramid fusion
        fpn_features = self.fpn([c1, c2, c3, c4])

        # Global feature aggregation
        features = []
        for fpn_feat in fpn_features:
            pooled = self.global_pool(fpn_feat)
            features.append(pooled.flatten(1))

        # Concatenate multi-scale features
        fused_features = torch.cat(features, dim=1)

        # Enhanced classification head
        char_features = self.classifier(fused_features)

        # Prepare for cross-attention (batch_size, num_characters, feature_dim)
        batch_size = char_features.size(0)
        char_context = char_features.unsqueeze(1).repeat(1, self.num_characters, 1)

        # Cross-character attention for context modeling
        attended_features, _ = self.cross_attention(char_context, char_context, char_context)
        attended_features = self.norm(attended_features + char_context)

        # Multi-head character classification
        logits = []
        for i, head in enumerate(self.character_heads):
            char_logit = head(attended_features[:, i, :])
            logits.append(char_logit)

        return torch.stack(logits, dim=1)  # Shape: (batch_size, num_characters, num_classes)


def _labels_to_tensor(labels: List[str], alphabet_map: Dict[str, int], num_characters: int) -> torch.Tensor:
    """Convert string labels to tensor format for training."""
    tensor_list = []
    for label in labels:
        label_indices = [alphabet_map.get(ch, 0) for ch in label[:num_characters]]
        while len(label_indices) < num_characters:
            label_indices.append(0)
        tensor_list.append(label_indices)
    return torch.tensor(tensor_list, dtype=torch.long)


def _logits_to_predictions(logits: torch.Tensor, alphabet: List[str], num_characters: int) -> List[str]:
    """Convert model logits to string predictions."""
    batch_size = logits.size(0)
    predicted_indices = torch.argmax(logits, dim=2)  # Shape: (batch_size, num_characters)

    predictions = []
    for i in range(batch_size):
        pred_chars = [alphabet[predicted_indices[i, j].item()] for j in range(num_characters)]
        predictions.append(''.join(pred_chars))

    return predictions


class UltraCNNTrainHandler(UltraCNNDependencyMixin, BaseTrainHandler):
    """Train the Ultra-CNN OCR model targeting 95% accuracy."""

    DESCRIPTION = "Train Ultra-CNN with ResNet+CBAM+FPN architecture for 95% accuracy target."
    SHORT_DESCRIPTION = "Train Ultra-CNN OCR for 4-char captchas."
    REQUIRED_DEPENDENCIES = ULTRA_CNN_DEPENDENCIES
    HANDLER_ID = "ultra_cnn_train"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.img_h = int(cfg.get("img_height", DEFAULT_IMG_HEIGHT))
        self.img_w = int(cfg.get("img_width", DEFAULT_IMG_WIDTH))
        self.num_workers = int(cfg.get("num_workers", 0))
        self.device_name = cfg.get("device", "auto")
        self.log_interval = max(0, int(cfg.get("log_interval", 50)))
        self.weight_decay = float(cfg.get("weight_decay", 1e-4))
        self.alphabet = _normalize_alphabet(cfg.get("alphabet", DEFAULT_ALPHABET))
        self.num_characters = int(cfg.get("num_characters", DEFAULT_NUM_CHARACTERS))
        self.alphabet_map = {ch: idx for idx, ch in enumerate(self.alphabet)}

        # Ultra-CNN specific settings
        self.dropout = float(cfg.get("dropout", 0.2))
        self.label_smoothing = float(cfg.get("label_smoothing", 0.1))
        self.cosine_annealing = cfg.get("cosine_annealing", True)
        self.early_stopping_patience = int(cfg.get("early_stopping_patience", 15))
        self.use_augmentation = cfg.get("use_augmentation", True)

    def train(self, config: TrainingConfig) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=format_dependency_error(missing, self._install_hint()))

        input_dir = Path(config.input_path)
        if not input_dir.exists():
            return HandlerResult(success=False, error=f"Input directory not found: {input_dir}")

        set_seed(config.seed)
        device = resolve_device(config.device if config.device != "auto" else self.device_name)

        try:
            # Use UltraOCRDataset with advanced augmentation
            dataset = UltraOCRDataset(
                input_dir,
                self.img_h,
                self.img_w,
                use_augmentation=self.use_augmentation,
                requirements_override=self._requirements_override(),
            )
            LOGGER.info("🎯 Using UltraOCRDataset with advanced data augmentation for 95% accuracy target")
        except Exception as exc:
            return HandlerResult(success=False, error=str(exc))

        alphabet_set = set(self.alphabet)
        original_count, filtered_count = _filter_dataset_samples(dataset, alphabet_set, self.num_characters)
        LOGGER.info(f"Dataset samples: {original_count} → {filtered_count} (filtered)")

        if filtered_count == 0:
            return HandlerResult(success=False, error="No valid samples found after filtering")

        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_batch,
        )

        # Create Ultra-CNN model
        model = UltraCNNClassifier(
            num_classes=len(self.alphabet),
            num_characters=self.num_characters,
            dropout=self.dropout
        )
        model.to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)

        # Enhanced AdamW optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        LOGGER.info("🚀 Ultra mode: Enhanced AdamW optimizer with optimized hyperparameters")

        # Mixed precision training (if available)
        use_amp = hasattr(torch.cuda, 'amp') and device.type == 'cuda'
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            LOGGER.info("⚡ Ultra mode: Mixed precision training enabled")
        else:
            scaler = None

        # Learning rate scheduler
        if self.cosine_annealing:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        else:
            scheduler = None

        LOGGER.info(
            f"Ultra-CNN training configured: version={ULTRA_CNN_HANDLER_VERSION} epochs={config.epochs}, "
            f"batches={len(train_loader)}, device={device}, log_interval={self.log_interval}"
        )

        best_accuracy = 0.0
        best_char_accuracy = 0.0
        patience_counter = 0

        for epoch in range(1, config.epochs + 1):
            if self.log_interval:
                setattr(train_loader, "_epoch_index", epoch)
            LOGGER.info("Epoch %d/%d started", epoch, config.epochs)
            print(
                f"[UltraCNNTrainHandler] epoch {epoch}/{config.epochs} started (version {ULTRA_CNN_HANDLER_VERSION})",
                flush=True,
            )

            model.train()
            running_loss = 0.0
            total_items = 0
            for batch_index, (inputs, labels, _) in enumerate(train_loader, start=1):
                inputs = inputs.to(device)
                targets = _labels_to_tensor(labels, self.alphabet_map, self.num_characters).to(device)

                optimizer.zero_grad()

                # Mixed precision training
                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        logits = model(inputs)
                        loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = model(inputs)
                    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
                    loss.backward()

                    # Enhanced gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                    optimizer.step()

                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                total_items += batch_size

                if self.log_interval and batch_index % self.log_interval == 0:
                    avg_loss = running_loss / total_items if total_items > 0 else 0.0
                    LOGGER.info(f"Epoch {epoch}, Batch {batch_index}/{len(train_loader)}, Loss: {avg_loss:.4f}")

            # Validation
            model.eval()
            val_correct = 0
            val_total = 0
            val_char_correct = 0
            val_char_total = 0

            with torch.no_grad():
                for inputs, labels, _ in val_loader:
                    inputs = inputs.to(device)
                    logits = model(inputs)
                    predictions = _logits_to_predictions(logits, self.alphabet, self.num_characters)

                    for pred, label in zip(predictions, labels):
                        val_total += 1
                        if pred == label:
                            val_correct += 1

                        for p_char, l_char in zip(pred, label):
                            val_char_total += 1
                            if p_char == l_char:
                                val_char_correct += 1

            val_accuracy = val_correct / val_total if val_total > 0 else 0.0
            val_char_accuracy = val_char_correct / val_char_total if val_char_total > 0 else 0.0

            # Learning rate scheduler step
            if scheduler:
                scheduler.step()

            avg_loss = running_loss / total_items if total_items > 0 else 0.0
            LOGGER.info(f"Epoch {epoch}/{config.epochs} finished: loss={avg_loss:.4f}, val_acc={val_accuracy:.4f}, val_char_acc={val_char_accuracy:.4f}")
            print(f"[UltraCNNTrainHandler] epoch {epoch}/{config.epochs} finished loss={avg_loss:.4f}, val_acc={val_accuracy:.4f}, val_char_acc={val_char_accuracy:.4f}")

            # Early stopping and model saving
            if val_accuracy > best_accuracy or val_char_accuracy > best_char_accuracy:
                if val_accuracy > best_accuracy:
                    best_accuracy = max(best_accuracy, val_accuracy)
                    patience_counter = 0
                if val_char_accuracy is not None:
                    best_char_accuracy = max(best_char_accuracy, val_char_accuracy)
                checkpoint = {
                    "model": model.state_dict(),
                    "alphabet": self.alphabet,
                    "num_characters": self.num_characters,
                    "img_h": self.img_h,
                    "img_w": self.img_w,
                    "dropout": self.dropout,
                    "handler_version": ULTRA_CNN_HANDLER_VERSION,
                }
                if not self.save_model(checkpoint, config.output_path):
                    return HandlerResult(success=False, error="Failed to save Ultra-CNN checkpoint")
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    LOGGER.info(f"Early stopping triggered after {epoch} epochs")
                    break

        return HandlerResult(
            success=True,
            data={
                "final_accuracy": best_accuracy,
                "final_char_accuracy": best_char_accuracy,
                "epochs_trained": epoch,
                "model_path": str(config.output_path),
            }
        )

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
            "version": ULTRA_CNN_HANDLER_VERSION,
            "description": self.DESCRIPTION,
            "handler_id": self.HANDLER_ID,
            "dependencies": ULTRA_CNN_DEPENDENCIES,
            "architecture": "ResNet+CBAM+FPN",
            "target_accuracy": "95%",
            "features": [
                "4-layer residual backbone (96→192→384→768)",
                "CBAM attention modules",
                "Feature Pyramid Network",
                "Advanced data augmentation",
                "Mixed precision training",
                "Cross-character attention"
            ]
        }


class UltraCNNEvaluateHandler(UltraCNNDependencyMixin, BaseEvaluateHandler):
    """Evaluate Ultra-CNN model performance."""

    DESCRIPTION = "Evaluate Ultra-CNN model accuracy on test datasets."
    SHORT_DESCRIPTION = "Evaluate Ultra-CNN OCR performance."
    REQUIRED_DEPENDENCIES = ULTRA_CNN_DEPENDENCIES
    HANDLER_ID = "ultra_cnn_evaluate"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.device_name = cfg.get("device", "auto")
        self.num_workers = int(cfg.get("num_workers", 0))

    def evaluate(self, model_path: Path, test_data_path: Path) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=format_dependency_error(missing, self._install_hint()))

        try:
            checkpoint = torch.load(str(model_path), map_location="cpu")
            alphabet = _normalize_alphabet(checkpoint.get("alphabet", DEFAULT_ALPHABET))
            num_characters = int(checkpoint.get("num_characters", DEFAULT_NUM_CHARACTERS))
            img_h = int(checkpoint.get("img_h", DEFAULT_IMG_HEIGHT))
            img_w = int(checkpoint.get("img_w", DEFAULT_IMG_WIDTH))
            dropout = checkpoint.get("dropout", 0.2)

            try:
                dataset = OCRDataset(
                    test_data_path,
                    img_h,
                    img_w,
                    requirements_override=self._requirements_override(),
                )
            except Exception as exc:
                return HandlerResult(success=False, error=f"Failed to load test dataset: {exc}")

            alphabet_set = set(alphabet)
            original_count, filtered_count = _filter_dataset_samples(dataset, alphabet_set, num_characters)

            if filtered_count == 0:
                return HandlerResult(success=False, error="No valid test samples found")

            loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_batch,
            )

            device = resolve_device(self.device_name)
            model = UltraCNNClassifier(
                num_classes=len(alphabet),
                num_characters=num_characters,
                dropout=dropout
            )
            model.load_state_dict(checkpoint["model"])
            model.to(device)
            model.eval()

            records: List[Tuple[Path, str, str]] = []
            with torch.no_grad():
                for inputs, labels, paths in loader:
                    inputs = inputs.to(device)
                    logits = model(inputs)
                    preds = _logits_to_predictions(logits, alphabet, num_characters)
                    records.extend(zip(paths, labels, preds))

            predictions = [pred for _, _, pred in records]
            ground_truth = [label for _, label, _ in records]
            metrics = self.calculate_metrics(predictions, ground_truth)
            metrics.total_samples = len(dataset)

            LOGGER.info(f"🎯 Ultra-CNN Evaluation Complete: {metrics.accuracy:.2%} accuracy, {metrics.character_accuracy:.2%} character accuracy")

            return HandlerResult(success=True, data=metrics)

        except Exception as exc:
            return HandlerResult(success=False, error=f"Evaluation failed: {exc}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": ULTRA_CNN_HANDLER_VERSION,
            "description": self.DESCRIPTION,
            "handler_id": self.HANDLER_ID,
            "dependencies": ULTRA_CNN_DEPENDENCIES,
        }


class UltraCNNOCRHandler(UltraCNNDependencyMixin, BaseOCRHandler):
    """Ultra-CNN OCR inference handler for production use."""

    DESCRIPTION = "Ultra-CNN OCR inference with 95% accuracy target."
    SHORT_DESCRIPTION = "Ultra-CNN OCR inference."
    REQUIRED_DEPENDENCIES = ULTRA_CNN_DEPENDENCIES
    HANDLER_ID = "ultra_cnn_ocr"

    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        cfg = config or {}
        self.device_name = cfg.get("device", "auto")
        self.alphabet: List[str] = []
        self.num_characters = DEFAULT_NUM_CHARACTERS
        self.img_h = DEFAULT_IMG_HEIGHT
        self.img_w = DEFAULT_IMG_WIDTH
        self.model: Optional[UltraCNNClassifier] = None
        self.device: Optional[torch.device] = None

    def load_model(self, model_path: Path) -> bool:
        missing = _missing_dependencies()
        if missing:
            raise RuntimeError(format_dependency_error(missing, self._install_hint()))

        try:
            checkpoint = torch.load(str(model_path), map_location="cpu")
            stored_alphabet = checkpoint.get("alphabet")
            if stored_alphabet:
                self.alphabet = _normalize_alphabet(stored_alphabet)
            self.num_characters = int(checkpoint.get("num_characters", self.num_characters))
            self.img_h = int(checkpoint.get("img_h", self.img_h))
            self.img_w = int(checkpoint.get("img_w", self.img_w))
            dropout = checkpoint.get("dropout", 0.2)

            self.model = UltraCNNClassifier(
                num_classes=len(self.alphabet),
                num_characters=self.num_characters,
                dropout=dropout
            )
            self.model.load_state_dict(checkpoint["model"])
            self.device = resolve_device(self.device_name)
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as exc:
            raise RuntimeError(f"Failed to load Ultra-CNN OCR checkpoint: {exc}")

    def predict(self, processed_image: Any) -> HandlerResult:
        missing = _missing_dependencies()
        if missing:
            return HandlerResult(success=False, error=format_dependency_error(missing, self._install_hint()))
        if self.model is None or self.device is None:
            return HandlerResult(success=False, error="Model not loaded. Call load_model() first.")

        try:
            if torch.is_tensor(processed_image):
                image_tensor = processed_image.to(self.device)
            else:
                image_tensor = torch.tensor(processed_image, dtype=torch.float32).to(self.device)

            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)

            with torch.no_grad():
                logits = self.model(image_tensor)
                predictions = _logits_to_predictions(logits, self.alphabet, self.num_characters)

            if len(predictions) > 0:
                prediction = predictions[0]
                # Calculate confidence (using max probability across all characters)
                probs = torch.softmax(logits[0], dim=-1)
                char_confidences = torch.max(probs, dim=-1)[0].cpu().numpy()
                avg_confidence = float(char_confidences.mean()) * 100

                return HandlerResult(
                    success=True,
                    data=prediction,
                    metadata={
                        "confidence": avg_confidence,
                        "character_confidences": [float(conf * 100) for conf in char_confidences],
                        "model_version": ULTRA_CNN_HANDLER_VERSION,
                        "architecture": "ResNet+CBAM+FPN"
                    }
                )
            else:
                return HandlerResult(success=False, error="No prediction generated")

        except Exception as exc:
            return HandlerResult(success=False, error=f"Prediction failed: {exc}")

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": ULTRA_CNN_HANDLER_VERSION,
            "description": self.DESCRIPTION,
            "handler_id": self.HANDLER_ID,
            "dependencies": ULTRA_CNN_DEPENDENCIES,
            "model_loaded": self.model is not None,
            "alphabet": self.alphabet,
            "num_characters": self.num_characters,
            "architecture": "ResNet+CBAM+FPN Ultra-CNN",
            "target_accuracy": "95%"
        }