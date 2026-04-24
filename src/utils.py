from __future__ import annotations

import os
import random
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

try:
    import yaml
except ImportError:
    yaml = None

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / ".data" / "mnist"
OUTPUT_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
MODEL_PATH = OUTPUT_DIR / "model.pth"
CONFUSION_MATRIX_PATH = OUTPUT_DIR / "confusion_matrix.png"

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)
IMAGE_SIZE = 28

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "epochs": 5,
    "batch_size": 64,
    "learning_rate": 1e-3,
    "num_workers": 2,
    "dropout": 0.25,
}

_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(MNIST_MEAN, MNIST_STD),
    ]
)


@lru_cache(maxsize=1)
def get_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sample = torch.zeros((1, 1, 4, 4), device="cuda")
            kernel = torch.zeros((1, 1, 1, 1), device="cuda")
            torch.nn.functional.conv2d(sample, kernel)
        return torch.device("cuda")
    except Exception:
        return torch.device("cpu")


def get_transform() -> transforms.Compose:
    return _TRANSFORM


def get_num_workers(requested: int = 2) -> int:
    if os.name == "nt":
        return 0
    cpu_count = os.cpu_count() or 0
    return min(requested, cpu_count) if cpu_count else 0


def ensure_runtime_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or ROOT_DIR / "config.yaml"
    if not path.exists():
        return DEFAULT_CONFIG.copy()

    if yaml is not None:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    else:
        loaded = _parse_simple_yaml(path)

    if not isinstance(loaded, dict):
        return DEFAULT_CONFIG.copy()

    config = DEFAULT_CONFIG.copy()
    config.update(loaded)
    return config


def prepare_image(image: Image.Image) -> torch.Tensor:
    processed = ImageOps.exif_transpose(image).convert("L")
    processed = ImageOps.autocontrast(processed)
    if np.asarray(processed, dtype=np.float32).mean() > 127:
        processed = ImageOps.invert(processed)
    processed = ImageOps.pad(processed, (IMAGE_SIZE, IMAGE_SIZE), color=0, method=_RESAMPLE)
    return _TRANSFORM(processed)


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", maxsplit=1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", maxsplit=1)
        parsed[key.strip()] = _coerce_scalar(value.strip())
    return parsed


def _coerce_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("'\"")
