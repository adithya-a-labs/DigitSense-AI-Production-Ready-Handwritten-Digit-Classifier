from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from model import DigitCNN
from utils import MODEL_PATH, get_device


@lru_cache(maxsize=1)
def _load_model_cached(model_path: str) -> tuple[DigitCNN, torch.device]:
    resolved_path = Path(model_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {resolved_path}")

    device = get_device()
    checkpoint = torch.load(resolved_path, map_location=device)
    state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint

    model = DigitCNN()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device


def load_model(model_path: str | Path = MODEL_PATH) -> tuple[DigitCNN, torch.device]:
    return _load_model_cached(str(Path(model_path).resolve()))


@torch.no_grad()
def predict(image_tensor: torch.Tensor, model_path: str | Path = MODEL_PATH) -> tuple[int, torch.Tensor]:
    model, device = load_model(model_path)

    if image_tensor.ndim == 2:
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
    elif image_tensor.ndim == 3:
        image_tensor = image_tensor.unsqueeze(0)
    elif image_tensor.ndim != 4:
        raise ValueError("image_tensor must be a 2D, 3D, or 4D tensor.")

    inputs = image_tensor.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    logits = model(inputs)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu()
    predicted_digit = int(probabilities.argmax().item())
    return predicted_digit, probabilities
