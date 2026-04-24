from __future__ import annotations

from contextlib import nullcontext

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from predict import load_model
from utils import CONFUSION_MATRIX_PATH, DATA_DIR, ensure_runtime_dirs, get_num_workers, get_transform, load_config


def autocast_context(use_amp: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()


def create_dataloader(batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    dataset = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=get_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


@torch.no_grad()
def main() -> None:
    config = load_config()
    ensure_runtime_dirs()
    model, device = load_model()
    dataloader = create_dataloader(
        batch_size=int(config["batch_size"]),
        num_workers=get_num_workers(int(config["num_workers"])),
        device=device,
    )

    predictions: list[int] = []
    targets: list[int] = []
    non_blocking = device.type == "cuda"

    model.eval()
    for images, labels in dataloader:
        images = images.to(device, non_blocking=non_blocking)
        with autocast_context(device.type == "cuda"):
            logits = model(images)

        predictions.extend(logits.argmax(dim=1).cpu().tolist())
        targets.extend(labels.tolist())

    matrix = confusion_matrix(targets, predictions)
    prediction_tensor = torch.tensor(predictions)
    target_tensor = torch.tensor(targets)
    accuracy = 100.0 * prediction_tensor.eq(target_tensor).float().mean().item()

    figure, axis = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axis)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"MNIST Confusion Matrix ({accuracy:.2f}% accuracy)")
    figure.tight_layout()
    figure.savefig(CONFUSION_MATRIX_PATH, dpi=200)
    plt.close(figure)

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
