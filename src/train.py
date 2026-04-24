from __future__ import annotations

from contextlib import nullcontext

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets

from model import DigitCNN
from utils import DATA_DIR, MODEL_PATH, PLOTS_DIR, ensure_runtime_dirs, get_device, get_num_workers, get_transform, load_config, seed_everything


def create_dataloader(train: bool, batch_size: int, num_workers: int, device: torch.device) -> DataLoader:
    dataset = datasets.MNIST(root=DATA_DIR, train=train, download=True, transform=get_transform())
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def autocast_context(use_amp: bool):
    return torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Adam,
    device: torch.device,
    scaler: torch.amp.GradScaler | None,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = scaler is not None
    non_blocking = device.type == "cuda"

    for images, targets in dataloader:
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    use_amp = device.type == "cuda"
    non_blocking = device.type == "cuda"

    for images, targets in dataloader:
        images = images.to(device, non_blocking=non_blocking)
        targets = targets.to(device, non_blocking=non_blocking)

        with autocast_context(use_amp):
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        total += batch_size

    return running_loss / total, 100.0 * correct / total


def save_metric_plot(train_values: list[float], eval_values: list[float], ylabel: str, path) -> None:
    epochs = range(1, len(train_values) + 1)
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.plot(epochs, train_values, marker="o", linewidth=2, label="Train")
    axis.plot(epochs, eval_values, marker="o", linewidth=2, label="Test")
    axis.set_xlabel("Epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(f"{ylabel} by Epoch")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def main() -> None:
    config = load_config()
    device = get_device()
    ensure_runtime_dirs()
    seed_everything(int(config["seed"]))

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    num_workers = get_num_workers(int(config["num_workers"]))
    batch_size = int(config["batch_size"])
    train_loader = create_dataloader(train=True, batch_size=batch_size, num_workers=num_workers, device=device)
    test_loader = create_dataloader(train=False, batch_size=batch_size, num_workers=num_workers, device=device)

    model = DigitCNN(dropout=float(config["dropout"])).to(device)
    sample_batch = next(iter(train_loader))[0][:1].to(device, non_blocking=device.type == "cuda")
    model(sample_batch)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=float(config["learning_rate"]))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda") if device.type == "cuda" else None

    history = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    best_accuracy = 0.0

    for epoch in range(1, int(config["epochs"]) + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        if test_acc >= best_accuracy:
            best_accuracy = test_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "accuracy": test_acc,
                    "config": config,
                },
                MODEL_PATH,
            )

        print(
            f"Epoch {epoch} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% "
            f"| Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.2f}%"
        )

    save_metric_plot(history["train_loss"], history["test_loss"], "Loss", PLOTS_DIR / "loss.png")
    save_metric_plot(history["train_acc"], history["test_acc"], "Accuracy (%)", PLOTS_DIR / "accuracy.png")


if __name__ == "__main__":
    main()
