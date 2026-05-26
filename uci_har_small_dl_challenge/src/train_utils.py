"""Training and evaluation helpers."""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainResult:
    best_state: dict[str, torch.Tensor]
    history: list[dict[str, float]]
    best_val_accuracy: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    preds = []
    labels = []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        if is_train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * xb.size(0)
        preds.append(logits.argmax(dim=1).detach().cpu().numpy())
        labels.append(yb.detach().cpu().numpy())

    pred_arr = np.concatenate(preds)
    label_arr = np.concatenate(labels)
    return total_loss / len(label_arr), accuracy_score(label_arr, pred_arr)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    patience: int,
) -> TrainResult:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_val = -1.0
    wait = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        row = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "lr": float(scheduler.get_last_lr()[0]),
        }
        history.append(row)

        if val_acc > best_val:
            best_val = val_acc
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1

        print(
            f"epoch {epoch:03d}/{epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )
        if wait >= patience:
            print(f"early stopping after {epoch} epochs; best val_acc={best_val:.4f}")
            break

    return TrainResult(best_state=best_state, history=history, best_val_accuracy=float(best_val))


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits_all = []
    labels_all = []
    for xb, yb in loader:
        logits_all.append(model(xb.to(device)).cpu().numpy())
        labels_all.append(yb.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }

