from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.deep_models import build_deep_model
from src.wisdm_data import load_wisdm_cache
from train_wisdm_deep import make_loader, predict_all, subject_masks


REPORT_DIR = Path("reports/wisdm_deep")
FIG_DIR = REPORT_DIR / "figures"
MODEL_PATH = Path("models/wisdm_deep/wisdm_deep_model.pt")
CACHE_PATH = Path("models/wisdm_deep/fused_windows_phone.npz")


def _load_model_and_data():
    bundle = load_wisdm_cache(CACHE_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    X = ((bundle.X - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    model = build_deep_model(
        checkpoint["model_type"],
        n_channels=X.shape[1],
        n_classes=len(bundle.label_names),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return bundle, checkpoint, X, model


def plot_model_comparison() -> None:
    df = pd.read_csv(REPORT_DIR / "deep_model_comparison.csv")
    df = df.sort_values("val_accuracy", ascending=False)
    plt.figure(figsize=(10, 5.6))
    sns.barplot(data=df, y="architecture", x="val_accuracy", color="#4C78A8", label="Validation")
    sns.scatterplot(data=df, y="architecture", x="test_accuracy", color="#F58518", s=80, label="Test")
    plt.xlabel("Accuracy")
    plt.ylabel("Architecture")
    plt.title("WISDM Deep Model Comparison")
    plt.xlim(0, max(df["val_accuracy"].max(), df["test_accuracy"].max()) + 0.08)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "model_comparison.png", dpi=180)
    plt.close()


def plot_training_curve() -> None:
    with (REPORT_DIR / "wisdm_final_training.json").open("r", encoding="utf-8") as f:
        history = json.load(f)["history"]
    df = pd.DataFrame(history)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df["epoch"], df["train_loss"], marker="o", color="#E45756", label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["train_acc"], marker="s", color="#4C78A8", label="Train accuracy")
    ax2.set_ylabel("Accuracy")
    fig.suptitle("Final Dual-Branch BiGRU Training Curve")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "training_curve.png", dpi=180)
    plt.close(fig)


def plot_confusion_and_f1(bundle, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    labels = list(range(len(bundle.label_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm_norm,
        xticklabels=bundle.label_names,
        yticklabels=bundle.label_names,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Recall-normalized count"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("WISDM Test Confusion Matrix (Normalized)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix_normalized.png", dpi=180)
    plt.close()

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=bundle.label_names,
        output_dict=True,
        zero_division=0,
    )
    rows = [
        {
            "activity": name,
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
        }
        for name in bundle.label_names
    ]
    df = pd.DataFrame(rows).sort_values("f1", ascending=True)
    plt.figure(figsize=(9, 7))
    sns.barplot(data=df, y="activity", x="f1", color="#54A24B")
    plt.xlabel("F1-score")
    plt.ylabel("Activity")
    plt.title("Per-Class F1 on Unseen Test Subjects")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "per_class_f1.png", dpi=180)
    plt.close()

    df.to_csv(REPORT_DIR / "per_class_metrics.csv", index=False, encoding="utf-8")


def ablation_eval(bundle, X: np.ndarray, y: np.ndarray, mask: np.ndarray, model) -> None:
    variants = {
        "accel+gyro": X.copy(),
        "accel_only": X.copy(),
        "gyro_only": X.copy(),
    }
    variants["accel_only"][:, 3:, :] = 0.0
    variants["gyro_only"][:, :3, :] = 0.0

    rows = []
    for name, X_variant in variants.items():
        loader = make_loader(X_variant, y, mask, 128, shuffle=False)
        y_true, y_pred = predict_all(model, loader, torch.device("cpu"))
        rows.append(
            {
                "input_variant": name,
                "accuracy": accuracy_score(y_true, y_pred),
                "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(REPORT_DIR / "sensor_ablation.csv", index=False, encoding="utf-8")
    plt.figure(figsize=(7, 4.8))
    melted = df.melt(id_vars="input_variant", value_vars=["accuracy", "macro_f1"], var_name="metric")
    sns.barplot(data=melted, x="input_variant", y="value", hue="metric")
    plt.xlabel("Input")
    plt.ylabel("Score")
    plt.title("Sensor Ablation on Test Subjects")
    plt.ylim(0, max(0.5, melted["value"].max() + 0.08))
    plt.tight_layout()
    plt.savefig(FIG_DIR / "sensor_ablation.png", dpi=180)
    plt.close()


def saliency_examples(bundle, X: np.ndarray, y: np.ndarray, mask: np.ndarray, model) -> None:
    channel_names = bundle.channels
    selected_labels = ["walking", "stairs", "jogging", "standing"]
    selected_indices = []
    mask_indices = np.flatnonzero(mask)
    for label in selected_labels:
        label_id = bundle.label_names.index(label)
        matches = [idx for idx in mask_indices if y[idx] == label_id]
        if matches:
            selected_indices.append(matches[0])

    rows = []
    for idx in selected_indices:
        x = torch.tensor(X[idx : idx + 1], dtype=torch.float32, requires_grad=True)
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        score = logits[0, pred]
        model.zero_grad(set_to_none=True)
        score.backward()
        saliency = x.grad.detach().abs().numpy()[0]
        saliency = saliency / max(float(saliency.max()), 1e-8)
        channel_importance = saliency.mean(axis=1)
        rows.append(
            {
                "sample_index": int(idx),
                "true_activity": bundle.label_names[int(y[idx])],
                "predicted_activity": bundle.label_names[pred],
                **{channel_names[i]: float(channel_importance[i]) for i in range(len(channel_names))},
            }
        )

        plt.figure(figsize=(11, 4.5))
        sns.heatmap(
            saliency,
            yticklabels=channel_names,
            xticklabels=False,
            cmap="mako",
            cbar_kws={"label": "Normalized |gradient|"},
        )
        plt.xlabel("Time step")
        plt.ylabel("Sensor channel")
        plt.title(f"Gradient Saliency: true={bundle.label_names[int(y[idx])]}, pred={bundle.label_names[pred]}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"saliency_{bundle.label_names[int(y[idx])]}.png", dpi=180)
        plt.close()

    with (REPORT_DIR / "saliency_channel_importance.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_index", "true_activity", "predicted_activity", *channel_names]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    bundle, _, X, model = _load_model_and_data()
    train_mask, val_mask, test_mask = subject_masks(bundle.subjects)
    y = bundle.y
    test_loader = make_loader(X, y, test_mask, 128, shuffle=False)
    y_true, y_pred = predict_all(model, test_loader, torch.device("cpu"))

    plot_model_comparison()
    plot_training_curve()
    plot_confusion_and_f1(bundle, y_true, y_pred)
    ablation_eval(bundle, X, y, test_mask, model)
    saliency_examples(bundle, X, y, test_mask, model)

    summary = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "figures": sorted(path.name for path in FIG_DIR.glob("*.png")),
    }
    (REPORT_DIR / "visualization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
