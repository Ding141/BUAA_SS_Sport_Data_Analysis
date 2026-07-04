from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.deep_models import build_deep_model
from src.wisdm_data import load_wisdm_cache
from train_wisdm_deep import make_loader, predict_all, subject_masks


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a WISDM deep model checkpoint.")
    parser.add_argument("--cache", default="models/wisdm_deep/fused_windows_phone.npz")
    parser.add_argument("--model", default="models/wisdm_deep/wisdm_deep_model.pt")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--report-dir", default="reports/wisdm_deep")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    bundle = load_wisdm_cache(Path(args.cache))
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    X = ((bundle.X - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    y = bundle.y
    train_mask, val_mask, test_mask = subject_masks(bundle.subjects)
    mask = {"train": train_mask, "val": val_mask, "test": test_mask}[args.split]

    model = build_deep_model(
        checkpoint.get("model_type", "cnn_bigru"),
        n_channels=X.shape[1],
        n_classes=len(bundle.label_names),
    )
    model.load_state_dict(checkpoint["model_state"])
    loader = make_loader(X, y, mask, args.batch_size, shuffle=False)
    y_true, y_pred = predict_all(model, loader, torch.device("cpu"))

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(bundle.label_names))),
        target_names=bundle.label_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(bundle.label_names))))

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"wisdm_deep_{args.split}_classification_report.txt"
    report_path.write_text(
        f"Model: {checkpoint.get('model_type', 'unknown')}\n"
        f"Split: {args.split}\n"
        f"Accuracy: {acc:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )
    np.savetxt(report_dir / f"wisdm_deep_{args.split}_confusion_matrix.csv", cm, delimiter=",", fmt="%d")

    print(f"Model: {checkpoint.get('model_type', 'unknown')}")
    print(f"Split: {args.split}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)
    print(f"Saved report: {report_path}")


if __name__ == "__main__":
    main()
