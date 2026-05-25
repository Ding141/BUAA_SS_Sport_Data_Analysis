from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.wisdm_arff import load_wisdm_arff_cache
from train_wisdm_feature_fusion import FeatureFusionNet
from train_wisdm_feature_mlp import FeatureMLP, subject_masks


def load_mlp_probabilities(bundle, test_mask: np.ndarray, model_path: Path) -> np.ndarray:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    X = ((bundle.X - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    model = FeatureMLP(X.shape[1], len(bundle.label_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(X[test_mask])), dim=1).numpy()


def load_fusion_probabilities(bundle, test_mask: np.ndarray, model_path: Path) -> np.ndarray:
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    X = ((bundle.X - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    accel_idx = checkpoint["accel_indices"]
    gyro_idx = checkpoint["gyro_indices"]
    model = FeatureFusionNet(len(accel_idx), len(gyro_idx), len(bundle.label_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        return torch.softmax(
            model(
                torch.from_numpy(X[test_mask][:, accel_idx]),
                torch.from_numpy(X[test_mask][:, gyro_idx]),
            ),
            dim=1,
        ).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the ARFF feature soft-voting ensemble.")
    parser.add_argument("--cache", default="models/wisdm_deep/fused_arff_features_phone.npz")
    parser.add_argument("--mlp-model", default="models/wisdm_deep/wisdm_feature_mlp_best.pt")
    parser.add_argument("--fusion-model", default="models/wisdm_deep/wisdm_feature_fusion.pt")
    parser.add_argument("--alpha-mlp", type=float, default=0.70)
    parser.add_argument("--report-dir", default="reports/wisdm_deep")
    args = parser.parse_args()

    bundle = load_wisdm_arff_cache(Path(args.cache))
    _, _, test_mask = subject_masks(bundle.subjects)
    y_true = bundle.y[test_mask]

    p_mlp = load_mlp_probabilities(bundle, test_mask, Path(args.mlp_model))
    p_fusion = load_fusion_probabilities(bundle, test_mask, Path(args.fusion_model))
    probabilities = args.alpha_mlp * p_mlp + (1.0 - args.alpha_mlp) * p_fusion
    y_pred = probabilities.argmax(axis=1)

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
    (report_dir / "wisdm_feature_ensemble_test_report.txt").write_text(
        "Model: feature_mlp + feature_fusion soft-voting ensemble\n"
        f"Alpha MLP: {args.alpha_mlp:.2f}\n"
        f"Accuracy: {acc:.4f}\n"
        f"Macro F1: {macro_f1:.4f}\n\n{report}",
        encoding="utf-8",
    )
    np.savetxt(report_dir / "wisdm_feature_ensemble_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
    (report_dir / "wisdm_feature_ensemble_summary.json").write_text(
        json.dumps(
            {
                "model": "feature_mlp_feature_fusion_ensemble",
                "alpha_mlp": args.alpha_mlp,
                "accuracy": acc,
                "macro_f1": macro_f1,
                "mlp_model": args.mlp_model,
                "fusion_model": args.fusion_model,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(report)


if __name__ == "__main__":
    main()
