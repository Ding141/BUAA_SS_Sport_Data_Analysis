from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.wisdm_arff import load_wisdm_arff_cache
from train_wisdm_feature_fusion import FeatureFusionNet
from train_wisdm_feature_mlp import FeatureMLP


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict a cached WISDM ARFF sample with the feature ensemble.")
    parser.add_argument("--cache", default="models/wisdm_deep/fused_arff_features_phone.npz")
    parser.add_argument("--mlp-model", default="models/wisdm_deep/wisdm_feature_mlp_best.pt")
    parser.add_argument("--fusion-model", default="models/wisdm_deep/wisdm_feature_fusion.pt")
    parser.add_argument(
        "--alpha-mlp",
        type=float,
        default=0.22,
        help="Soft-voting weight for FeatureMLP. FusionNet receives 1 - alpha.",
    )
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    bundle = load_wisdm_arff_cache(Path(args.cache))

    mlp_checkpoint = torch.load(args.mlp_model, map_location="cpu", weights_only=False)
    X_mlp = ((bundle.X - mlp_checkpoint["mean"]) / mlp_checkpoint["std"]).astype(np.float32)
    mlp = FeatureMLP(X_mlp.shape[1], len(bundle.label_names))
    mlp.load_state_dict(mlp_checkpoint["model_state"])
    mlp.eval()

    fusion_checkpoint = torch.load(args.fusion_model, map_location="cpu", weights_only=False)
    X_fusion = ((bundle.X - fusion_checkpoint["mean"]) / fusion_checkpoint["std"]).astype(np.float32)
    accel_idx = fusion_checkpoint["accel_indices"]
    gyro_idx = fusion_checkpoint["gyro_indices"]
    fusion = FeatureFusionNet(len(accel_idx), len(gyro_idx), len(bundle.label_names))
    fusion.load_state_dict(fusion_checkpoint["model_state"])
    fusion.eval()

    with torch.no_grad():
        p_mlp = torch.softmax(mlp(torch.from_numpy(X_mlp[args.sample : args.sample + 1])), dim=1).numpy()[0]
        p_fusion = torch.softmax(
            fusion(
                torch.from_numpy(X_fusion[args.sample : args.sample + 1, accel_idx]),
                torch.from_numpy(X_fusion[args.sample : args.sample + 1, gyro_idx]),
            ),
            dim=1,
        ).numpy()[0]

    probabilities = args.alpha_mlp * p_mlp + (1.0 - args.alpha_mlp) * p_fusion
    order = np.argsort(probabilities)[::-1]
    best = int(order[0])

    print(f"Input: {args.cache} sample #{args.sample}")
    print(f"Alpha MLP: {args.alpha_mlp:.2f}")
    print(f"Alpha FusionNet: {1.0 - args.alpha_mlp:.2f}")
    print(f"True activity: {bundle.label_names[int(bundle.y[args.sample])]}")
    print(f"Predicted label code: {bundle.label_codes[best]}")
    print(f"Predicted activity: {bundle.label_names[best]}")
    print(f"Confidence: {probabilities[best]:.4f}")
    print("Top classes:")
    for idx in order[: args.top_k]:
        print(f"  {bundle.label_codes[int(idx)]}: {bundle.label_names[int(idx)]} {probabilities[int(idx)]:.4f}")


if __name__ == "__main__":
    main()
