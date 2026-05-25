from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from train_wisdm_feature_mlp import FeatureMLP
from src.wisdm_arff import load_wisdm_arff_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct test entry for the optimized WISDM ARFF feature MLP.")
    parser.add_argument("--model", default="models/wisdm_deep/wisdm_feature_mlp_best.pt")
    parser.add_argument("--cache", default="models/wisdm_deep/fused_arff_features_phone.npz")
    parser.add_argument("--sample", type=int, default=0)
    args = parser.parse_args()

    bundle = load_wisdm_arff_cache(Path(args.cache))
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    x = ((bundle.X[args.sample : args.sample + 1] - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    model = FeatureMLP(x.shape[1], len(checkpoint["label_names"]))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    with torch.no_grad():
        prob = torch.softmax(model(torch.from_numpy(x)), dim=1).numpy()[0]
    order = np.argsort(prob)[::-1]
    best = int(order[0])
    print(f"Input: {args.cache} sample #{args.sample}")
    print(f"True activity: {bundle.label_names[int(bundle.y[args.sample])]}")
    print(f"Predicted activity: {checkpoint['label_names'][best]}")
    print(f"Confidence: {prob[best]:.4f}")
    print("Top classes:")
    for idx in order[:3]:
        print(f"  {checkpoint['label_names'][int(idx)]}: {prob[int(idx)]:.4f}")


if __name__ == "__main__":
    main()
