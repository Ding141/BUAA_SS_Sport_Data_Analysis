from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.wisdm_arff import load_wisdm_arff_cache
from train_wisdm_feature_mlp import FeatureMLP, subject_masks


MODEL_PATH = Path("models/wisdm_deep/wisdm_feature_mlp_best.pt")
CACHE_PATH = Path("models/wisdm_deep/fused_arff_features_phone.npz")
REPORT_DIR = Path("reports/wisdm_deep")
FIG_DIR = REPORT_DIR / "figures"


def configure_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def parse_feature(name: str) -> tuple[str, str, str]:
    sensor, raw = name.split("_", 1)
    axis = "unknown"
    if raw.startswith("X"):
        axis = "x"
    elif raw.startswith("Y"):
        axis = "y"
    elif raw.startswith("Z"):
        axis = "z"

    if "MFCC" in raw:
        family = "mfcc_frequency"
    elif raw.endswith("AVG"):
        family = "mean"
    elif raw.endswith("PEAK"):
        family = "peak"
    elif raw.endswith("ABSOLDEV"):
        family = "absolute_deviation"
    elif raw.endswith("STANDDEV"):
        family = "standard_deviation"
    elif raw.endswith("VAR"):
        family = "variance"
    elif raw[1:].isdigit():
        family = "raw_sample_bins"
    else:
        family = "other"
    return sensor, axis, family


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    configure_chinese_font()

    bundle = load_wisdm_arff_cache(CACHE_PATH)
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    X = ((bundle.X - checkpoint["mean"]) / checkpoint["std"]).astype(np.float32)
    _, _, test_mask = subject_masks(bundle.subjects)
    X_test = torch.tensor(X[test_mask], dtype=torch.float32, requires_grad=True)

    model = FeatureMLP(X.shape[1], len(bundle.label_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    logits = model(X_test)
    pred = torch.argmax(logits, dim=1)
    selected = logits[torch.arange(logits.shape[0]), pred].sum()
    model.zero_grad(set_to_none=True)
    selected.backward()

    importance = X_test.grad.detach().abs().mean(dim=0).numpy()
    importance = importance / max(float(importance.sum()), 1e-12)

    rows = []
    for name, value in zip(bundle.feature_names, importance, strict=True):
        sensor, axis, family = parse_feature(name)
        rows.append(
            {
                "feature": name,
                "sensor": sensor,
                "axis": axis,
                "family": family,
                "normalized_importance": float(value),
            }
        )
    df = pd.DataFrame(rows).sort_values("normalized_importance", ascending=False)
    df.to_csv(REPORT_DIR / "feature_mlp_explainability.csv", index=False)

    group = (
        df.groupby(["sensor", "axis", "family"], as_index=False)["normalized_importance"]
        .sum()
        .sort_values("normalized_importance", ascending=False)
    )
    group.to_csv(REPORT_DIR / "feature_mlp_group_importance.csv", index=False)

    plt.figure(figsize=(9, 7))
    sns.barplot(data=df.head(20), y="feature", x="normalized_importance", hue="sensor", dodge=False)
    plt.xlabel("归一化梯度重要性")
    plt.ylabel("特征")
    plt.title("FeatureMLP 前 20 个重要输入特征")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_mlp_top_feature_importance.png", dpi=180)
    plt.close()

    top_group = group.head(18).copy()
    top_group["group"] = top_group["sensor"] + " / " + top_group["axis"] + " / " + top_group["family"]
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_group, y="group", x="normalized_importance", hue="sensor", dodge=False)
    plt.xlabel("归一化梯度重要性")
    plt.ylabel("传感器 / 轴向 / 特征族")
    plt.title("FeatureMLP 分组特征重要性")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_mlp_group_importance.png", dpi=180)
    plt.close()

    print("Top features:")
    print(df.head(12).to_string(index=False))
    print("\nTop groups:")
    print(group.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
