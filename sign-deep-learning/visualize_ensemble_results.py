from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report


REPORT_DIR = Path("reports/wisdm_deep")
FIG_DIR = REPORT_DIR / "figures"


def configure_font() -> None:
    sns.set_theme(style="whitegrid")
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    configure_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    label_map = pd.read_json(REPORT_DIR / "wisdm_label_map.json", typ="series")
    labels = list(label_map.values)
    cm = np.loadtxt(REPORT_DIR / "wisdm_feature_ensemble_confusion_matrix.csv", delimiter=",")
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm_norm,
        xticklabels=labels,
        yticklabels=labels,
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "按真实类别归一化的召回比例"},
    )
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("最终集成模型归一化混淆矩阵")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ensemble_confusion_matrix_normalized.png", dpi=180)
    plt.close()

    # Reconstruct y_true/y_pred from the confusion matrix for a report dict.
    y_true: list[int] = []
    y_pred: list[int] = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = int(cm[i, j])
            y_true.extend([i] * count)
            y_pred.extend([j] * count)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    rows = [
        {
            "activity": name,
            "precision": report[name]["precision"],
            "recall": report[name]["recall"],
            "f1": report[name]["f1-score"],
            "support": report[name]["support"],
        }
        for name in labels
    ]
    df = pd.DataFrame(rows).sort_values("f1", ascending=True)
    df.to_csv(REPORT_DIR / "ensemble_per_class_metrics.csv", index=False)

    plt.figure(figsize=(9, 7))
    sns.barplot(data=df, y="activity", x="f1", color="#4C78A8")
    plt.xlabel("F1 分数")
    plt.ylabel("动作类别")
    plt.title("最终集成模型每类 F1")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ensemble_per_class_f1.png", dpi=180)
    plt.close()

    print("Saved ensemble visualizations.")


if __name__ == "__main__":
    main()
