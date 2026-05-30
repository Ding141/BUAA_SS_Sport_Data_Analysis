from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
WISDM = ROOT / "sign-deep-learning" / "reports" / "wisdm_deep"
UCI = ROOT / "uci_har_small_dl_challenge" / "reports"
UCI_EXAMPLES = ROOT / "uci_har_small_dl_challenge" / "examples"

COLORS = {
    "blue": "#3B6FB6",
    "orange": "#D97941",
    "green": "#4B8B5B",
    "red": "#B94A48",
    "gray": "#6F7782",
    "light_gray": "#E8EBEF",
    "dark": "#24292F",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.edgecolor": "#C9CED6",
            "axes.labelcolor": COLORS["dark"],
            "xtick.color": COLORS["dark"],
            "ytick.color": COLORS["dark"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=240)
    plt.close(fig)


def load_wisdm_labels() -> list[str]:
    label_map = json.loads((WISDM / "wisdm_label_map.json").read_text(encoding="utf-8"))
    return list(label_map.values())


def fig_wisdm_model_comparison() -> None:
    df = pd.read_csv(WISDM / "unified_model_selection.csv")
    df = df.sort_values("accuracy", ascending=True)
    y = np.arange(len(df))
    color_by_family = {
        "ensemble": COLORS["orange"],
        "engineered features": COLORS["green"],
        "raw sequence": COLORS["blue"],
    }
    colors = [color_by_family.get(family, COLORS["gray"]) for family in df["family"]]

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    ax.barh(y, df["accuracy"] * 100, color=colors, height=0.68, label="Accuracy")
    ax.scatter(df["macro_f1"] * 100, y, color=COLORS["dark"], s=55, zorder=3, label="Macro F1")

    for yi, acc, f1 in zip(y, df["accuracy"] * 100, df["macro_f1"] * 100):
        ax.text(acc + 0.7, yi, f"{acc:.1f}", va="center", fontsize=10)
        ax.text(f1 + 0.9, yi - 0.20, f"F1 {f1:.1f}", va="center", fontsize=8.8, color=COLORS["dark"])

    ax.set_yticks(y)
    ax.set_yticklabels(df["model"].str.replace("_", " ", regex=False), fontsize=10)
    ax.set_xlabel("Test score (%)")
    ax.set_title("WISDM 12-class model comparison", loc="left", weight="bold")
    ax.set_xlim(0, max(65, float(df[["accuracy", "macro_f1"]].max().max() * 100 + 7)))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    save(fig, "fig01_wisdm_model_comparison.png")


def fig_wisdm_alpha_scan() -> None:
    summary = json.loads((WISDM / "wisdm_feature_ensemble_summary.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(summary["alpha_scan"])
    best = df.loc[df["macro_f1"].idxmax()]

    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    ax.plot(df["alpha_mlp"], df["accuracy"] * 100, color=COLORS["blue"], linewidth=2.3, label="Accuracy")
    ax.plot(df["alpha_mlp"], df["macro_f1"] * 100, color=COLORS["orange"], linewidth=2.3, label="Macro F1")
    ax.plot(df["alpha_mlp"], df["balanced_accuracy"] * 100, color=COLORS["green"], linewidth=2.0, label="Balanced accuracy")
    ax.axvline(best["alpha_mlp"], color=COLORS["red"], linestyle="--", linewidth=1.8)
    ax.scatter([best["alpha_mlp"]], [best["macro_f1"] * 100], color=COLORS["red"], s=80, zorder=4)
    ax.text(
        best["alpha_mlp"] + 0.025,
        best["macro_f1"] * 100 + 0.4,
        f"best alpha={best['alpha_mlp']:.2f}",
        fontsize=10,
        color=COLORS["red"],
    )
    ax.set_xlabel("FeatureMLP weight in soft voting")
    ax.set_ylabel("Test score (%)")
    ax.set_title("Soft-voting weight sensitivity on WISDM", loc="left", weight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(47, 61)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    save(fig, "fig02_wisdm_alpha_scan.png")


def fig_wisdm_confusion_matrix() -> None:
    labels = load_wisdm_labels()
    cm = np.loadtxt(WISDM / "wisdm_feature_ensemble_confusion_matrix.csv", delimiter=",")
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(8.8, 7.4))
    sns.heatmap(
        cm_norm,
        ax=ax,
        cmap="Blues",
        vmin=0,
        vmax=1,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Recall-normalized proportion"},
    )
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    ax.set_title("WISDM ensemble normalized confusion matrix", loc="left", weight="bold")
    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    save(fig, "fig03_wisdm_confusion_matrix.png")


def fig_wisdm_per_class_metrics() -> None:
    df = pd.read_csv(WISDM / "ensemble_per_class_metrics.csv").sort_values("f1")
    colors = [
        COLORS["red"] if value < 0.35 else COLORS["orange"] if value < 0.60 else COLORS["green"]
        for value in df["f1"]
    ]

    fig, ax = plt.subplots(figsize=(9.2, 6.8))
    ax.barh(df["activity"], df["f1"] * 100, color=colors)
    for y, value in enumerate(df["f1"] * 100):
        ax.text(value + 1.0, y, f"{value:.1f}", va="center", fontsize=10)
    ax.set_xlabel("F1 score (%)")
    ax.set_ylabel("")
    ax.set_title("WISDM per-class F1 score", loc="left", weight="bold")
    ax.set_xlim(0, 100)
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, "fig04_wisdm_per_class_f1.png")


def fig_wisdm_precision_recall() -> None:
    df = pd.read_csv(WISDM / "ensemble_per_class_metrics.csv")
    sizes = 50 + (df["support"] / df["support"].max()) * 260

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.scatter(
        df["recall"] * 100,
        df["precision"] * 100,
        s=sizes,
        color=COLORS["blue"],
        alpha=0.72,
        edgecolor="white",
        linewidth=1.1,
    )
    for row in df.itertuples():
        ax.text(row.recall * 100 + 1.0, row.precision * 100 + 0.6, row.activity, fontsize=8.5)
    ax.set_xlabel("Recall (%)")
    ax.set_ylabel("Precision (%)")
    ax.set_title("WISDM precision-recall by activity", loc="left", weight="bold")
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig05_wisdm_precision_recall_scatter.png")


def fig_wisdm_feature_importance() -> None:
    top = pd.read_csv(WISDM / "feature_mlp_explainability.csv").head(15).iloc[::-1]
    group = pd.read_csv(WISDM / "feature_mlp_group_importance.csv")
    fam = group.groupby("family", as_index=False)["normalized_importance"].sum().sort_values("normalized_importance")

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.2), gridspec_kw={"width_ratios": [1.15, 0.85]})
    axes[0].barh(top["feature"], top["normalized_importance"] * 100, color=COLORS["blue"])
    axes[0].set_xlabel("Normalized importance (%)")
    axes[0].set_title("Top input features", loc="left", weight="bold")
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    axes[0].tick_params(axis="y", labelsize=8.8)

    axes[1].barh(fam["family"].str.replace("_", " ", regex=False), fam["normalized_importance"] * 100, color=COLORS["green"])
    axes[1].set_xlabel("Aggregated importance (%)")
    axes[1].set_title("Feature-family contribution", loc="left", weight="bold")
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    axes[1].tick_params(axis="y", labelsize=9.2)

    fig.suptitle("WISDM FeatureMLP interpretability evidence", x=0.03, ha="left", weight="bold", fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "fig06_wisdm_feature_importance.png")


def fig_wisdm_sensor_contribution() -> None:
    group = pd.read_csv(WISDM / "feature_mlp_group_importance.csv")
    sensor = group[group["sensor"] != "unknown"].groupby("sensor", as_index=False)["normalized_importance"].sum()
    axis = group[group["axis"] != "unknown"].groupby(["sensor", "axis"], as_index=False)["normalized_importance"].sum()
    axis["label"] = axis["sensor"] + "_" + axis["axis"]
    axis = axis.sort_values("normalized_importance")

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), gridspec_kw={"width_ratios": [0.75, 1.25]})
    axes[0].pie(
        sensor["normalized_importance"],
        labels=sensor["sensor"],
        autopct="%1.1f%%",
        colors=[COLORS["orange"], COLORS["blue"]],
        startangle=90,
        textprops={"fontsize": 10},
    )
    axes[0].set_title("Sensor contribution", weight="bold")

    axes[1].barh(axis["label"], axis["normalized_importance"] * 100, color=COLORS["gray"])
    axes[1].set_xlabel("Normalized importance (%)")
    axes[1].set_title("Axis-level contribution", loc="left", weight="bold")
    axes[1].spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout()
    save(fig, "fig07_wisdm_sensor_axis_contribution.png")


def fig_uci_training_curve() -> None:
    hist = pd.read_json(UCI / "training_history.json")
    fig, ax1 = plt.subplots(figsize=(8.8, 5.4))
    ax1.plot(hist["epoch"], hist["train_accuracy"] * 100, marker="o", linewidth=2.2, color=COLORS["blue"], label="Train accuracy")
    ax1.plot(hist["epoch"], hist["val_accuracy"] * 100, marker="o", linewidth=2.2, color=COLORS["orange"], label="Validation accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_ylim(78, 100)
    ax1.set_title("UCI HAR training and validation accuracy", loc="left", weight="bold")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(frameon=False, loc="lower right")
    save(fig, "fig08_uci_training_curve.png")


def fig_uci_confusion_and_f1() -> None:
    metrics = json.loads((UCI / "evaluation_metrics.json").read_text(encoding="utf-8"))
    names = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    full_names = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    cm = np.asarray(metrics["confusion_matrix"], dtype=float)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    f1 = [metrics["classification_report"][name]["f1-score"] * 100 for name in full_names]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), gridspec_kw={"width_ratios": [1, 0.9]})
    sns.heatmap(
        cm_norm,
        ax=axes[0],
        cmap="Greens",
        vmin=0,
        vmax=1,
        square=True,
        annot=True,
        fmt=".2f",
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Recall-normalized proportion"},
    )
    axes[0].set_title("Normalized confusion matrix", loc="left", weight="bold")
    axes[0].set_xlabel("Predicted activity")
    axes[0].set_ylabel("True activity")
    axes[0].tick_params(axis="x", rotation=45, labelsize=8.5)
    axes[0].tick_params(axis="y", rotation=0, labelsize=8.5)

    colors = [COLORS["green"] if value >= 90 else COLORS["orange"] for value in f1]
    axes[1].barh(names, f1, color=colors)
    axes[1].set_xlim(70, 102)
    axes[1].set_xlabel("F1 score (%)")
    axes[1].set_title("Per-class F1 score", loc="left", weight="bold")
    for y, value in enumerate(f1):
        axes[1].text(value + 0.5, y, f"{value:.1f}", va="center", fontsize=9.5)
    axes[1].spines[["top", "right", "left"]].set_visible(False)

    fig.suptitle(
        f"UCI HAR test performance: {metrics['accuracy']*100:.2f}% accuracy, {metrics['macro_f1']*100:.2f}% macro F1",
        x=0.02,
        ha="left",
        weight="bold",
        fontsize=16,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "fig09_uci_confusion_f1.png")


def fig_uci_signal_examples() -> None:
    rows = []
    for path in sorted(UCI_EXAMPLES.glob("*.npy")):
        if not path.stem[0].isdigit():
            continue
        arr = np.load(path)
        meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        label = meta.get("label_name", path.stem).replace("WALKING_", "").replace("_", " ")
        acc_mag = np.linalg.norm(arr[[0, 1, 2], :], axis=0)
        gyro_mag = np.linalg.norm(arr[[3, 4, 5], :], axis=0)
        rows.append((label.title(), acc_mag, gyro_mag))
    rows = rows[:6]

    fig, axes = plt.subplots(3, 2, figsize=(11.4, 7.2), sharex=True)
    axes = axes.ravel()
    t = np.arange(128)
    for ax, (label, acc, gyro) in zip(axes, rows):
        ax.plot(t, acc, color=COLORS["blue"], linewidth=1.8, label="acc magnitude")
        ax.plot(t, gyro, color=COLORS["orange"], linewidth=1.4, alpha=0.85, label="gyro magnitude")
        ax.set_title(label, loc="left", fontsize=10.5, weight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.supxlabel("Sample index")
    fig.supylabel("Magnitude")
    fig.suptitle("UCI HAR example raw windows", x=0.03, ha="left", weight="bold", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "fig10_uci_signal_examples.png")


def fig_dataset_result_summary() -> None:
    wisdm = json.loads((WISDM / "wisdm_feature_ensemble_summary.json").read_text(encoding="utf-8"))
    uci = json.loads((UCI / "evaluation_metrics.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(
        [
            {"dataset": "WISDM 12-class", "metric": "Accuracy", "score": wisdm["accuracy"] * 100},
            {"dataset": "WISDM 12-class", "metric": "Macro F1", "score": wisdm["macro_f1"] * 100},
            {"dataset": "UCI HAR 6-class", "metric": "Accuracy", "score": uci["accuracy"] * 100},
            {"dataset": "UCI HAR 6-class", "metric": "Macro F1", "score": uci["macro_f1"] * 100},
        ]
    )

    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    sns.barplot(data=df, x="dataset", y="score", hue="metric", palette=[COLORS["blue"], COLORS["orange"]], ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=3, fontsize=10)
    ax.set_ylabel("Test score (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 105)
    ax.set_title("Final result summary", loc="left", weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig11_final_result_summary.png")


def main() -> None:
    setup()
    fig_wisdm_model_comparison()
    fig_wisdm_alpha_scan()
    fig_wisdm_confusion_matrix()
    fig_wisdm_per_class_metrics()
    fig_wisdm_precision_recall()
    fig_wisdm_feature_importance()
    fig_wisdm_sensor_contribution()
    fig_uci_training_curve()
    fig_uci_confusion_and_f1()
    fig_uci_signal_examples()
    fig_dataset_result_summary()


if __name__ == "__main__":
    main()
