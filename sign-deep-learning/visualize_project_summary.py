from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import seaborn as sns


REPORT_DIR = Path("reports/wisdm_deep")
FIG_DIR = REPORT_DIR / "figures"


def _style() -> None:
    sns.set_theme(style="whitegrid")
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS"]
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.dpi": 150,
            "axes.unicode_minus": False,
        }
    )


def plot_model_selection() -> None:
    raw = pd.read_csv(REPORT_DIR / "deep_model_comparison.csv")
    raw_rows = raw.rename(columns={"architecture": "model"}).assign(
        family="raw sequence",
        accuracy=lambda d: d["test_accuracy"],
        macro_f1=lambda d: d["test_macro_f1"],
    )[["model", "family", "accuracy", "macro_f1"]]
    final_raw_path = REPORT_DIR / "wisdm_final_training.json"
    if final_raw_path.exists():
        final_raw = json.loads(final_raw_path.read_text(encoding="utf-8"))
        raw_rows = pd.concat(
            [
                raw_rows,
                pd.DataFrame(
                    [
                        {
                            "model": f"{final_raw['architecture']}_final",
                            "family": "raw sequence",
                            "accuracy": final_raw["test_accuracy"],
                            "macro_f1": final_raw["test_macro_f1"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )

    extras = []
    feature_report = json.loads((REPORT_DIR / "wisdm_feature_mlp_training.json").read_text(encoding="utf-8"))
    extras.append(
        {
            "model": "feature_mlp",
            "family": "engineered features",
            "accuracy": feature_report["test_accuracy"],
            "macro_f1": feature_report["test_macro_f1"],
        }
    )
    fusion_path = REPORT_DIR / "wisdm_feature_fusion_training.json"
    if fusion_path.exists():
        fusion_report = json.loads(fusion_path.read_text(encoding="utf-8"))
        extras.append(
            {
                "model": "feature_fusion",
                "family": "engineered features",
                "accuracy": fusion_report["test_accuracy"],
                "macro_f1": fusion_report["test_macro_f1"],
            }
        )
    resnet_path = REPORT_DIR / "wisdm_feature_resnet_training.json"
    if resnet_path.exists():
        resnet_report = json.loads(resnet_path.read_text(encoding="utf-8"))
        extras.append(
            {
                "model": "feature_resnet",
                "family": "engineered features",
                "accuracy": resnet_report["test_accuracy"],
                "macro_f1": resnet_report["test_macro_f1"],
            }
        )
    ensemble_path = REPORT_DIR / "wisdm_feature_ensemble_summary.json"
    if ensemble_path.exists():
        ensemble_report = json.loads(ensemble_path.read_text(encoding="utf-8"))
        extras.append(
            {
                "model": "feature_ensemble",
                "family": "ensemble",
                "accuracy": ensemble_report["accuracy"],
                "macro_f1": ensemble_report["macro_f1"],
            }
        )
    extra_rows = pd.DataFrame(extras)
    df = pd.concat([raw_rows, extra_rows], ignore_index=True).sort_values("accuracy", ascending=True)

    plt.figure(figsize=(10, 6))
    palette = {"raw sequence": "#4C78A8", "engineered features": "#54A24B", "ensemble": "#E45756"}
    sns.barplot(data=df, y="model", x="accuracy", hue="family", dodge=False, palette=palette)
    for y, value in enumerate(df["accuracy"]):
        plt.text(value + 0.006, y, f"{value:.3f}", va="center", fontsize=9)
    plt.xlabel("跨受试者测试准确率")
    plt.ylabel("模型")
    plt.title("WISDM 12 类模型选择对比")
    plt.xlim(0, max(0.45, df["accuracy"].max() + 0.06))
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "project_model_selection.png", dpi=180)
    plt.close()

    df.sort_values("accuracy", ascending=False).to_csv(REPORT_DIR / "unified_model_selection.csv", index=False)


def plot_feature_training_curve() -> None:
    payload = json.loads((REPORT_DIR / "wisdm_feature_mlp_training.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(payload["history"])
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(df["epoch"], df["train_loss"], color="#E45756", linewidth=1.8, label="Train loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df["train_acc"], color="#4C78A8", linewidth=1.8, label="Train accuracy")
    ax2.set_ylabel("Accuracy")
    fig.suptitle("FeatureMLP 训练曲线")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_mlp_training_curve.png", dpi=180)
    plt.close(fig)


def plot_feature_confusion_matrix() -> None:
    labels = list(json.loads((REPORT_DIR / "wisdm_label_map.json").read_text(encoding="utf-8")).values())
    cm = np.loadtxt(REPORT_DIR / "wisdm_feature_mlp_confusion_matrix.csv", delimiter=",")
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        cm_norm,
        xticklabels=labels,
        yticklabels=labels,
        cmap="Greens",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": "Recall-normalized count"},
    )
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title("FeatureMLP 跨受试者测试归一化混淆矩阵")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_mlp_confusion_matrix_normalized.png", dpi=180)
    plt.close()


def plot_feature_per_class_f1() -> None:
    lines = (REPORT_DIR / "wisdm_feature_mlp_test_report.txt").read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, float | str]] = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            name, precision, recall, f1, support = parts
            try:
                rows.append(
                    {
                        "activity": name,
                        "precision": float(precision),
                        "recall": float(recall),
                        "f1": float(f1),
                        "support": int(support),
                    }
                )
            except ValueError:
                continue
    df = pd.DataFrame(rows).sort_values("f1", ascending=True)
    df.to_csv(REPORT_DIR / "feature_mlp_per_class_metrics.csv", index=False)

    plt.figure(figsize=(9, 7))
    sns.barplot(data=df, y="activity", x="f1", color="#54A24B")
    plt.xlabel("F1 分数")
    plt.ylabel("动作类别")
    plt.title("FeatureMLP 每类 F1 分数")
    plt.xlim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "feature_mlp_per_class_f1.png", dpi=180)
    plt.close()


def plot_architecture_overview() -> None:
    ensemble = json.loads((REPORT_DIR / "wisdm_feature_ensemble_summary.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(REPORT_DIR / "unified_model_selection.csv")
    raw_rows = raw[raw["family"].eq("raw sequence")]
    raw_best = raw_rows.sort_values("accuracy", ascending=False).iloc[0] if not raw_rows.empty else None
    raw_accuracy = float(raw_best["accuracy"]) if raw_best is not None else 0.0
    raw_model = str(raw_best["model"]).replace("_", " ") if raw_best is not None else "raw sequence"
    final_accuracy = float(ensemble["accuracy"])
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)

    boxes = [
        ("WISDM 原始数据\n手机加速度计 + 陀螺仪", 0.4, 3.8, 2.1, 1.0, "#E8EEF7"),
        ("6 通道窗口\n6 x 200, 步长 100", 3.1, 4.2, 2.2, 1.0, "#DDECCB"),
        (f"{raw_model}\n端到端基线", 6.0, 4.2, 2.3, 1.0, "#BFD7EA"),
        (f"原始序列评估\n准确率 {raw_accuracy:.3f}", 9.2, 4.2, 2.3, 1.0, "#F6D6AD"),
        ("ARFF 时频特征\n182 维工程特征", 3.1, 2.1, 2.2, 1.0, "#DDECCB"),
        ("FeatureFusionNet + MLP\n软投票集成", 6.0, 2.1, 2.3, 1.0, "#C7E9C0"),
        (f"最终报告与图表\n准确率 {final_accuracy:.3f}", 9.2, 2.1, 2.3, 1.0, "#F6D6AD"),
    ]
    for text, x, y, w, h, color in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#3A3A3A", linewidth=1.3)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10, weight="bold")

    arrows = [
        ((2.5, 4.3), (3.1, 4.7)),
        ((5.3, 4.7), (6.0, 4.7)),
        ((8.3, 4.7), (9.2, 4.7)),
        ((2.5, 4.3), (3.1, 2.6)),
        ((5.3, 2.6), (6.0, 2.6)),
        ((8.3, 2.6), (9.2, 2.6)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.8, color="#444"))

    ax.text(
        0.5,
        0.8,
        "选择原则：以跨受试者测试作为最终指标，优先选择能反映跨用户泛化能力、可解释性更强且复杂度适中的模型。\n原18类中soup/chips/drinking/typing/clapping/dribbling六类因验证准确率过低已剔除，精简为12类。",
        fontsize=10,
        color="#333",
        wrap=True,
    )
    ax.set_title("WISDM 12 类动作识别深度学习项目架构", fontsize=15, weight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "unified_project_architecture.png", dpi=180)
    plt.close()


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _style()
    plot_model_selection()
    plot_feature_training_curve()
    plot_feature_confusion_matrix()
    plot_feature_per_class_f1()
    plot_architecture_overview()
    ensemble = json.loads((REPORT_DIR / "wisdm_feature_ensemble_summary.json").read_text(encoding="utf-8"))
    fusion = json.loads((REPORT_DIR / "wisdm_feature_fusion_training.json").read_text(encoding="utf-8"))
    model_selection = pd.read_csv(REPORT_DIR / "unified_model_selection.csv")
    raw_rows = model_selection[model_selection["family"].eq("raw sequence")]
    raw_best = raw_rows.sort_values("accuracy", ascending=False).iloc[0] if not raw_rows.empty else None
    summary = {
        "recommended_model": "feature_ensemble",
        "recommended_accuracy": ensemble["accuracy"],
        "recommended_balanced_accuracy": ensemble.get("balanced_accuracy"),
        "recommended_macro_f1": ensemble["macro_f1"],
        "alpha_mlp": ensemble["alpha_mlp"],
        "alpha_fusion": ensemble["alpha_fusion"],
        "base_feature_model": "feature_fusion",
        "base_feature_accuracy": fusion["test_accuracy"],
        "base_feature_macro_f1": fusion["test_macro_f1"],
        "raw_sequence_baseline": str(raw_best["model"]) if raw_best is not None else None,
        "raw_sequence_baseline_accuracy": float(raw_best["accuracy"]) if raw_best is not None else None,
        "raw_sequence_baseline_macro_f1": float(raw_best["macro_f1"]) if raw_best is not None else None,
        "figures": sorted(path.name for path in FIG_DIR.glob("*.png")),
    }
    (REPORT_DIR / "project_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
