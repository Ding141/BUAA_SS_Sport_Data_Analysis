"""
Classifier Showcase: Decision Tree + 4-Classifier Comparison
=============================================================
UCI HAR Dataset, 72-D features (6 channels x 12 features), 6-class activity recognition

Output (figures/classifier_showcase/):
  01_Decision_Tree_Structure.png         — Decision tree top 3 levels, class names, node coloring
  02_Decision_Tree_Feature_Importance.png — Top-20 Gini importance, freq/time domain color-coded
  03_Decision_Tree_Confusion_Matrix.png   — Normalized confusion matrix + per-class F1
  04_Four_Classifiers_Confusion_Matrix.png — DT/KNN/SVM/RF side-by-side
  05_Classifier_Metrics_Comparison.png    — Accuracy + Macro-F1 + Weighted-F1
  06_Classifier_Training_Time.png         — Training time comparison
  07_PCA_Decision_Boundary.png            — 2D PCA projection + decision regions
"""

import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import entropy as stats_entropy
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, classification_report,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ── Config ───────────────────────────────────────────────────────
FS = 50
N_SAMPLES = 128
DATA_DIR = "UCI HAR Dataset"
SAVE_DIR = os.path.join("..", "..", "figures", "classifier_showcase")
os.makedirs(SAVE_DIR, exist_ok=True)

ACTIVITY_LABEL = {
    1: "Walking", 2: "Upstairs", 3: "Downstairs",
    4: "Sitting", 5: "Standing", 6: "Laying",
}
ACTIVITY_SHORT = {
    1: "Walk", 2: "Up", 3: "Down", 4: "Sit", 5: "Stand", 6: "Lay",
}

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "font.size": 13, "axes.titlesize": 16, "axes.labelsize": 14,
    "legend.fontsize": 11,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "sans-serif"],
})
sns.set_style("whitegrid")

# ═══════════════════════════════════════════════════════════════
#  Data Loading + Feature Extraction (72-D)
# ═══════════════════════════════════════════════════════════════

def load_uci_har():
    """Load UCI HAR, return train/test raw signals + labels."""
    X, y = {}, {}
    for subset in ["train", "test"]:
        path = os.path.join(DATA_DIR, subset, "Inertial Signals")
        channels = []
        for axis in ["x", "y", "z"]:
            for signal in ["body_acc", "body_gyro"]:
                data = np.loadtxt(
                    os.path.join(path, f"{signal}_{axis}_{subset}.txt"),
                    dtype=np.float32)
                channels.append(data)
        X[subset] = np.stack(channels, axis=-1)  # (N, 128, 6)
        y[subset] = np.loadtxt(
            os.path.join(DATA_DIR, subset, f"y_{subset}.txt")).astype(int)
    return X, y

def fft_spectrum(signal):
    n = len(signal)
    vals = fft(signal)
    mag = np.abs(vals) / n
    mag = mag[: n // 2 + 1]
    mag[1:-1] *= 2
    freqs = fftfreq(n, 1 / FS)[: n // 2 + 1]
    return freqs, mag

def extract_features(raw_data):
    """72-D: 6 channels x (8 freq-domain + 4 time-domain)."""
    n_win, _, n_ch = raw_data.shape
    rows = []
    for i in range(n_win):
        row = []
        for ch in range(n_ch):
            sig = raw_data[i, :, ch]
            freqs, mag = fft_spectrum(sig)
            total = np.sum(mag)
            eps = 1e-12
            # Frequency domain 8
            row.append(freqs[np.argmax(mag)])
            if total > eps:
                row.append(np.sum(freqs * mag) / total)
                cum = np.cumsum(mag)
                row.append(freqs[np.searchsorted(cum, total / 2)])
                row.append(np.sum(mag ** 2))
                row.append(stats_entropy(mag / total + eps))
                nyq = freqs[-1]
                row.append(np.sum(mag[(freqs >= 0) & (freqs < nyq * 0.2)] ** 2))
                row.append(np.sum(mag[(freqs >= nyq * 0.2) & (freqs < nyq * 0.6)] ** 2))
                row.append(np.sum(mag[(freqs >= nyq * 0.6)] ** 2))
            else:
                row.extend([0.0] * 7)
            # Time domain 4
            row.append(np.mean(sig))
            row.append(np.var(sig))
            row.append(np.ptp(sig))
            row.append(np.sum(np.diff(np.signbit(sig))) / len(sig))
        rows.append(row)
    return np.array(rows, dtype=np.float64)

# ═══════════════════════════════════════════════════════════════
#  Fig 1: Decision Tree Structure
# ═══════════════════════════════════════════════════════════════

def plot_decision_tree_structure(clf, feat_names):
    """Decision tree top 3 levels with class names and filled nodes."""
    fig, ax = plt.subplots(figsize=(48, 22))
    plot_tree(
        clf, max_depth=3, feature_names=feat_names,
        class_names=[ACTIVITY_LABEL[i] for i in range(1, 7)],
        filled=True, rounded=True, fontsize=22, ax=ax,
        impurity=False, proportion=True,
        node_ids=False,
        label="all",
    )
    ax.set_title(
        "Fig 1: Decision Tree Structure (Top 3 Levels) — UCI HAR 6-Class Activity Recognition",
        fontsize=26, fontweight="bold", pad=8)
    fig.tight_layout(pad=1)
    path = os.path.join(SAVE_DIR, "01_Decision_Tree_Structure.png")
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  Fig 2: Feature Importance
# ═══════════════════════════════════════════════════════════════

def plot_feature_importance_dt(clf, feat_names):
    """Top-20 Gini importance, freq/time domain color-coded with values."""
    importances = clf.feature_importances_
    top_n = 20
    idx = np.argsort(importances)[-top_n:]
    names = [feat_names[i] for i in idx]
    values = importances[idx]

    colors = [
        "#E53935" if any(kw in n for kw in
            ["Freq", "Spectral", "Band", "Entropy", "Energy"])
        else "#1E88E5" for n in names
    ]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(top_n), values, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Gini Importance", fontsize=14)
    ax.invert_yaxis()

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=10)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#E53935", label="Frequency Domain"),
        Patch(color="#1E88E5", label="Time Domain"),
    ], fontsize=12, loc="lower right")
    ax.set_title(
        "Fig 2: Decision Tree Feature Importance Top-20",
        fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "02_Decision_Tree_Feature_Importance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  Fig 3: Decision Tree Confusion Matrix
# ═══════════════════════════════════════════════════════════════

def plot_dt_confusion_matrix(y_true, y_pred):
    """Normalized confusion matrix + per-class F1 annotation."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = [ACTIVITY_LABEL[i] for i in range(1, 7)]
    f1_per_class = f1_score(y_true, y_pred, average=None)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=labels, yticklabels=labels,
                linewidths=0.5, vmin=0, vmax=1, ax=ax,
                annot_kws={"fontsize": 14})
    ax.set_xlabel("Predicted", fontsize=14)
    ax.set_ylabel("True", fontsize=14)
    ax.set_title(
        f"Fig 3: Decision Tree Confusion Matrix (Normalized) — Acc={accuracy_score(y_true, y_pred)*100:.1f}%",
        fontsize=16, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "03_Decision_Tree_Confusion_Matrix.png")
    fig.savefig(path)
    plt.close(fig)

    # Terminal output
    print(f"  ✓ {path}")
    print(f"    Per-class F1: ", end="")
    for i, (name, f1) in enumerate(zip(labels, f1_per_class)):
        print(f"{name}={f1:.3f}", end="  ")
    print()

# ═══════════════════════════════════════════════════════════════
#  Fig 4: Four Classifier Confusion Matrices
# ═══════════════════════════════════════════════════════════════

def plot_four_confusion_matrices(y_test, classifiers_dict):
    """DT / KNN / SVM / RF confusion matrices in 2x2 layout."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.flatten()
    labels = [ACTIVITY_SHORT[i] for i in range(1, 7)]

    for ax, (name, y_pred) in zip(axes, classifiers_dict.items()):
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        acc = accuracy_score(y_test, y_pred)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="YlOrRd",
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, vmin=0, vmax=1, ax=ax,
                    annot_kws={"fontsize": 14},
                    cbar=True)
        ax.set_xlabel("Predicted", fontsize=13)
        ax.set_ylabel("True", fontsize=13)
        ax.set_title(f"{name}  |  Acc = {acc*100:.1f}%",
                     fontsize=14, fontweight="bold")

    fig.suptitle("Fig 4: Confusion Matrices — 4 Classifiers (Normalized)",
                 fontsize=18, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "04_Four_Classifiers_Confusion_Matrix.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  Fig 5: Classifier Metrics Comparison
# ═══════════════════════════════════════════════════════════════

def plot_metrics_comparison(results):
    """Accuracy + Macro-F1 + Weighted-F1 grouped bar chart."""
    names = [r["name"] for r in results]
    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(names))
    width = 0.22

    metrics = [
        ("accuracy", "Accuracy", "#1976D2"),
        ("macro_f1", "Macro-F1", "#388E3C"),
        ("weighted_f1", "Weighted-F1", "#F57C00"),
    ]

    for i, (key, label, color) in enumerate(metrics):
        values = [r[key] for r in results]
        bars = ax.bar(x + i * width, values, width, label=label,
                      color=color, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks(x + width)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_title("Fig 5: Classifier Performance Comparison — UCI HAR 6-Class Activity Recognition",
                 fontsize=17, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "05_Classifier_Metrics_Comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  Fig 6: Training Time Comparison
# ═══════════════════════════════════════════════════════════════

def plot_training_time(times_dict):
    """Horizontal bar chart: classifier training time."""
    names = list(times_dict.keys())
    times = list(times_dict.values())
    colors = ["#E53935", "#1E88E5", "#FB8C00", "#43A047"]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(names, times, color=colors, edgecolor="white", height=0.55)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{t:.3f}s", va="center", fontsize=13, fontweight="bold")

    ax.set_xlabel("Training Time (seconds)", fontsize=14)
    ax.set_title("Fig 6: Classifier Training Time Comparison", fontsize=16, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "06_Classifier_Training_Time.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  Fig 7: PCA 2D Projection + Decision Boundary
# ═══════════════════════════════════════════════════════════════

def plot_decision_boundary_2d(X_train, y_train, clf, scaler):
    """
    Project 72-D to 2-D via PCA, plot training scatter + decision region overlay.
    Sample decision boundary via meshgrid.
    """
    # PCA projection
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X_train)  # (7352, 2)

    # Train a simplified classifier in PCA 2D space for visualization
    clf_2d = DecisionTreeClassifier(max_depth=6, min_samples_leaf=10, random_state=42)
    clf_2d.fit(X_2d, y_train)

    # meshgrid
    x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
    y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300))
    Z = clf_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(13, 10))
    colors_act = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0", "#00BCD4"]
    cmap_bg = matplotlib.colors.ListedColormap(
        ["#BBDEFB", "#C8E6C9", "#FFE0B2", "#FFCDD2", "#E1BEE7", "#B2EBF2"])

    # Decision region background
    ax.pcolormesh(xx, yy, Z, cmap=cmap_bg, alpha=0.35, shading="auto")

    # Scatter: sample 500 per class to avoid overcrowding
    for act_id in range(1, 7):
        mask = y_train == act_id
        idx = np.where(mask)[0]
        sample = np.random.RandomState(42).choice(
            idx, min(500, len(idx)), replace=False)
        ax.scatter(X_2d[sample, 0], X_2d[sample, 1],
                   s=6, alpha=0.5, color=colors_act[act_id - 1],
                   label=ACTIVITY_LABEL[act_id], rasterized=True)

    ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}%)", fontsize=14)
    ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}%)", fontsize=14)
    ax.set_title(
        "Fig 7: PCA 2D Projection + Decision Tree Decision Regions (max_depth=6)",
        fontsize=16, fontweight="bold")
    ax.legend(fontsize=11, markerscale=3, loc="upper right")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "07_PCA_Decision_Boundary.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  Fig 8: Accuracy Before vs After Tuning
# ═══════════════════════════════════════════════════════════════

def plot_tuning_comparison(before, after):
    """Default vs tuned hyperparameter accuracy comparison."""
    names = [r["name"] for r in before]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    x = np.arange(len(names))
    width = 0.3

    acc_before = [r["accuracy"] for r in before]
    acc_after = [r["accuracy"] for r in after]

    bars1 = ax.bar(x - width / 2, acc_before, width,
                   color="#90CAF9", edgecolor="white", label="Default Params")
    bars2 = ax.bar(x + width / 2, acc_after, width,
                   color="#1565C0", edgecolor="white", label="GridSearchCV Tuned")

    for bar, val in zip(bars1, acc_before):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=11)
    for bar, val in zip(bars2, acc_after):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
        # Gain annotation
        idx = list(acc_after).index(val)
        gain = acc_after[idx] - acc_before[idx]
        if gain > 0.001:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() - 0.035,
                    f"+{gain:.4f}", ha="center", fontsize=10,
                    color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=14)
    ax.set_ylim(0, 1.0)
    ax.set_title("Fig 8: Accuracy Before vs After Hyperparameter Tuning (GridSearchCV 5-Fold CV)",
                 fontsize=17, fontweight="bold")
    ax.legend(fontsize=12, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(SAVE_DIR, "08_Tuning_Comparison.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  ✓ {path}")

# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    print("╔" + "═" * 55 + "╗")
    print("║  Classifier Showcase: Decision Tree + 4-Classifiers ║")
    print("║  UCI HAR Dataset · 72-D features · 6 activities     ║")
    print("║  With GridSearchCV hyperparameter tuning            ║")
    print("╚" + "═" * 55 + "╝")

    # ── 1. Load data ──
    print("\n[1/8] Loading UCI HAR data...")
    X_raw, y = load_uci_har()
    print(f"  Train: {X_raw['train'].shape[0]} windows (21 subjects)")
    print(f"  Test:  {X_raw['test'].shape[0]} windows (9 subjects)")
    print(f"  Channels: body_acc x/y/z + body_gyro x/y/z")

    # ── 2. Feature extraction ──
    print("\n[2/8] Extracting 72-D features (6 channels × 12 features)...")
    X_train = extract_features(X_raw["train"])
    X_test = extract_features(X_raw["test"])
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}")

    # Build feature names
    ch_names = ["AccX", "AccY", "AccZ", "GyroX", "GyroY", "GyroZ"]
    freq_tags = ["PeakFreq", "MeanFreq", "MedianFreq", "Energy",
                 "Entropy", "BandLow", "BandMid", "BandHigh"]
    time_tags = ["Mean", "Var", "PTP", "ZCR"]
    feat_names = []
    for ch in ch_names:
        for t in freq_tags:
            feat_names.append(f"{ch}_{t}")
        for t in time_tags:
            feat_names.append(f"{ch}_{t}")

    # Standardize (required for KNN, SVM)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # ── 3. Default parameter baseline (sklearn original defaults) ──
    print("\n[3/8] sklearn default parameter baseline...")
    results_before = []
    preds_before = {}

    # DT — sklearn default: criterion='gini', max_depth=None, min_samples_leaf=1
    dt_def = DecisionTreeClassifier(random_state=42)
    dt_def.fit(X_train, y["train"])
    preds_before["Decision Tree"] = dt_def.predict(X_test)
    acc = accuracy_score(y["test"], preds_before["Decision Tree"])
    results_before.append({"name": "Decision Tree", "accuracy": acc})
    print(f"  DT (default, unlimited depth): Acc={acc*100:.2f}%  depth={dt_def.get_depth()}")

    # KNN — sklearn default: n_neighbors=5, weights='uniform', metric='euclidean'
    knn_def = KNeighborsClassifier(n_jobs=-1)
    knn_def.fit(X_train_s, y["train"])
    preds_before["KNN"] = knn_def.predict(X_test_s)
    acc = accuracy_score(y["test"], preds_before["KNN"])
    results_before.append({"name": "KNN", "accuracy": acc})
    print(f"  KNN (default, k=5, uniform):      Acc={acc*100:.2f}%")

    # SVM — sklearn default: C=1.0, gamma='scale', kernel='rbf'
    svm_def = SVC(random_state=42)
    svm_def.fit(X_train_s, y["train"])
    preds_before["SVM"] = svm_def.predict(X_test_s)
    acc = accuracy_score(y["test"], preds_before["SVM"])
    results_before.append({"name": "SVM", "accuracy": acc})
    print(f"  SVM (default, C=1):               Acc={acc*100:.2f}%")

    # RF — sklearn default: n_estimators=100, max_depth=None, min_samples_leaf=1
    rf_def = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_def.fit(X_train, y["train"])
    preds_before["Random Forest"] = rf_def.predict(X_test)
    acc = accuracy_score(y["test"], preds_before["Random Forest"])
    results_before.append({"name": "Random Forest", "accuracy": acc})
    print(f"  RF (default, 100 trees):          Acc={acc*100:.2f}%")

    # ── 4. GridSearchCV hyperparameter search ──
    print("\n[4/8] GridSearchCV hyperparameter search (5-Fold Stratified CV)...")
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    preds_tuned = {}
    times_tuned = {}
    results_after = []
    best_params = {}

    # ── 4a. Decision Tree ──
    print("\n  ── Decision Tree: searching max_depth, criterion, min_samples_leaf ──")
    param_dt = {
        "max_depth": [6, 10, 15],
        "min_samples_leaf": [5, 20],
        "criterion": ["gini"],
    }
    t0 = time.time()
    grid_dt = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_dt, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_dt.fit(X_train, y["train"])
    times_tuned["Decision Tree"] = time.time() - t0
    dt_best = grid_dt.best_estimator_
    preds_tuned["Decision Tree"] = dt_best.predict(X_test)
    best_params["Decision Tree"] = grid_dt.best_params_
    acc = accuracy_score(y["test"], preds_tuned["Decision Tree"])
    results_after.append({"name": "Decision Tree", "accuracy": acc})
    print(f"    Best: {grid_dt.best_params_}  →  Test Acc: {acc*100:.2f}%")

    # ── 4b. KNN ──
    print("\n  ── KNN: searching n_neighbors, weights, metric ──")
    param_knn = {
        "n_neighbors": [5, 9, 15],
        "weights": ["uniform"],
        "metric": ["euclidean"],
    }
    t0 = time.time()
    grid_knn = GridSearchCV(
        KNeighborsClassifier(n_jobs=-1),
        param_knn, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_knn.fit(X_train_s, y["train"])
    times_tuned["KNN"] = time.time() - t0
    knn_best = grid_knn.best_estimator_
    preds_tuned["KNN"] = knn_best.predict(X_test_s)
    best_params["KNN"] = grid_knn.best_params_
    acc = accuracy_score(y["test"], preds_tuned["KNN"])
    results_after.append({"name": "KNN", "accuracy": acc})
    print(f"    Best: {grid_knn.best_params_}  →  Test Acc: {acc*100:.2f}%")

    # ── 4c. SVM ──
    print("\n  ── SVM: searching C, gamma ──")
    param_svm = {
        "C": [1, 10, 100],
        "gamma": ["scale"],
    }
    t0 = time.time()
    grid_svm = GridSearchCV(
        SVC(kernel="rbf", random_state=42),
        param_svm, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_svm.fit(X_train_s, y["train"])
    times_tuned["SVM"] = time.time() - t0
    svm_best = grid_svm.best_estimator_
    preds_tuned["SVM"] = svm_best.predict(X_test_s)
    best_params["SVM"] = grid_svm.best_params_
    acc = accuracy_score(y["test"], preds_tuned["SVM"])
    results_after.append({"name": "SVM", "accuracy": acc})
    print(f"    Best: {grid_svm.best_params_}  →  Test Acc: {acc*100:.2f}%")

    # ── 4d. Random Forest ──
    print("\n  ── Random Forest: searching n_estimators, max_depth, max_features ──")
    param_rf = {
        "n_estimators": [100],
        "max_depth": [10, 20],
        "max_features": ["sqrt"],
        "min_samples_leaf": [2],
    }
    t0 = time.time()
    grid_rf = GridSearchCV(
        RandomForestClassifier(random_state=42, n_jobs=-1),
        param_rf, cv=cv, scoring="accuracy", n_jobs=-1)
    grid_rf.fit(X_train, y["train"])
    times_tuned["Random Forest"] = time.time() - t0
    rf_best = grid_rf.best_estimator_
    preds_tuned["Random Forest"] = rf_best.predict(X_test)
    best_params["Random Forest"] = grid_rf.best_params_
    acc = accuracy_score(y["test"], preds_tuned["Random Forest"])
    results_after.append({"name": "Random Forest", "accuracy": acc})
    print(f"    Best: {grid_rf.best_params_}  →  Test Acc: {acc*100:.2f}%")

    # ── 5. Evaluate tuned classifiers ──
    print("\n[5/8] Evaluating tuned classifiers...")
    preds = {}
    times = {}
    classifier_order = ["Decision Tree", "KNN", "SVM", "Random Forest"]
    for name in classifier_order:
        preds[name] = preds_tuned[name]
        times[name] = times_tuned[name]
        acc = accuracy_score(y["test"], preds[name])
        f1m = f1_score(y["test"], preds[name], average="macro")
        f1w = f1_score(y["test"], preds[name], average="weighted")
        # Update results_after
        for r in results_after:
            if r["name"] == name:
                r["macro_f1"] = f1m
                r["weighted_f1"] = f1w
        gain = results_after[classifier_order.index(name)]["accuracy"] \
               - results_before[classifier_order.index(name)]["accuracy"]
        print(f"  {name:<16s}: Acc={acc*100:.2f}%  F1_m={f1m:.4f}  "
              f"F1_w={f1w:.4f}  Δ={gain*100:+.2f}pp  time={times[name]:.1f}s")

    # ── 6. Decision Tree visualizations (3 figs) ──
    print("\n[6/8] Decision Tree visualizations (3 figures)...")
    dt_display = dt_best
    plot_decision_tree_structure(dt_display, feat_names)
    plot_feature_importance_dt(dt_display, feat_names)
    plot_dt_confusion_matrix(y["test"], preds["Decision Tree"])

    # ── 7. Classifier comparison visualizations (4 figs) ──
    print("\n[7/8] Classifier comparison visualizations (4 figures)...")
    plot_four_confusion_matrices(y["test"], preds)
    plot_metrics_comparison(results_after)
    plot_training_time(times)
    plot_decision_boundary_2d(X_train, y["train"], dt_display, scaler)

    # ── 8. Tuning comparison ──
    print("\n[8/8] Tuning comparison figure...")
    plot_tuning_comparison(results_before, results_after)

    # ── Terminal summary ──
    print(f"\n{'═' * 55}")
    print(f"  All figures saved to: {os.path.abspath(SAVE_DIR)}/")
    print(f"  Total: 8 figures")
    print(f"{'═' * 55}")

    # Tuning summary table
    print(f"\n  {'Classifier':<16s} {'Default':<10s} {'Tuned':<10s} {'Gain':<8s} {'Best Params'}")
    print(f"  {'─' * 70}")
    for i, name in enumerate(classifier_order):
        before_acc = results_before[i]["accuracy"]
        after_acc = results_after[i]["accuracy"]
        gain = after_acc - before_acc
        params_str = str(best_params[name])
        print(f"  {name:<16s} {before_acc*100:>7.2f}%   {after_acc*100:>7.2f}%   "
              f"{gain*100:>+6.2f}pp   {params_str}")

    # Key Decision Tree info
    root_feat = feat_names[np.argmax(dt_display.feature_importances_)]
    print(f"\n  DT root feature: {root_feat}")
    print(f"  Best DT params: {best_params['Decision Tree']}")
    print(f"  Tree depth={dt_display.get_depth()}, leaves={dt_display.get_n_leaves()}")

if __name__ == "__main__":
    main()
