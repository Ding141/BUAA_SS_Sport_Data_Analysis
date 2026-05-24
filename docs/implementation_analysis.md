# 基础任务实现分析文档

> 对照 `docs/assignment.md` 中的基础任务要求，逐项分析代码实现、数学原理与功能解析。

---

## 任务分解

基础任务来自 `docs/assignment.md` 第 2 节，分三大类共 9 个子任务：

| 编号 | 任务 | 实现位置 |
|------|------|----------|
| **A1** | 时频转换：FFT / STFT 将时域信号转为频域信号 | `main.py:229` `compute_fft_spectrum`, `main.py:364` `plot_stft_spectrogram` |
| **A2a** | 频率轴特征：主频率、平均频率、中位数频率 | `main.py:248-253` `extract_frequency_features` |
| **A2b** | 能量特征：频谱能量分布、谱熵 | `main.py:254-261` `extract_frequency_features` |
| **B1** | 决策树分类器 | `main.py:618-641` `run_pipeline` |
| **B2** | 特征工程：频域 + 时域特征合并 | `main.py:285-320` `build_feature_matrix` |
| **B3** | 训练/测试划分 + 准确率 + 混淆矩阵 | `main.py:587-707` `run_pipeline` |
| **C1** | 原始波形图 | `main.py:327` `plot_waveforms` |
| **C2** | 频谱对比图 | `main.py:395` `plot_spectrum_comparison`, `main.py:364` `plot_stft_spectrogram` |
| **C3** | 特征分布散点图 | `main.py:435` `plot_feature_scatter` |
| **C4** | 决策树可视化 | `main.py:481` `plot_decision_tree` |

---

## 一、任务 A：信号频率分析

### A1. 时频转换 — FFT

**实现位置**：`code/main/main.py` 第 229–239 行，函数 `compute_fft_spectrum`

```python
def compute_fft_spectrum(window_signal, fs):
    n = len(window_signal)                                   # 信号长度 N=128
    fft_vals = fft(window_signal)                            # scipy.fft.fft → 复数序列
    mag = np.abs(fft_vals) / n                               # 归一化: |X[k]| / N
    mag = mag[: n // 2 + 1]                                  # 取单边谱 (0 ~ Nyquist)
    mag[1:-1] *= 2                                           # 非DC/Nyquist分量 ×2
    freqs = fftfreq(n, 1 / fs)[: n // 2 + 1]                # 频率轴 (Hz)
    return freqs, mag
```

#### 数学原理

**离散傅里叶变换 (DFT)**：

$$X[k] = \sum_{n=0}^{N-1} x[n] \cdot e^{-j 2\pi k n / N}, \quad k = 0, 1, \dots, N-1$$

其中 $x[n]$ 为 128 点采样信号，$N = 128$，采样频率 $f_s = 50\text{ Hz}$。

**频率分辨率**：

$$\Delta f = \frac{f_s}{N} = \frac{50}{128} \approx 0.39\text{ Hz}$$

即相邻频率点间隔约 0.39 Hz，足以分辨步态基频（~1.5 Hz）及其谐波。

**Nyquist 频率**：

$$f_{\text{Nyq}} = \frac{f_s}{2} = 25\text{ Hz}$$

人体运动有意义频率集中在 0–15 Hz，25 Hz 上限已充分覆盖。单边谱索引范围为 $k = 0, 1, \dots, N/2 = 64$。

**幅度谱归一化**：

$$\text{mag}[k] = \frac{|X[k]|}{N}$$

对单边谱的非直流 ($k \neq 0$) 且非 Nyquist ($k \neq N/2$) 分量乘以 2，补偿共轭对称分量携带的能量：

$$\text{mag}_{\text{onesided}}[k] = \begin{cases} |X[k]| / N, & k = 0 \text{ 或 } k = N/2 \\ 2|X[k]| / N, & k = 1, \dots, N/2-1 \end{cases}$$

#### 功能说明

- **输入**：`window_signal` — 单窗口单通道的 128 点加速度/陀螺仪信号；`fs` — 采样率 (Hz)
- **输出**：`(freqs, mag)` — 频率轴数组 (0~25 Hz, 65 个点) 与对应归一化幅度
- **用途**：为后续 8 个频域特征的提取提供频谱数据

---

### A1 (续). 时频转换 — STFT

**实现位置**：`code/main/main.py` 第 364–392 行，函数 `plot_stft_spectrogram`

```python
f_vals, t_vals, Sxx = spectrogram(
    raw_data[idx, :, 0], fs=fs,
    nperseg=min(32, raw_data.shape[1] // 4),      # 子窗长度 32 点
    noverlap=min(24, raw_data.shape[1] // 5),      # 重叠 24 点 (75%)
    nfft=min(64, raw_data.shape[1] // 2),          # FFT 点数 64
)
# Sxx 是功率谱密度 (V²/Hz)，取 10·log₁₀ 转为 dB 便于可视化
im = ax.pcolormesh(t_vals, f_vals, 10 * np.log10(Sxx + 1e-12), ...)
```

#### 数学原理

**短时傅里叶变换 (STFT)**：

$$\text{STFT}\{x[n]\}(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] \cdot w[n - mR] \cdot e^{-j\omega n}$$

其中 $w[n]$ 为窗函数（默认 Hann 窗），$m$ 为时间帧索引，$R$ 为帧移（步长）。

参数选择：子窗 $L = 32$ 点 (0.64 s)，帧移 $R = L - \text{noverlap} = 8$ 点，FFT 点数 $N_{\text{fft}} = 64$（补零至 64）。

**时间分辨率**：

$$\Delta t = \frac{L}{f_s} = \frac{32}{50} = 0.64\text{ s}$$

**频率分辨率**：

$$\Delta f = \frac{f_s}{N_{\text{fft}}} = \frac{50}{64} \approx 0.78\text{ Hz}$$

**功率谱密度 (PSD) 转 dB**：

$$S_{xx}[m, k]_{\text{dB}} = 10 \cdot \log_{10}\left(S_{xx}[m, k] + \varepsilon\right)$$

对数尺度增强了弱频率分量的可视性，使静态动作的低幅值频谱也能清晰呈现。

#### 功能说明

- **参数选择理由**：32 点子窗 (0.64 s) 在时域上可分辨步态周期内的不同相位；75% 重叠保证时间轴的平滑过渡
- **输出**：时频谱图，展示频率内容如何随时间演化，区分稳态动作（频谱恒定）与周期动作（频谱随时间脉动）
- **对应关系**：analysis_uci.py 中也实现了更详细的 STFT 分析（`figure_d_stft_comparison`），覆盖 Walk/Sit/Stand/Lay 四类动作

---

### A2a. 频率轴特征提取

**实现位置**：`code/main/main.py` 第 242–272 行，函数 `extract_frequency_features`

```python
def extract_frequency_features(mag, freqs):
    f = {}
    total_mag = np.sum(mag)
    eps = 1e-12

    # ① 主频率 (Peak Frequency)
    f["PeakFreq"] = freqs[np.argmax(mag)]

    if total_mag > eps:
        # ② 平均频率 / 频谱质心 (Mean/Spectral Centroid)
        f["MeanFreq"] = np.sum(freqs * mag) / total_mag

        # ③ 中位数频率 (Median Frequency)
        cumulative = np.cumsum(mag)
        f["MedianFreq"] = freqs[np.searchsorted(cumulative, total_mag / 2)]
        ...
```

#### 数学原理

设有单边幅度谱 $\text{mag}[k]$，对应频率 $f_k = k \cdot \Delta f$。

**① 主频率 (Peak Frequency)**：

$$f_{\text{peak}} = f_{k^*}, \quad k^* = \arg\max_k \text{mag}[k]$$

幅度谱最大值处的频率。物理含义：传感器感知到的最强周期性运动分量。例如步行时 body_acc_x 的主频约为 1.5–2.0 Hz（步频）。

**② 平均频率 / 频谱质心 (Mean Frequency / Spectral Centroid)**：

$$f_{\text{mean}} = \frac{\sum_{k} f_k \cdot \text{mag}[k]}{\sum_k \text{mag}[k]}$$

以幅度为权重的加权平均频率。物理含义：频谱能量的"中心位置"。动态动作质心偏低频（1–3 Hz），静态动作质心接近 DC。

**③ 中位数频率 (Median Frequency)**：

$$f_{\text{median}} = f_{k_m}, \quad k_m = \min \left\{ k \;\middle|\; \sum_{i=0}^{k} \text{mag}[i] \geq \frac{1}{2} \sum_{i} \text{mag}[i] \right\}$$

累计幅度达到总幅度 50% 时的截止频率。比质心更稳健——不受离群高频分量的影响。

#### 功能说明

这三个特征从不同角度描述频谱的能量分布位置，共同构成"频域指纹"：
- **PeakFreq** 回答"哪个频率最活跃"
- **MeanFreq** 回答"能量整体偏向哪里"
- **MedianFreq** 回答"能量的中间线在哪里"

代码实现使用 `np.searchsorted(cumulative, total_mag/2)` 在累计幅度数组中二分查找 50% 分位点，时间复杂度 $O(\log N)$。

---

### A2b. 能量特征提取

**实现位置**：同 `extract_frequency_features`，第 254–261 行

```python
        # ④ 频谱能量 (Spectral Energy)
        f["SpectralEnergy"] = np.sum(mag ** 2)

        # ⑤ 谱熵 (Spectral Entropy)
        prob = mag / total_mag                              # 归一化为概率分布
        f["SpectralEntropy"] = stats_entropy(prob + eps)    # scipy.stats.entropy

        # ⑥–⑧ 三频带能量
        nyq = freqs[-1]                                     # Nyquist = 25 Hz
        f["BandEnergy_Low"]  = np.sum(mag[(freqs >= 0) & (freqs < nyq * 0.2)] ** 2)   # 0–5 Hz
        f["BandEnergy_Mid"]  = np.sum(mag[(freqs >= nyq * 0.2) & (freqs < nyq * 0.6)] ** 2)  # 5–15 Hz
        f["BandEnergy_High"] = np.sum(mag[(freqs >= nyq * 0.6)] ** 2)                   # 15–25 Hz
```

#### 数学原理

**④ 频谱能量 (Spectral Energy)**：

$$E_{\text{spectral}} = \sum_k \text{mag}[k]^2$$

根据 Parseval 定理，频域能量与时域能量等价（差一个常数因子）。该特征直接反映传感器感受到的运动强度——动态动作（行走/上下楼）的能量比静态动作高 200–2000 倍。

**⑤ 谱熵 (Spectral Entropy)**：

$$H = -\sum_k p[k] \cdot \log_2\left(p[k] + \varepsilon\right), \quad p[k] = \frac{\text{mag}[k]}{\sum_i \text{mag}[i]}$$

将归一化幅度谱视为概率分布，计算其信息熵。

- **高谱熵** ($H$ 较大)：频谱"杂乱"，能量分散在多个频率上 → 典型于动态动作（步态包含基频 + 多次谐波）
- **低谱熵** ($H$ 较小)：频谱"纯净"，能量集中在少数频率 → 典型于静态动作（几乎只有 DC 分量）

**⑥–⑧ 频带能量 (Band Energy)**：

$$E_{\text{band}} = \sum_{k: f_k \in [f_{\text{lo}}, f_{\text{hi}})} \text{mag}[k]^2$$

三频带划分：
| 频带 | 频率范围 | 物理含义 |
|------|----------|----------|
| Low | 0–5 Hz (0–20% Nyq) | DC + 步态基频，静态动作能量几乎全在此 |
| Mid | 5–15 Hz (20–60% Nyq) | 步态谐波区，上下楼时此频带能量显著 |
| High | 15–25 Hz (60–100% Nyq) | 高频冲击/传感器噪声 |

#### 功能说明

这三个能量特征回答"信号有多强、多复杂、能量分布在哪里"：
- `SpectralEnergy` 直接区分动/静态（动态 >> 静态）
- `SpectralEntropy` 区分运动复杂度（平地走 vs 上下楼）
- `BandEnergy_Low/Mid/High` 提供精细的频谱形状刻画

`analysis_uci.py` 中还使用了更细粒度的 5 段分频（DC–2, 2–5, 5–10, 10–15, 15–25 Hz），进一步区分了平地走（~1.5 Hz）和上楼（~1 Hz + 更高谐波）。

---

### 特征提取汇总

**实现位置**：`code/main/main.py` 第 285–320 行，函数 `build_feature_matrix`

```python
def build_feature_matrix(raw_data, channel_names, fs, verbose=True):
    n_windows = raw_data.shape[0]          # 窗口数 (训练集 7352)
    n_channels = raw_data.shape[2]         # 通道数 (6)
    all_rows = []

    for i in range(n_windows):
        row = []
        for ch in range(n_channels):
            signal = raw_data[i, :, ch]                        # (128,) 单通道信号
            freqs, mag = compute_fft_spectrum(signal, fs)      # FFT → 频域
            freq_f = extract_frequency_features(mag, freqs)     # 8 频域特征
            time_f = extract_time_features(signal)              # 4 时域特征
            row.extend(freq_f.values())
            row.extend(time_f.values())
        all_rows.append(row)

    return np.array(all_rows), col_names     # (n_windows, 72), (72,) 列名
```

**时域特征（每通道 4 维）**：

| 特征 | 公式 | 物理意义 |
|------|------|----------|
| Mean | $\mu = \frac{1}{N}\sum x[n]$ | 传感器静态偏置（重力分量在感测轴上的投影） |
| Var | $\sigma^2 = \frac{1}{N}\sum (x[n]-\mu)^2$ | 信号波动强度，等价于交流能量 |
| PeakToPeak | $\max(x) - \min(x)$ | 窗口内最大冲击幅度 |
| ZeroCrossingRate | $\frac{1}{N}\sum \mathbb{1}[\text{sign}(x[n]) \neq \text{sign}(x[n-1])]$ | 过零率，对频率的粗略估计 |

**最终特征矩阵**：

$$X \in \mathbb{R}^{n_{\text{windows}} \times 72}, \quad 72 = 6\text{ 通道} \times (8\text{ 频域} + 4\text{ 时域})$$

---

## 二、任务 B：动作分类模型

### B1. 决策树分类器

**实现位置**：`code/main/main.py` 第 618–641 行

```python
# ── 超参数搜索 (GridSearchCV + 5-Fold Stratified Cross-Validation) ──
param_grid = {
    "max_depth": [6, 8, 10, 12, 15, None],        # 树深度上限
    "min_samples_split": [5, 10, 20, 50],          # 内部节点最少样本数
    "min_samples_leaf": [2, 5, 10, 20],            # 叶节点最少样本数
}
grid = GridSearchCV(
    DecisionTreeClassifier(criterion="gini", random_state=42),
    param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),  # 5 折分层交叉验证
    scoring="accuracy",
    n_jobs=-1,
)
grid.fit(X_train, y["train"])                       # 搜索最优超参数
print(f"最佳参数: {grid.best_params_}")              # 输出: max_depth=12, min_samples_leaf=5

# ── 用最优参数训练最终模型 ──
clf = DecisionTreeClassifier(criterion="gini", random_state=42, **grid.best_params_)
clf.fit(X_train, y["train"])
y_pred = clf.predict(X_test)
```

#### 数学原理

**决策树 (Decision Tree)** 是一种基于递归分区的监督学习算法。

**Gini 不纯度 (Gini Impurity)**：

节点 $m$ 的 Gini 指数定义为：

$$G_m = \sum_{c=1}^{C} p_{m,c}(1 - p_{m,c}) = 1 - \sum_{c=1}^{C} p_{m,c}^2$$

其中 $p_{m,c}$ 是节点 $m$ 中类别 $c$ 的样本比例，$C = 6$ 为类别数。

- $G_m = 0$：节点内所有样本属于同一类（完全纯净）
- $G_m = 1 - 1/C$（最大值）：各类均匀分布（完全混杂）

**分裂准则**：

选择特征 $j$ 和阈值 $t$，最小化分裂后的加权 Gini 不纯度：

$$(j^*, t^*) = \arg\min_{j, t} \left[ \frac{n_L}{n_m} G_L + \frac{n_R}{n_m} G_R \right]$$

其中 $n_L, n_R$ 分别为左、右子节点的样本数。

**特征重要性 (Feature Importance)**：

$$I_j = \sum_{m: \text{split on } j} \frac{n_m}{n_{\text{total}}} \cdot \left(G_m - \frac{n_L}{n_m}G_L - \frac{n_R}{n_m}G_R\right)$$

即该特征在所有分裂点上的不纯度减少量（按样本量加权）。

**GridSearchCV**：

对于参数网格 $\Theta = \{\theta_1, \dots, \theta_K\}$（共 $6 \times 4 \times 4 = 96$ 种组合），5 折交叉验证的分数为：

$$\text{CV-score}(\theta) = \frac{1}{5} \sum_{k=1}^{5} \text{Accuracy}(M_\theta^{(k)})$$

其中 $M_\theta^{(k)}$ 是在第 $k$ 折验证集上训练的模型。

**StratifiedKFold** 确保每折中各类别比例与整体一致：

$$\frac{n_{c,k}}{n_k} \approx \frac{n_c}{n_{\text{total}}}, \quad \forall c, k$$

#### 功能说明

- **超参数搜索**：GridSearchCV 搜索 96 种参数组合，5 折 CV 评估，选出泛化最优的配置
- **最优参数**：`max_depth=12`（限制深度防止过拟合），`min_samples_leaf=5`（避免叶节点样本过少）
- **训练方式**：21 人训练 / 9 人测试（UCI 官方划分），同一人的窗口不会同时出现在训练和测试集，保证评估的可靠性
- **结果**：UCI 测试准确率 77.67%，WISDM 63.71%（仅加速度，无陀螺仪）

---

### B2. 特征工程流程

```
原始信号                    逐通道 FFT                 频域特征
(N_windows, 128, 6)  ────────────────────────────→  每通道 8 维 (PeakFreq,
  │                                                  MeanFreq, MedianFreq,
  │                                                  SpectralEnergy,
  │                                                  SpectralEntropy,
  │                                                  BandLow/Mid/High)
  │
  └── 时域统计 ──────────────────────────────────→  时域特征
      每通道: mean, var, peak-to-peak, ZCR          每通道 4 维

                         ↓
                  拼接 → 6通道 × (8频域 + 4时域) = 72 维特征向量
                         ↓
                  (N_windows, 72) 特征矩阵
                         ↓
                    决策树分类器
```

### B3. 训练/测试划分 + 评估

**实现位置**：`code/main/main.py` 第 596–707 行

```python
# 数据划分: train/test 来自 UCI 官方预划分 (21人/9人)
X_train, feat_names = build_feature_matrix(X_raw["train"], channel_names, fs)  # (7352, 72)
X_test, _          = build_feature_matrix(X_raw["test"],  channel_names, fs)  # (2947, 72)

# 评估指标
test_acc    = accuracy_score(y["test"], y_pred)                                  # 整体准确率
macro_f1    = f1_score(y["test"], y_pred, average='macro')                       # 宏平均 F1
weighted_f1 = f1_score(y["test"], y_pred, average='weighted')                    # 加权 F1
cm          = confusion_matrix(y_true, y_pred)                                   # 混淆矩阵
```

#### 数学原理

**准确率 (Accuracy)**：

$$\text{Acc} = \frac{\sum_{i=1}^{n} \mathbb{1}[y_i = \hat{y}_i]}{n}$$

**F1 分数**：

$$\text{F1}_c = 2 \cdot \frac{\text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

$$\text{Macro-F1} = \frac{1}{C}\sum_{c=1}^{C} \text{F1}_c, \quad \text{Weighted-F1} = \sum_{c=1}^{C} \frac{n_c}{n} \cdot \text{F1}_c$$

Macro-F1 对各分类平等对待（适合评估小类表现），Weighted-F1 按样本量加权（反映整体性能）。

**消融实验**：

分别用"仅时域特征 (24-D)"、"仅频域特征 (48-D)"、"融合特征 (72-D)"训练决策树，量化频域特征的贡献：

$$\Delta_{\text{freq}} = \text{Acc}_{\text{fusion}} - \text{Acc}_{\text{time-only}}$$

实际结果：$\Delta_{\text{freq}} = +2.88\text{pp}$（77.67% − 74.79%），验证了频域分析的价值。

---

## 三、任务 C：可视化演示

### C1. 原始波形图

**实现位置**：`main.py:327` `plot_waveforms`

- **内容**：每种动作取一条典型窗口（训练集中间位置的样本），画加速度计三轴（body_acc_x/y/z）的时域波形
- **观察要点**：动态动作（Walking/Upstairs/Downstairs）呈现明显的周期性波动（~1.5–2 Hz 步频）；静态动作（Sitting/Standing/Laying）波形接近平坦，仅有微小的高频噪声
- **对应图片**：`分类流水线/01_原始波形图.png`

### C2. 频谱对比图

**实现位置**：`main.py:395` `plot_spectrum_comparison` + `main.py:364` `plot_stft_spectrogram`

- **FFT 幅度谱对比**：取两组动作对（Walking vs Walking Upstairs, Walking vs Laying, Walking Upstairs vs Laying），分别画 FFT 幅度谱并标注主峰频率
- **STFT 时频谱**：取前 4 类动作的 body_acc_x 通道，画出频率随时间演化的热力图（dB 刻度）
- **观察要点**：动态动作频谱在 1–5 Hz 有明显主峰，静态动作频谱能量集中在 DC 附近；STFT 显示动态动作的频谱随时间脉动（步态周期），静态动作频谱恒定
- **对应图片**：`分类流水线/02_STFT时频谱.png`, `分类流水线/03_FFT幅度谱对比.png`

### C3. 特征分布散点图

**实现位置**：`main.py:435` `plot_feature_scatter`

- **内容**：从 72 维特征中选取 MeanFreq、SpectralEntropy、SpectralEnergy、PeakFreq 四类频域特征（来自不同通道），两两配对生成散点图矩阵（最多 6 幅）
- **观察要点**：不同动作在频域特征空间中形成可区分的聚类——动态动作（蓝/绿/橙）集中在高频/高能/高熵区域，静态动作（红/紫/青）集中在低频/低能/低熵区域
- **对应图片**：`分类流水线/04_频域特征散点图矩阵.png`

### C4. 决策树可视化

**实现位置**：`main.py:481` `plot_decision_tree`

- **内容**：使用 `sklearn.tree.plot_tree` 导出决策树的前 3 层结构，节点标注分裂特征和阈值、各类样本比例
- **观察要点**：根节点通常选择 SpectralEnergy（区分动/静态的最强特征），第二层开始区分具体动作类型
- **对应图片**：`分类流水线/05_决策树结构图.png`

### 补充可视化（超出基础要求，有助于分析与报告）

| 图 | 函数 | 内容 |
|----|------|------|
| 混淆矩阵 | `plot_confusion_matrix` (line 497) | 6×6 矩阵，显示各类的预测分布 |
| 特征重要性 | `plot_feature_importance` (line 517) | Top-20 Gini 重要度，红色=频域，蓝色=时域 |
| 频带能量饼图 | `plot_band_energy` (line 540) | 4 类动作的三频带能量占比 |

---

## 四、实现完整性对照表

| 基础任务要求 | 实现状态 | 实现文件:行号 | 输出验证 |
|-------------|---------|-------------|---------|
| A1. FFT 时频转换 | ✅ 完整 | `main.py:229` `compute_fft_spectrum` | 72 维特征矩阵中 48 维频域特征 |
| A1. STFT 时频转换 | ✅ 完整 | `main.py:364` `plot_stft_spectrogram` | STFT 时频谱图 |
| A1. 分窗处理 | ✅ 完整 | UCI 预分窗 (128点), WISDM `_segment_windows` (line 164) | 7352 训练窗 / 2947 测试窗 |
| A2a. 主频率 PeakFreq | ✅ 完整 | `main.py:248` | 特征列 `*_PeakFreq` (6列) |
| A2a. 平均频率 MeanFreq | ✅ 完整 | `main.py:251` | 特征列 `*_MeanFreq` (6列) |
| A2a. 中位数频率 MedianFreq | ✅ 完整 | `main.py:253` | 特征列 `*_MedianFreq` (6列) |
| A2b. 频谱能量分布 | ✅ 完整 | `main.py:254,259-261` | SpectralEnergy + 3 频带能量 |
| A2b. 谱熵 | ✅ 完整 | `main.py:256` | 特征列 `*_SpectralEntropy` (6列) |
| B1. 决策树分类器 | ✅ 完整 | `main.py:618-641` | 最优参数 + 77.67% 准确率 |
| B2. 特征工程（频域+时域合并） | ✅ 完整 | `main.py:285` `build_feature_matrix` | 72 维特征向量 |
| B2. 时域均值/方差 | ✅ 完整 | `main.py:275` `extract_time_features` | Mean, Var 各 6 列 |
| B3. 训练集/测试集划分 | ✅ 完整 | `main.py:596-616` (21人/9人) | 官方预划分 |
| B3. 准确率 + 混淆矩阵 | ✅ 完整 | `main.py:644,497` | Accuracy + CM 图 |
| C1. 原始波形图 | ✅ 完整 | `main.py:327` `plot_waveforms` | figure1_waveforms.png |
| C2. 频谱对比图 | ✅ 完整 | `main.py:395,364` | figure2_stft + figure3_spectrum |
| C3. 特征分布散点图 | ✅ 完整 | `main.py:435` `plot_feature_scatter` | figure4_scatter.png |
| C4. 决策树可视化 | ✅ 完整 | `main.py:481` `plot_decision_tree` | figure5_tree.png |

**覆盖率：9/9 基础任务全部实现。** 代码分布在 `code/main/main.py`（主要）、`code/main/analysis_uci.py`（频域深度分析补充）和 `code/main/demo_waveforms.py`（演示补充）中。

---

## 五、运行命令

```bash
# 基础任务的核心脚本
make classify-uci      # 运行 main.py uci → 决策树分类 + 8 张可视化图
make analyze           # 运行 analysis_uci.py → 频域深度分析 11 张图
make demo              # 运行 demo_waveforms.py →	 6 动作波形演示

# 或直接调用
python code/main/main.py uci
python code/main/analysis_uci.py
python code/main/demo_waveforms.py
```

所有基础任务的实现集中在 `code/main/main.py` 一个文件中（核心流水线），`analysis_uci.py` 提供更深的频域分析视角。进阶分类器（KNN/SVM/RF）和 V2–V5 特征工程属于扩展任务，不在本文档范围内。
