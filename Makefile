# 信号与系统大作业：运动传感器数据频域分析与分类
# =====================================================
# 使用方法: make <target>
# 查看所有目标: make help

.PHONY: help demo analyze classify classify-uci classify-wisdm classify-all \
        advanced sliding report showcase all clean

# ── 默认目标 ──────────────────────────────────────────────────

help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║  信号与系统大作业 — Makefile 命令清单                       ║"
	@echo "╠══════════════════════════════════════════════════════════════╣"
	@echo "║  make demo             6 动作时域波形演示                   ║"
	@echo "║  make analyze          UCI 频域深度分析 (A-I, 11 张图)      ║"
	@echo "║  make sliding          滑动窗时频峰度分析 (5 张图)          ║"
	@echo "║  make classify-uci     UCI 决策树分类流水线 (8 图)          ║"
	@echo "║  make classify-wisdm   WISDM 决策树分类流水线               ║"
	@echo "║  make classify-all     双数据集分类流水线                   ║"
	@echo "║  make advanced         最优分类器对比 (V5, 258-D)           ║"
	@echo "║  make report           258 维特征重要性排名 → CSV           ║"
	@echo "║  make showcase         决策树详解 + 四分类对比 (7 张图)    ║"
	@echo "║  make all              运行全部脚本                         ║"
	@echo "║  make clean            清理输出图像                         ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"

# ── 演示 ──────────────────────────────────────────────────────

demo:
	@echo "=== 6 动作时域波形演示 ==="
	python code/main/demo_waveforms.py

# ── 频域分析 ──────────────────────────────────────────────────

analyze:
	@echo "=== UCI 频域深度分析 (A-I, 11 张图) ==="
	python code/main/analysis_uci.py

sliding:
	@echo "=== 滑动窗时频峰度分析 (5 张图) ==="
	python code/main/advanced_analysis_v4.py

# ── 分类流水线 ────────────────────────────────────────────────

classify-uci:
	@echo "=== UCI 决策树分类流水线 (8 图) ==="
	python code/main/main.py uci

classify-wisdm:
	@echo "=== WISDM 决策树分类流水线 ==="
	python code/main/main.py wisdm

classify-all:
	@echo "=== 双数据集分类流水线 ==="
	python code/main/main.py all

classify: classify-uci
	@# shorthand: make classify = make classify-uci

# ── 进阶分类器 ────────────────────────────────────────────────

advanced:
	@echo "=== 最优分类器对比 (V5, 258-D + MI/RF 特征选择) ==="
	python code/main/advanced_classifiers_v5.py

# ── 特征分析 ──────────────────────────────────────────────────

report:
	@echo "=== 258 维特征重要性排名 → CSV ==="
	python code/main/feature_report.py

# ── 分类器展示 ────────────────────────────────────────────────

showcase:
	@echo "=== 决策树详解 + 四分类器对比 (7 张图) ==="
	python code/main/classifier_showcase.py

# ── 全部 ──────────────────────────────────────────────────────

all: demo analyze sliding classify-all advanced report
	@echo ""
	@echo "=== 全部脚本运行完成 ==="
	@echo "输出目录: figures/"
	@ls figures/频谱与特征分析/ figures/分类流水线/ figures/分类器对比/ figures/频域分析/ figures/演示/ 2>/dev/null | head -40

# ── 清理 ──────────────────────────────────────────────────────

clean:
	@echo "清理输出图像..."
	rm -rf figures/频谱与特征分析/ figures/分类流水线/ figures/分类器对比/ figures/频域分析/ figures/演示/
	@echo "已清理 (原始英文名目录 figures/analysis/ figures/uci/ figures/demo/ 未删除)"
