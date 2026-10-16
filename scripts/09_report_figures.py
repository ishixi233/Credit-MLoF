"""报告正文和附录的图，全部从已提交的表格和预测文件生成，不重新拟合，不改任何结果。

运行：
    python scripts/09_report_figures.py
输出（results/figures/，每张 PNG + PDF）：
    fig_class_balance          训练开发集违约比例
    fig_cost_improvement       测试集：各模型相对最佳平凡策略的成本降低百分比（按成本比）
    fig_roc_isocost            测试集：前三名 ROC 曲线 + r=1,5,10 的等成本线与冻结操作点
    fig_confusion_rf           测试集：随机森林在默认阈值 0.5 与冻结阈值（r=5）下的混淆矩阵
    fig_calibration_test       测试集校准曲线
    fig_thresholds_vs_ratio    折外选出的阈值随成本比的变化，对比理论值 1/(1+r)
    fig_pr_curves              测试集 PR 曲线（附录）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config as C
from src.data.load import load_dev
from src.evaluation.metrics import calibration_table, decision_metrics
from src.visualization.style import COLORS, FULL, HALF, LABELS, MODEL_ORDER, apply, save


def load_test_preds() -> dict[str, pd.DataFrame]:
    return {m: pd.read_csv(C.PRED_DIR / f"test_{m}.csv") for m in MODEL_ORDER}


def fig_class_balance():
    _, y, _, _ = load_dev()
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(HALF[0], 2.2))
    bars = ax.bar(["No default (0)", "Default (1)"], counts.values, color=[COLORS["random_forest"], COLORS["adaboost"]], width=0.6)
    for b, n in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, n, f"{n:,}\n({n / len(y):.1%})", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Clients (development set)")
    ax.set_ylim(0, counts.max() * 1.25)
    ax.grid(axis="x", visible=False)
    save(fig, C.FIG_DIR / "fig_class_balance")


def fig_cost_improvement():
    d = pd.read_csv(C.TABLE_DIR / "test_decisions.csv")
    fig, ax = plt.subplots(figsize=FULL)
    for m in MODEL_ORDER:
        s = d[d.model == m].sort_values("cost_ratio")
        ax.plot(s.cost_ratio, s.improvement_vs_trivial_pct, marker="o", color=COLORS[m], label=LABELS[m])
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(list(C.COST_RATIOS))
    ax.set_xlabel("Cost ratio $r$ (one missed default = $r$ false alarms)")
    ax.set_ylabel("Cost reduction vs. best trivial policy (%)")
    ax.set_title("Test-set weighted cost at frozen thresholds, relative to flagging everyone / no one")
    ax.set_ylim(-4, 46)
    ax.axvline(3.52, color="gray", ls=":", lw=0.8)
    ax.text(3.35, 45, "best trivial policy: flag no one", ha="right", va="top", fontsize=7, color="gray")
    ax.text(3.7, 45, "best trivial policy: flag everyone", ha="left", va="top", fontsize=7, color="gray")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, C.FIG_DIR / "fig_cost_improvement")


def fig_roc_isocost(preds):
    frozen = json.loads(C.FROZEN_FILE.read_text())
    y = preds["random_forest"].y.to_numpy()
    pi = y.mean()
    top = ("random_forest", "adaboost", "logistic")
    fig, ax = plt.subplots(figsize=(FULL[0], 3.6))
    for m in top:
        fpr, tpr, _ = roc_curve(y, preds[m].score)
        ax.plot(fpr, tpr, color=COLORS[m], label=LABELS[m])
        for r, mk in zip((1, 5, 10), ("s", "o", "^")):
            dm = decision_metrics(y, preds[m].score, frozen["models"][m]["thresholds"][str(r)], r)
            ax.plot(dm["fp"] / (dm["fp"] + dm["tn"]), dm["recall"], marker=mk, color=COLORS[m], ms=5, mec="black", mew=0.5, ls="")
    # 等成本线：成本 = r*pi*(1-TPR) + (1-pi)*FPR 为常数 → 斜率 (1-pi)/(r*pi)
    for r, mk in zip((1, 5, 10), ("s", "o", "^")):
        slope = (1 - pi) / (r * pi)
        dm = decision_metrics(y, preds["random_forest"].score, frozen["models"]["random_forest"]["thresholds"][str(r)], r)
        x0, y0 = dm["fp"] / (dm["fp"] + dm["tn"]), dm["recall"]
        xs = np.linspace(0, 1, 50)
        ax.plot(xs, y0 + slope * (xs - x0), color="gray", lw=0.7, ls="--")
        ax.plot([], [], marker=mk, color="gray", ls="", mec="black", mew=0.5, label=f"frozen point, $r={r}$")
    ax.plot([0, 1], [0, 1], color="lightgray", lw=0.6)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate (recall)")
    ax.set_title("ROC curves of the top three models with iso-cost lines (slope $(1-\\pi)/(r\\pi)$, through RF)")
    ax.legend(loc="lower right", ncol=2)
    save(fig, C.FIG_DIR / "fig_roc_isocost")


def fig_confusion_rf(preds):
    frozen = json.loads(C.FROZEN_FILE.read_text())
    d = preds["random_forest"]
    t5 = frozen["models"]["random_forest"]["thresholds"]["5"]
    fig, axes = plt.subplots(1, 2, figsize=(FULL[0], 2.6))
    fig.subplots_adjust(top=0.72, wspace=0.5)
    for ax, (t, title) in zip(axes, [(0.5, "Default threshold 0.5"), (t5, f"Frozen threshold {t5:.3f} ($r=5$)")]):
        m = decision_metrics(d.y, d.score, t, 5)
        mat = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=mat.max() * 1.3)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{mat[i, j]:,}", ha="center", va="center", fontsize=9, color="black")
        ax.set_xticks([0, 1], ["Pred. no default", "Pred. default"])
        ax.set_yticks([0, 1], ["No default", "Default"])
        ax.set_title(f"{title}\ncost/client {m['cost_per_client']:.3f}, recall {m['recall']:.2f}", fontsize=8.5)
        ax.grid(False)
    fig.suptitle("Random Forest on the test set (9,003 clients)", y=0.98)
    save(fig, C.FIG_DIR / "fig_confusion_rf")


def fig_calibration_test(preds):
    fig, ax = plt.subplots(figsize=(HALF[0] + 0.4, HALF[0] + 0.4))
    ax.plot([0, 1], [0, 1], "k--", lw=0.7, label="Perfect calibration")
    for m in MODEL_ORDER:
        t = calibration_table(preds[m].y, preds[m].score, 10)
        ax.plot(t.mean_score, t.default_rate, marker="o", color=COLORS[m], label=LABELS[m])
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean predicted probability (decile bins)", ylabel="Observed default rate")
    ax.set_title("Calibration on the test set")
    ax.legend(loc="upper left")
    save(fig, C.FIG_DIR / "fig_calibration_test")


def fig_thresholds_vs_ratio():
    frozen = json.loads(C.FROZEN_FILE.read_text())
    fig, ax = plt.subplots(figsize=FULL)
    rs = list(C.COST_RATIOS)
    for m in MODEL_ORDER:
        ts = [frozen["models"][m]["thresholds"][str(r)] for r in rs]
        ax.plot(rs, ts, marker="o", color=COLORS[m], label=LABELS[m])
    ax.plot(rs, [1 / (1 + r) for r in rs], "k--", lw=0.9, label="$1/(1+r)$ if calibrated")
    ax.set_xticks(rs)
    ax.set(xlabel="Cost ratio $r$", ylabel="Selected alert threshold (out-of-fold)", ylim=(-0.02, 1.02))
    ax.set_title("Cost-minimising thresholds move with $r$; distance from $1/(1+r)$ reflects miscalibration")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    save(fig, C.FIG_DIR / "fig_thresholds_vs_ratio")


def fig_pr_curves(preds):
    fig, ax = plt.subplots(figsize=(HALF[0] + 0.4, HALF[0] + 0.2))
    y = preds["random_forest"].y
    for m in MODEL_ORDER:
        p, r, _ = precision_recall_curve(y, preds[m].score)
        ax.plot(r, p, color=COLORS[m], label=LABELS[m], lw=1)
    ax.axhline(y.mean(), color="gray", ls=":", lw=0.8, label=f"Default rate {y.mean():.3f}")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Recall", ylabel="Precision")
    ax.set_title("Precision–recall curves on the test set")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    save(fig, C.FIG_DIR / "fig_pr_curves")


def main():
    apply()
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    preds = load_test_preds()
    fig_class_balance()
    fig_cost_improvement()
    fig_roc_isocost(preds)
    fig_confusion_rf(preds)
    fig_calibration_test(preds)
    fig_thresholds_vs_ratio()
    fig_pr_curves(preds)
    print("Saved 7 figures (PNG + PDF) in results/figures/.")


if __name__ == "__main__":
    main()
