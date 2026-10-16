"""折外预测上的校准分析：Brier score、Brier skill score、校准曲线（reliability diagram）。

只读 results/predictions/oof_*.csv，不读测试集。运行：
    python scripts/05_calibration.py
输出：
    results/tables/oof_calibration_summary.csv   每个模型一行：Brier、Brier_ref、skill、AP、ROC-AUC
    results/tables/oof_calibration_bins.csv      每个模型十个等频箱：平均预测概率、实际违约率、样本数
    results/figures/oof_calibration_curve.png    六个模型的校准曲线，对角线为完美校准
    results/figures/oof_score_histograms.png     六个模型的分数分布
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config as C
from src.evaluation.metrics import calibration_table, ranking_metrics

MODEL_NAMES = ("logistic", "gaussian_nb", "knn", "decision_tree", "random_forest", "adaboost")
LABELS = {
    "logistic": "Logistic Regression",
    "gaussian_nb": "Gaussian NB",
    "knn": "KNN",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "adaboost": "AdaBoost",
}


def main():
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    oof = {m: pd.read_csv(C.PRED_DIR / f"oof_{m}.csv") for m in MODEL_NAMES}

    summary, bins = [], []
    for m, d in oof.items():
        r = ranking_metrics(d.y, d.score)
        summary.append({"model": m, **r, "n_distinct_scores": d.score.nunique()})
        t = calibration_table(d.y, d.score, n_bins=10)
        t.insert(0, "bin", range(len(t)))
        t.insert(0, "model", m)
        bins.append(t)
    summary = pd.DataFrame(summary).sort_values("brier")
    bins = pd.concat(bins, ignore_index=True)
    summary.to_csv(C.TABLE_DIR / "oof_calibration_summary.csv", index=False)
    bins.to_csv(C.TABLE_DIR / "oof_calibration_bins.csv", index=False)

    # 校准曲线
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    for m in MODEL_NAMES:
        t = bins[bins.model == m]
        ax.plot(t.mean_score, t.default_rate, marker="o", ms=4, label=LABELS[m])
    ax.set(
        xlabel="Mean predicted probability (equal-frequency bins)",
        ylabel="Observed default rate",
        title="Calibration curves (out-of-fold, development set)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "oof_calibration_curve.png", dpi=180)
    plt.close(fig)

    # 分数分布：解释为什么有的模型校准差
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, m in zip(axes.ravel(), MODEL_NAMES):
        d = oof[m]
        ax.hist(d.score[d.y == 0], bins=40, range=(0, 1), alpha=0.6, label="No default")
        ax.hist(d.score[d.y == 1], bins=40, range=(0, 1), alpha=0.6, label="Default")
        ax.set_title(LABELS[m], fontsize=10)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("Predicted P(default)")
    fig.suptitle("Out-of-fold score distributions by true class (log count)")
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "oof_score_histograms.png", dpi=180)
    plt.close(fig)

    cols = ["model", "brier", "brier_ref", "brier_skill", "ap", "roc_auc", "n_distinct_scores"]
    print(summary[cols].round(4).to_string(index=False))
    print("\nSaved 2 tables in results/tables/ and 2 figures in results/figures/.")


if __name__ == "__main__":
    main()
