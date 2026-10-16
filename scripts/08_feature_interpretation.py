"""特征解释：从冻结后拟合的 pipeline（models/*.joblib）提取 Logistic 系数和树模型的特征重要性。

不读测试集，不重新拟合。运行：
    python scripts/08_feature_interpretation.py
输出：
    results/tables/feature_logistic_coefficients.csv   标准化特征上的系数（one-hot 列为相对所有类别的平均）
    results/tables/feature_importance_trees.csv        决策树、随机森林、AdaBoost 的不纯度重要性，按原始变量汇总
    results/figures/fig_feature_logistic.{png,pdf}     Logistic 系数前 15
    results/figures/fig_feature_importance.{png,pdf}   三个树模型的重要性前 12（按原始变量）
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.visualization.style import COLORS, FULL, LABELS, apply, save

MODEL_DIR = C.ROOT / "models"
TREES = ("decision_tree", "random_forest", "adaboost")


def feature_names(pipe) -> list[str]:
    return list(pipe.named_steps["prep"].get_feature_names_out())


def raw_variable(name: str) -> str:
    """one-hot 列 'EDUCATION_2' → 'EDUCATION'；'PAY_0_-1' → 'PAY_0'；数值列原样。"""
    m = re.match(r"^(SEX|EDUCATION|MARRIAGE|PAY_\d)_(-?\d+)$", name)
    return m.group(1) if m else name


def logistic_table() -> pd.DataFrame:
    pipe = joblib.load(MODEL_DIR / "logistic.joblib")
    names = feature_names(pipe)
    coef = pipe.named_steps["clf"].coef_.ravel()
    df = pd.DataFrame({"feature": names, "coef": coef})
    df["variable"] = df.feature.map(raw_variable)
    df["type"] = np.where(df.feature == df.variable, "numeric (standardised)", "one-hot")
    df["odds_ratio"] = np.exp(df.coef)
    return df.reindex(df.coef.abs().sort_values(ascending=False).index).reset_index(drop=True)


def tree_table() -> pd.DataFrame:
    rows = []
    for m in TREES:
        pipe = joblib.load(MODEL_DIR / f"{m}.joblib")
        names = feature_names(pipe)
        imp = pipe.named_steps["clf"].feature_importances_
        df = pd.DataFrame({"feature": names, "importance": imp})
        df["variable"] = df.feature.map(raw_variable)
        agg = df.groupby("variable", sort=False).importance.sum()
        rows.append(agg.rename(m))
    out = pd.concat(rows, axis=1).fillna(0.0)
    out.index.name = "variable"
    return out.sort_values("random_forest", ascending=False)


def display_name(name: str) -> str:
    """'PAY_0_-2' → 'PAY_0 = −2'，数值列原样。"""
    m = re.match(r"^(SEX|EDUCATION|MARRIAGE|PAY_\d)_(-?\d+)$", name)
    return f"{m.group(1)} = {m.group(2).replace('-', '−')}" if m else name


def plot_logistic(df: pd.DataFrame, top: int = 15):
    d = df.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(FULL[0], 3.4))
    colors = np.where(d.coef > 0, COLORS["adaboost"], COLORS["random_forest"])
    ax.barh(d.feature.map(display_name), d.coef, color=colors, height=0.7)
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Coefficient (log-odds); orange raises default risk, blue lowers it")
    ax.set_title(f"Logistic Regression: {top} largest coefficients (all one-hot categories kept, L2-penalised)")
    ax.grid(axis="y", visible=False)
    save(fig, C.FIG_DIR / "fig_feature_logistic")


def plot_trees(table: pd.DataFrame, top: int = 12):
    fig, axes = plt.subplots(1, 3, figsize=(FULL[0], 3.0), sharey=False)
    for ax, m in zip(axes, TREES):
        d = table[m].sort_values(ascending=False).head(top).iloc[::-1]
        ax.barh(d.index, d.values, color=COLORS[m], height=0.7)
        ax.set_title(LABELS[m])
        ax.set_xlabel("Impurity importance")
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle("Feature importance by original variable (one-hot columns summed)", y=1.02)
    save(fig, C.FIG_DIR / "fig_feature_importance")


def main():
    apply()
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    lg = logistic_table()
    lg.to_csv(C.TABLE_DIR / "feature_logistic_coefficients.csv", index=False)
    tr = tree_table()
    tr.to_csv(C.TABLE_DIR / "feature_importance_trees.csv")
    plot_logistic(lg)
    plot_trees(tr)

    pd.set_option("display.width", 200)
    print("Logistic: 15 largest |coef|")
    print(lg.head(15)[["feature", "coef", "odds_ratio"]].round(3).to_string(index=False))
    print("\nTree importances by variable (top 12 by random forest)")
    print(tr.head(12).round(3))


if __name__ == "__main__":
    main()
