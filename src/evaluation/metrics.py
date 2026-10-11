"""全组共用的评价函数。约定：y=1 为违约；score 为 P(y=1)；score >= t 时报警。"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def ranking_metrics(y, score) -> dict:
    """基于连续分数的指标：AP、ROC-AUC、Brier。

    brier_ref 是对所有人都输出该评价集违约率时的 Brier；brier_skill = 1 - brier / brier_ref，
    小于 0 说明概率还不如基础违约率。
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    rate = y.mean()
    brier = brier_score_loss(y, score)
    ref = rate * (1 - rate)
    return {
        "ap": average_precision_score(y, score),
        "roc_auc": roc_auc_score(y, score),
        "brier": brier,
        "brier_ref": ref,
        "brier_skill": 1 - brier / ref,
    }


def decision_metrics(y, score, threshold: float, cost_ratio: float) -> dict:
    """给定阈值下的混淆矩阵和派生指标。没有任何报警时 precision 记为 NaN。"""
    y = np.asarray(y).astype(int)
    flag = np.asarray(score, dtype=float) >= threshold
    tp = int((flag & (y == 1)).sum())
    fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum())
    tn = int((~flag & (y == 0)).sum())
    n = len(y)
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else np.nan,
        "recall": tp / (tp + fn) if tp + fn else np.nan,
        "error_rate": (fp + fn) / n,
        "flag_rate": (tp + fp) / n,
        "cost_per_client": (cost_ratio * fn + fp) / n,
    }


def cost_curve(y, score, cost_ratio: float) -> pd.DataFrame:
    """遍历分数能产生的全部不同决策，从"全不报警"到"全部报警"，报警人数逐行增加。

    每个决策对应一段阈值区间，这里取区间中点作为代表，迁移到新数据时不偏向任何一端。
    "全不报警"的阈值记为 +inf，"全部报警"记为 -inf。
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    order = np.argsort(-score, kind="mergesort")
    s, ys = score[order], y[order]
    last = np.r_[np.diff(s) != 0, True]
    values = s[last]
    tp = np.r_[0, np.cumsum(ys)[last]]
    fp = np.r_[0, np.cumsum(1 - ys)[last]]
    lower = np.r_[values[1:], -np.inf]
    threshold = np.r_[np.inf, (values + lower) / 2]
    fn = y.sum() - tp
    return pd.DataFrame(
        {
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "cost_per_client": (cost_ratio * fn + fp) / len(y),
        }
    )


def select_threshold(y, score, cost_ratio: float) -> float:
    """选出使 C_r(t) = r*FN + FP 最小的阈值；成本相同时取误报较少的那个。"""
    curve = cost_curve(y, score, cost_ratio)
    return float(curve.loc[curve["cost_per_client"].idxmin(), "threshold"])


def trivial_costs(y, cost_ratio: float) -> dict:
    """两种不需要模型的策略的每客户成本，作为模型成本的参照。"""
    rate = float(np.mean(y))
    none, everyone = cost_ratio * rate, 1 - rate
    return {"flag_none": none, "flag_all": everyone, "best_trivial": min(none, everyone)}


def calibration_table(y, score, n_bins: int = 10) -> pd.DataFrame:
    """等频分箱的校准表：每箱的平均预测概率、实际违约率和样本数。分数取值很少时箱数会变少。"""
    df = pd.DataFrame({"y": np.asarray(y).astype(int), "score": np.asarray(score, dtype=float)})
    df["bin"] = pd.qcut(df["score"], n_bins, duplicates="drop")
    out = df.groupby("bin", observed=True).agg(
        mean_score=("score", "mean"), default_rate=("y", "mean"), n=("y", "size")
    )
    return out.reset_index(drop=True)
