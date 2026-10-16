"""测试集的补充诊断：五档成本比的成对 bootstrap 区间，以及两个基准的测试表现。

只读已提交的 results/predictions/test_*.csv，不重新拟合，不改任何冻结配置。运行：
    python scripts/10_test_uncertainty.py
输出：
    results/tables/test_bootstrap_all_ratios.csv   每模型 × 每成本比：成本的 95% 区间，与参照模型（测试 AP 最高者）
                                                   的 Δ成本区间；AP 区间与 ΔAP 区间
    results/tables/test_baselines.csv              多数类预测与常数分数基准在测试集上的指标
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import config as C
from src.evaluation.metrics import decision_metrics, ranking_metrics, trivial_costs

N_BOOT = 2000
MODEL_NAMES = ("random_forest", "adaboost", "logistic", "knn", "decision_tree", "gaussian_nb")


def main():
    frozen = json.loads(C.FROZEN_FILE.read_text())
    preds = {m: pd.read_csv(C.PRED_DIR / f"test_{m}.csv") for m in MODEL_NAMES}
    y = preds["random_forest"].y.to_numpy()
    scores = {m: d.score.to_numpy() for m, d in preds.items()}
    ref_model = pd.read_csv(C.TABLE_DIR / "test_ranking.csv").iloc[0]["model"]

    rng = np.random.default_rng(C.SEED)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    ap = {m: np.empty(N_BOOT) for m in MODEL_NAMES}
    cost = {(m, r): np.empty(N_BOOT) for m in MODEL_NAMES for r in C.COST_RATIOS}
    for b in range(N_BOOT):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        yb = y[idx]
        for m in MODEL_NAMES:
            s = scores[m][idx]
            ap[m][b] = average_precision_score(yb, s)
            for r in C.COST_RATIOS:
                flag = s >= frozen["models"][m]["thresholds"][str(r)]
                cost[(m, r)][b] = (r * ((~flag) & (yb == 1)).sum() + (flag & (yb == 0)).sum()) / len(yb)

    q = lambda a: (np.percentile(a, 2.5), np.percentile(a, 97.5))
    rows = []
    for m in MODEL_NAMES:
        for r in C.COST_RATIOS:
            lo, hi = q(cost[(m, r)])
            dlo, dhi = q(cost[(m, r)] - cost[(ref_model, r)])
            alo, ahi = q(ap[m])
            dalo, dahi = q(ap[m] - ap[ref_model])
            rows.append(
                {
                    "model": m, "cost_ratio": r,
                    "cost_point": decision_metrics(y, scores[m], frozen["models"][m]["thresholds"][str(r)], r)["cost_per_client"],
                    "cost_lo": lo, "cost_hi": hi,
                    "reference_model": ref_model, "delta_cost_lo": dlo, "delta_cost_hi": dhi,
                    "delta_cost_excludes_zero": (dlo > 0) or (dhi < 0),
                    "ap_lo": alo, "ap_hi": ahi, "delta_ap_lo": dalo, "delta_ap_hi": dahi,
                }
            )
    boot = pd.DataFrame(rows)
    boot.to_csv(C.TABLE_DIR / "test_bootstrap_all_ratios.csv", index=False)

    # 基准：多数类预测（永不报警）和常数分数（= 开发集违约率）
    rate = float(pd.read_csv(C.PRED_DIR / "oof_baseline_rate.csv").y.mean())
    const = np.full(len(y), rate)
    base_rows = []
    for r in C.COST_RATIOS:
        none = decision_metrics(y, const, np.inf, r)
        base_rows.append({"baseline": "majority_class_never_flag", "cost_ratio": r, **{k: none[k] for k in ("precision", "recall", "error_rate", "cost_per_client")}, **trivial_costs(y, r)})
    base = pd.DataFrame(base_rows)
    rk = ranking_metrics(y, const)
    base["constant_score_ap"], base["constant_score_roc_auc"], base["constant_score_brier"] = rk["ap"], rk["roc_auc"], rk["brier"]
    base.to_csv(C.TABLE_DIR / "test_baselines.csv", index=False)

    pd.set_option("display.width", 220)
    print(f"Paired stratified bootstrap, n={N_BOOT}, reference = {ref_model}\n")
    print(boot.pivot(index="model", columns="cost_ratio", values="delta_cost_excludes_zero"))
    print()
    print(boot[["model", "cost_ratio", "cost_point", "cost_lo", "cost_hi", "delta_cost_lo", "delta_cost_hi"]].round(4).to_string(index=False))
    print("\nBaselines on the test set:")
    print(base.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
