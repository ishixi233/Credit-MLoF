"""最终测试：按冻结清单在完整开发集上重新拟合每个模型，对 30% 测试集打分并评价。只运行一次。

前提：configs/frozen.json 已提交并合并。脚本会核对清单里的 commit 是本分支历史的一部分。
运行：
    python scripts/07_run_test.py
输出：
    results/predictions/test_<model>.csv      列为 ID, y, score
    results/tables/test_ranking.csv           每模型：测试 AP、ROC-AUC、Brier、skill；并列 CV AP 作对照
    results/tables/test_decisions.csv         每模型 × 每成本比：冻结阈值与默认阈值 0.5 下的混淆矩阵、成本、
                                              相对平凡策略的改善；另有 post-hoc 最优成本（只作诊断，不用于选择）
    results/tables/test_cost_matrix.csv       模型 × 成本比的每客户成本，附参照策略
    results/tables/test_rank_comparison.csv   AP 名次 vs 各成本比下的成本名次
    results/tables/test_bootstrap.csv         成对分层 bootstrap：各模型 AP、主情景成本的 95% 区间，
                                              以及与 AP 最优模型之差的区间
    models/<model>.joblib                     冻结配置在完整开发集上拟合的 pipeline
"""
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src import config as C
from src.data.load import load_dev, load_test
from src.evaluation.cv import positive_score
from src.evaluation.metrics import decision_metrics, ranking_metrics, select_threshold, trivial_costs
from src.models.registry import MODELS

N_BOOT = 2000
MODEL_DIR = C.ROOT / "models"


def check_frozen() -> dict:
    frozen = json.loads(C.FROZEN_FILE.read_text())
    r = subprocess.run(["git", "merge-base", "--is-ancestor", frozen["git_commit"], "HEAD"], cwd=C.ROOT)
    if r.returncode != 0:
        sys.exit("冻结清单中的 commit 不在当前分支历史里，先合并冻结清单再运行。")
    if (C.PRED_DIR / "test_logistic.csv").exists():
        sys.exit("测试预测已存在。测试只运行一次；若确需重跑（例如修复脚本错误），先在 PR 中说明并删除旧结果。")
    return frozen


def fit_and_score(frozen, X_dev, y_dev, X_test):
    scores, fitted = {}, {}
    for m, spec in frozen["models"].items():
        start = time.perf_counter()
        pipe = MODELS[m]["build"](**spec["params"]).fit(X_dev, y_dev)
        fitted[m] = pipe
        scores[m] = positive_score(pipe, X_test)
        print(f"{m:15s} fitted on {len(y_dev):,} clients in {time.perf_counter() - start:.1f}s")
    return scores, fitted


def bootstrap(y, scores: dict, best_ap_model: str, thresholds: dict, r: int, seed: int):
    """成对分层 bootstrap：每次按类别重抽样同一组索引，所有模型用同一份重抽样数据。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    names = list(scores)
    ap = {m: np.empty(N_BOOT) for m in names}
    cost = {m: np.empty(N_BOOT) for m in names}
    for b in range(N_BOOT):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        yb = y[idx]
        for m in names:
            s = scores[m][idx]
            ap[m][b] = average_precision_score(yb, s)
            flag = s >= thresholds[m]
            cost[m][b] = (r * ((~flag) & (yb == 1)).sum() + (flag & (yb == 0)).sum()) / len(yb)
    rows = []
    for m in names:
        d_ap = ap[m] - ap[best_ap_model]
        d_cost = cost[m] - cost[best_ap_model]
        rows.append(
            {
                "model": m,
                "ap_lo": np.percentile(ap[m], 2.5),
                "ap_hi": np.percentile(ap[m], 97.5),
                f"cost_r{r}_lo": np.percentile(cost[m], 2.5),
                f"cost_r{r}_hi": np.percentile(cost[m], 97.5),
                "reference_model": best_ap_model,
                "delta_ap_lo": np.percentile(d_ap, 2.5),
                "delta_ap_hi": np.percentile(d_ap, 97.5),
                f"delta_cost_r{r}_lo": np.percentile(d_cost, 2.5),
                f"delta_cost_r{r}_hi": np.percentile(d_cost, 97.5),
            }
        )
    return pd.DataFrame(rows)


def main():
    frozen = check_frozen()
    X_dev, y_dev, _, _ = load_dev()
    X_test, y_test, ids = load_test()
    print(f"test set: {len(y_test):,} clients, default rate {y_test.mean():.4f}")

    scores, fitted = fit_and_score(frozen, X_dev, y_dev, X_test)
    MODEL_DIR.mkdir(exist_ok=True)
    for m, pipe in fitted.items():
        joblib.dump(pipe, MODEL_DIR / f"{m}.joblib")
        pd.DataFrame({C.ID_COL: ids, "y": y_test, "score": scores[m]}).to_csv(C.PRED_DIR / f"test_{m}.csv", index=False)

    # 排序指标
    ranking = []
    for m, s in scores.items():
        ranking.append({"model": m, "cv_ap_mean": frozen["models"][m]["cv_ap_mean"], **ranking_metrics(y_test, s)})
    ranking = pd.DataFrame(ranking).sort_values("ap", ascending=False)
    ranking["ap_rank"] = range(1, len(ranking) + 1)
    ranking.to_csv(C.TABLE_DIR / "test_ranking.csv", index=False)

    # 决策指标
    rows, refs = [], {}
    for r in C.COST_RATIOS:
        ref = trivial_costs(y_test, r)
        refs[r] = ref
        for m, s in scores.items():
            t = frozen["models"][m]["thresholds"][str(r)]
            fro = decision_metrics(y_test, s, t, r)
            dft = decision_metrics(y_test, s, 0.5, r)
            post = decision_metrics(y_test, s, select_threshold(y_test, s, r), r)  # 仅诊断
            rows.append(
                {
                    "model": m,
                    "cost_ratio": r,
                    "frozen_threshold": t,
                    **{f"frozen_{k}": fro[k] for k in ("tp", "fp", "fn", "tn", "precision", "recall", "error_rate", "flag_rate", "cost_per_client")},
                    **{f"default_{k}": dft[k] for k in ("fp", "fn", "precision", "recall", "cost_per_client")},
                    "best_trivial_cost": ref["best_trivial"],
                    "improvement_vs_trivial_pct": 100 * (ref["best_trivial"] - fro["cost_per_client"]) / ref["best_trivial"],
                    "posthoc_oracle_cost": post["cost_per_client"],
                    "regret_vs_oracle": fro["cost_per_client"] - post["cost_per_client"],
                }
            )
    decisions = pd.DataFrame(rows)
    decisions.to_csv(C.TABLE_DIR / "test_decisions.csv", index=False)

    matrix = decisions.pivot(index="model", columns="cost_ratio", values="frozen_cost_per_client")
    matrix = matrix.reindex(ranking.model)
    for key in ("flag_none", "flag_all", "best_trivial"):
        matrix.loc[key] = [refs[r][key] for r in C.COST_RATIOS]
    matrix.index.name = "model_or_reference"
    matrix.to_csv(C.TABLE_DIR / "test_cost_matrix.csv")

    # AP 名次 vs 成本名次
    comp = ranking[["model", "ap", "ap_rank"]].set_index("model")
    for r in C.COST_RATIOS:
        c = decisions[decisions.cost_ratio == r].set_index("model")["frozen_cost_per_client"]
        comp[f"cost_rank_r{r}"] = c.rank(method="min").astype(int)
    comp.to_csv(C.TABLE_DIR / "test_rank_comparison.csv")

    # bootstrap
    best = ranking.iloc[0]["model"]
    thr = {m: frozen["models"][m]["thresholds"][str(C.PRIMARY_RATIO)] for m in scores}
    boot = bootstrap(y_test, scores, best, thr, C.PRIMARY_RATIO, C.SEED)
    boot.to_csv(C.TABLE_DIR / "test_bootstrap.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nTest ranking metrics:")
    print(ranking[["model", "cv_ap_mean", "ap", "roc_auc", "brier", "brier_skill"]].round(4).to_string(index=False))
    print("\nTest cost per client at frozen thresholds:")
    print(matrix.round(4))
    print("\nAP rank vs cost rank:")
    print(comp)
    print(f"\nBootstrap 95% intervals (n={N_BOOT}, reference = {best}):")
    print(boot.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
