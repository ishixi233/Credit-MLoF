"""在共享的五折上做交叉验证，产出折外预测，并按平均 AP 选配置。"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone

from src import config as C
from src.evaluation.metrics import ranking_metrics


def positive_score(model, X) -> np.ndarray:
    """取 y=1 那一列的概率，不依赖 classes_ 的排列顺序。"""
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def cross_validate(pipeline, X, y, fold):
    """每折用其余四折拟合完整 pipeline（含预处理），给留出的一折打分。

    返回折外分数（与 X 行对齐）和每折的 AP、ROC-AUC、Brier、拟合耗时。
    """
    oof = np.full(len(y), np.nan)
    rows = []
    for k in sorted(fold.unique()):
        val = (fold == k).to_numpy()
        start = time.perf_counter()
        model = clone(pipeline).fit(X[~val], y[~val])
        seconds = time.perf_counter() - start
        oof[val] = positive_score(model, X[val])
        m = ranking_metrics(y[val], oof[val])
        rows.append({"fold": k, "ap": m["ap"], "roc_auc": m["roc_auc"], "brier": m["brier"], "fit_seconds": seconds})
    assert not np.isnan(oof).any()
    return oof, pd.DataFrame(rows)


def run_model(name: str, build, grid: list[dict], X, y, fold, ids) -> dict:
    """对 grid 中每套参数做五折，按平均 AP 选一套；并列时取 grid 中靠前的（应把较简单的放前面）。

    写出三个文件：
      results/experiments/cv_<name>.csv        每套参数、每折的指标
      results/experiments/selected_<name>.json 选定参数及其五折汇总
      results/predictions/oof_<name>.csv       选定参数的折外预测，列为 ID, fold, y, score
    """
    records, best = [], None
    for i, params in enumerate(grid):
        oof, folds = cross_validate(build(**params), X, y, fold)
        folds.insert(0, "params", json.dumps(params))
        folds.insert(0, "config", i)
        records.append(folds)
        mean_ap = folds["ap"].mean()
        if best is None or mean_ap > best["mean_ap"]:
            best = {"config": i, "params": params, "mean_ap": mean_ap, "oof": oof, "folds": folds}

    for d in (C.EXP_DIR, C.PRED_DIR):
        d.mkdir(parents=True, exist_ok=True)
    pd.concat(records).to_csv(C.EXP_DIR / f"cv_{name}.csv", index=False)
    pd.DataFrame({C.ID_COL: ids, "fold": fold, "y": y, "score": best["oof"]}).to_csv(
        C.PRED_DIR / f"oof_{name}.csv", index=False
    )

    f = best["folds"]
    pooled = ranking_metrics(y, best["oof"])
    summary = {
        "model": name,
        "params": best["params"],
        "n_configs": len(grid),
        "cv_ap_mean": f["ap"].mean(),
        "cv_ap_std": f["ap"].std(ddof=1),
        "cv_roc_auc_mean": f["roc_auc"].mean(),
        "cv_roc_auc_std": f["roc_auc"].std(ddof=1),
        "cv_brier_mean": f["brier"].mean(),
        "oof_brier_skill": pooled["brier_skill"],
        "fit_seconds_mean": f["fit_seconds"].mean(),
    }
    (C.EXP_DIR / f"selected_{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def collect_summary() -> pd.DataFrame:
    """把所有已跑模型的 selected_*.json 汇总成一张表，写入 results/tables/cv_summary.csv。"""
    rows = [json.loads(p.read_text()) for p in sorted(C.EXP_DIR.glob("selected_*.json"))]
    table = pd.DataFrame(rows).sort_values("cv_ap_mean", ascending=False)
    table["params"] = table["params"].map(json.dumps)
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.TABLE_DIR / "cv_summary.csv", index=False)
    return table
