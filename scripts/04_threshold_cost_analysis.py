"""OOF threshold optimization and cost-ratio sensitivity for Credit-MLoF.

Run from the repository root (do not access the held-out test set):
    python scripts/04_threshold_cost_analysis.py

Reuses the team's src.config and src.evaluation.metrics functions. The five
cost ratios and primary ratio are imported from src.config; do not change them
without agreeing with the team.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/04_threshold_cost_analysis.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")  # Save figures without requiring a graphical window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config as C
from src.evaluation.metrics import (
    cost_curve,
    decision_metrics,
    ranking_metrics,
    select_threshold,
    trivial_costs,
)

MODEL_NAMES = (
    "logistic", "gaussian_nb", "knn", "decision_tree", "random_forest", "adaboost"
)
ALL_NAMES = ("baseline_rate",) + MODEL_NAMES
REQUIRED_COLUMNS = ("ID", "fold", "y", "score")


def read_oof(name: str, reference: pd.DataFrame | None):
    path = C.PRED_DIR / f"oof_{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Switch GitHub Desktop to main, Pull origin, "
            "then create your threshold-analysis branch."
        )
    d = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS).difference(d.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if d["ID"].isna().any() or d["ID"].duplicated().any():
        raise ValueError(f"{path}: ID is missing or duplicated")
    if d[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError(f"{path}: null values in required columns")
    d = d[list(REQUIRED_COLUMNS)].copy().sort_values("ID").reset_index(drop=True)
    for col in ("fold", "y", "score"):
        d[col] = pd.to_numeric(d[col], errors="raise")
    if not d["y"].isin([0, 1]).all():
        raise ValueError(f"{path}: y must be 0 or 1")
    if not d["fold"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError(f"{path}: fold must be 0..4")
    if not np.isfinite(d["score"].to_numpy()).all():
        raise ValueError(f"{path}: score has NaN or infinity")
    if not d["score"].between(0, 1).all():
        raise ValueError(f"{path}: score must be a probability in [0,1]")
    if reference is not None:
        if len(d) != len(reference) or not d[["ID", "fold", "y"]].equals(reference[["ID", "fold", "y"]]):
            raise ValueError(f"{path}: ID/fold/y do not align with baseline_rate")
    return d


def finite_threshold(raw: float) -> tuple[float, str]:
    """Make +/-inf 'all/no alerts' decisions safe for standard JSON freezing."""
    if raw == float("inf"):
        return float(np.nextafter(1.0, np.inf)), "none_alerted"
    if raw == float("-inf"):
        return 0.0, "all_alerted"
    if not np.isfinite(raw):
        raise ValueError(f"Unexpected nonfinite threshold: {raw}")
    return float(raw), "interior"


def plot_cost_vs_ratio(costs: pd.DataFrame, refs: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in MODEL_NAMES:
        sub = costs[costs.model == name].sort_values("cost_ratio")
        ax.plot(sub.cost_ratio, sub.optimized_cost_per_client,
                marker="o", linewidth=1.8, label=name)
    ax.plot(refs.cost_ratio, refs.flag_none, "k--", label="Flag none")
    ax.plot(refs.cost_ratio, refs.flag_all, "k:", label="Flag all")
    ax.set(xlabel="False-negative / false-positive cost ratio (r)",
           ylabel="OOF cost per client (relative units)",
           title="Cost sensitivity across models (OOF, training/development set)")
    ax.set_xticks(list(C.COST_RATIOS))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "oof_cost_vs_ratio.png", dpi=180)
    plt.close(fig)


def plot_threshold_vs_ratio(thresholds: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in MODEL_NAMES:
        sub = thresholds[thresholds.model == name].sort_values("cost_ratio")
        y = sub.threshold.to_numpy(dtype=float)
        y = np.where(y > 1.0, np.nan, y)  # 'No alerts' cannot be plotted in [0,1].
        ax.plot(sub.cost_ratio, y, marker="o", label=name)
        for r, t, action in zip(sub.cost_ratio, sub.threshold, sub.threshold_case):
            if action == "none_alerted":
                ax.annotate("none", (r, 0.97), fontsize=7, ha="center")
    ax.plot(list(C.COST_RATIOS), [1 / (1 + r) for r in C.COST_RATIOS],
            "k--", label="1/(1+r), if calibrated")
    ax.set(xlabel="False-negative / false-positive cost ratio (r)",
           ylabel="OOF-selected alert threshold",
           title="Decision thresholds by cost assumption (OOF)", ylim=(-0.03, 1.03))
    ax.set_xticks(list(C.COST_RATIOS))
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "oof_thresholds_vs_ratio.png", dpi=180)
    plt.close(fig)


def plot_primary_cost_curves(datasets: dict[str, pd.DataFrame], thresholds: pd.DataFrame):
    r = C.PRIMARY_RATIO
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in MODEL_NAMES:
        d = datasets[name]
        cv = cost_curve(d.y, d.score, r)
        cv = cv[np.isfinite(cv.threshold)].sort_values("threshold")
        if cv.empty:
            continue
        # Downsample only for drawing; optimization was on the entire curve.
        if len(cv) > 2500:
            indices = np.unique(np.linspace(0, len(cv) - 1, 2500).astype(int))
            cv = cv.iloc[indices]
        line, = ax.plot(cv.threshold, cv.cost_per_client, lw=1.2, alpha=0.85, label=name)
        hit = thresholds[(thresholds.model == name) & (thresholds.cost_ratio == r)].iloc[0]
        if 0 <= hit.threshold <= 1:
            ax.scatter([hit.threshold], [hit.optimized_cost_per_client],
                       color=line.get_color(), s=25, zorder=3)
    ax.set(xlabel="Classification threshold (score >= t means alert)",
           ylabel="OOF cost per client (relative units)",
           title=f"Threshold-cost curves at primary ratio {r}:1 (OOF)", xlim=(0, 1))
    ax.axvline(0.5, color="gray", ls="--", alpha=0.5, lw=1, label="Default threshold 0.5")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIG_DIR / "oof_cost_curve_r5.png", dpi=180)
    plt.close(fig)


def main():
    if list(C.COST_RATIOS) != [1, 2, 3, 5, 10]:
        raise ValueError("Unexpected COST_RATIOS; confirm with the team before running")
    if C.PRIMARY_RATIO != 5:
        raise ValueError("Unexpected PRIMARY_RATIO; confirm with the team")
    C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Read and rigorously align the team's already-produced OOF files.
    datasets = {}
    reference = None
    check_rows = []
    for name in ALL_NAMES:
        d = read_oof(name, reference)
        if reference is None:
            reference = d
        datasets[name] = d
        check_rows.append({
            "model": name, "rows": len(d), "unique_id": d.ID.nunique(),
            "n_defaults": int(d.y.sum()), "default_rate": float(d.y.mean()),
            "folds": len(d.fold.unique()), "score_min": float(d.score.min()),
            "score_max": float(d.score.max()), "aligned": True,
        })
    alignment = pd.DataFrame(check_rows)
    alignment.to_csv(C.TABLE_DIR / "oof_alignment_check.csv", index=False)
    print(f"Validated {len(datasets)} OOF files; {len(reference):,} development clients. ")
    if len(reference) != 20997:
        print("WARNING: expected 20,997 rows from the 2026-10-15 plan; verify the split version.")

    threshold_rows, refs_rows = [], []
    for r in C.COST_RATIOS:
        ref = trivial_costs(reference.y, r)
        refs_rows.append({"cost_ratio": r, **ref})
        for name, d in datasets.items():
            raw = select_threshold(d.y, d.score, r)
            threshold, kind = finite_threshold(raw)
            best = decision_metrics(d.y, d.score, threshold, r)
            default = decision_metrics(d.y, d.score, 0.5, r)
            original = decision_metrics(d.y, d.score, raw, r)
            assert best["tp"] == original["tp"] and best["fp"] == original["fp"]
            assert best["fn"] == original["fn"] and best["tn"] == original["tn"]
            if best["cost_per_client"] > default["cost_per_client"] + 1e-12:
                raise AssertionError(f"{name}, r={r}: optimized worse than default")
            threshold_rows.append({
                "model": name, "cost_ratio": r, "threshold": threshold,
                "threshold_case": kind, "theoretical_if_calibrated": 1 / (1 + r),
                "tp": best["tp"], "fp": best["fp"], "fn": best["fn"], "tn": best["tn"],
                "precision": best["precision"], "recall": best["recall"],
                "flag_rate": best["flag_rate"],
                "optimized_cost_per_client": best["cost_per_client"],
                "default_t": 0.5, "default_fp": default["fp"], "default_fn": default["fn"],
                "default_cost_per_client": default["cost_per_client"],
                "best_trivial_cost": ref["best_trivial"],
                "improvement_vs_trivial_pct":
                    100 * (ref["best_trivial"] - best["cost_per_client"]) / ref["best_trivial"],
                "improvement_vs_default_pct":
                    100 * (default["cost_per_client"] - best["cost_per_client"]) /
                    default["cost_per_client"] if default["cost_per_client"] > 0 else np.nan,
            })

    thresholds = pd.DataFrame(threshold_rows)
    refs = pd.DataFrame(refs_rows)
    costs = thresholds[[
        "model", "cost_ratio", "threshold", "optimized_cost_per_client",
        "default_cost_per_client", "best_trivial_cost",
        "improvement_vs_trivial_pct", "improvement_vs_default_pct",
    ]].copy()
    # Winner only among the six actual classifiers, never the baseline.
    valid = costs[costs.model.isin(MODEL_NAMES)]
    winners = valid.loc[valid.groupby("cost_ratio").optimized_cost_per_client.idxmin(),
                        ["cost_ratio", "model"]].set_index("cost_ratio").model.to_dict()
    costs["best_classifier_at_ratio"] = costs.apply(
        lambda row: row.model == winners[row.cost_ratio], axis=1
    )
    costs["beats_best_trivial"] = costs.optimized_cost_per_client < costs.best_trivial_cost

    matrix = costs.pivot(index="model", columns="cost_ratio", values="optimized_cost_per_client")
    matrix = matrix.reindex(list(MODEL_NAMES) + ["baseline_rate"])
    matrix.loc["flag_none"] = refs.set_index("cost_ratio").flag_none
    matrix.loc["flag_all"] = refs.set_index("cost_ratio").flag_all
    matrix.loc["best_trivial"] = refs.set_index("cost_ratio").best_trivial
    matrix.index.name = "model_or_reference"

    thresholds.to_csv(C.TABLE_DIR / "oof_thresholds.csv", index=False)
    costs.to_csv(C.TABLE_DIR / "oof_cost_comparison.csv", index=False)
    matrix.to_csv(C.TABLE_DIR / "oof_cost_matrix.csv")
    plot_cost_vs_ratio(costs, refs)
    plot_threshold_vs_ratio(thresholds)
    plot_primary_cost_curves(datasets, thresholds)

    print("\nPrimary scenario r=5 (OOF only, NOT held-out test):")
    cols = ["model", "threshold", "fp", "fn", "optimized_cost_per_client",
            "default_cost_per_client", "best_trivial_cost"]
    print(thresholds[thresholds.cost_ratio == 5][cols].round(4).to_string(index=False))
    print("\nBest classifier by cost ratio (OOF):")
    print(pd.DataFrame({"ratio": list(winners), "best_classifier": list(winners.values())}).to_string(index=False))
    print("\nSaved 4 CSV tables in results/tables/ and 3 PNG figures in results/figures/.")
    print("IMPORTANT: Do not interpret OOF-optimized costs as unbiased test performance.")
    print("Send oof_thresholds.csv to Li Erteng for the frozen configuration; do NOT load test data here.")


if __name__ == "__main__":
    main()
