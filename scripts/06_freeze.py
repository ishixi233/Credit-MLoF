"""写冻结清单 configs/frozen.json。

把每个模型由交叉验证选定的参数（results/experiments/selected_*.json）和折外预测上选出的
各成本比阈值（results/tables/oof_thresholds.csv）汇总到一个文件，并记录生成时的 git commit。

规则：
  - 清单提交进仓库之后，才允许运行 scripts/07_run_test.py。
  - 清单一旦提交，不再根据任何测试结果修改。若发现错误必须修改，要在 PR 中写明原因。
运行：
    python scripts/06_freeze.py
"""
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config as C

MODEL_NAMES = ("logistic", "gaussian_nb", "knn", "decision_tree", "random_forest", "adaboost")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=C.ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main():
    if C.FROZEN_FILE.exists():
        sys.exit(f"{C.FROZEN_FILE.relative_to(C.ROOT)} 已存在。冻结清单不应重写；确需修改请先删除并在 PR 中说明原因。")
    dirty = git("status", "--porcelain", "--", "src", "scripts", "results/predictions", "results/experiments", "results/tables")
    if dirty:
        sys.exit("工作区有未提交的改动，先提交再冻结：\n" + dirty)

    thresholds = pd.read_csv(C.TABLE_DIR / "oof_thresholds.csv")
    models = {}
    for m in MODEL_NAMES:
        sel = json.loads((C.EXP_DIR / f"selected_{m}.json").read_text())
        t = thresholds[thresholds.model == m].set_index("cost_ratio")
        assert list(t.index) == list(C.COST_RATIOS), m
        models[m] = {
            "params": sel["params"],
            "cv_ap_mean": round(sel["cv_ap_mean"], 6),
            "thresholds": {str(r): float(t.loc[r, "threshold"]) for r in C.COST_RATIOS},
            "threshold_case": {str(r): t.loc[r, "threshold_case"] for r in C.COST_RATIOS},
        }

    frozen = {
        "frozen_on": date.today().isoformat(),
        "git_commit": git("rev-parse", "HEAD"),
        "seed": C.SEED,
        "splits_file": str(C.SPLITS_FILE.relative_to(C.ROOT)),
        "cost_ratios": list(C.COST_RATIOS),
        "primary_ratio": C.PRIMARY_RATIO,
        "default_threshold": 0.5,
        "selection_rule": "mean 5-fold AP selects params; OOF cost r*FN+FP selects threshold per ratio; ties favour fewer false alarms",
        "models": models,
    }
    C.FROZEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    C.FROZEN_FILE.write_text(json.dumps(frozen, indent=2, ensure_ascii=False) + "\n")
    print(f"已写入 {C.FROZEN_FILE.relative_to(C.ROOT)}（commit {frozen['git_commit'][:7]}）。")
    print("下一步：全组核对 → 提交并合并 → 运行 scripts/07_run_test.py，仅一次。")


if __name__ == "__main__":
    main()
