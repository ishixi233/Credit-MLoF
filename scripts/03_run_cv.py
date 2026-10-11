"""在共享五折上跑登记表里的模型，保存折外预测并更新汇总表。只使用 70% 开发集。

用法：
    python scripts/03_run_cv.py logistic gaussian_nb    # 跑指定模型
    python scripts/03_run_cv.py                         # 跑登记表里的全部模型
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.load import load_dev
from src.evaluation.cv import collect_summary, run_model
from src.models.registry import MODELS

names = sys.argv[1:] or list(MODELS)
unknown = [n for n in names if n not in MODELS]
if unknown:
    sys.exit(f"未登记的模型：{unknown}；可用：{list(MODELS)}")

X, y, fold, ids = load_dev()
for name in names:
    s = run_model(name, MODELS[name]["build"], MODELS[name]["grid"], X, y, fold, ids)
    print(f"{name:15s} AP {s['cv_ap_mean']:.4f} ± {s['cv_ap_std']:.4f}  params {s['params']}")

cols = ["model", "cv_ap_mean", "cv_ap_std", "cv_roc_auc_mean", "cv_brier_mean", "oof_brier_skill"]
print(collect_summary()[cols].round(4).to_string(index=False))
