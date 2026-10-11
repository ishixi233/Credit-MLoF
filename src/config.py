"""全组共用的路径和实验约定。改这里的任何数值前先在群里说一声。"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "default of credit card clients.xls"
RAW_SHA256 = "30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933"
SPLITS_FILE = ROOT / "data" / "splits" / "splits.csv"
FROZEN_FILE = ROOT / "configs" / "frozen.json"

PRED_DIR = ROOT / "results" / "predictions"
EXP_DIR = ROOT / "results" / "experiments"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

SEED = 42
TEST_SIZE = 0.30
N_FOLDS = 5

# 漏报相对误报的成本比 r：C_r(t) = r*FN + FP。5 是事先声明的主情景，其余为敏感性情景。
COST_RATIOS = (1, 2, 3, 5, 10)
PRIMARY_RATIO = 5

ID_COL = "ID"
TARGET_RAW = "default payment next month"
TARGET = "y"

CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
NUM_COLS = ["LIMIT_BAL", "AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS
FEATURES = ["LIMIT_BAL"] + CAT_COLS + ["AGE"] + PAY_COLS + BILL_COLS + PAY_AMT_COLS
