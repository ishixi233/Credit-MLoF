"""原始数据的结构核查，结果写入 docs/data_check.md。只看结构和编码，不看特征与标签的关系。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src.data.load import load_clean, load_raw

raw = load_raw()
clean = load_clean()
y = raw[C.TARGET_RAW]

lines = [
    "# 数据核查记录",
    "",
    "由 `scripts/01_check_data.py` 自动生成，不要手工修改。",
    "",
    f"- 文件：`{C.RAW_FILE.name}`，SHA-256 `{C.RAW_SHA256}`",
    f"- 行数 {len(raw):,}，列数 {raw.shape[1]}（ID + 23 个特征 + 标签），全部为整数列",
    f"- 缺失值 {int(raw.isna().sum().sum())} 个；重复 ID {int(raw[C.ID_COL].duplicated().sum())} 个",
    f"- 标签：违约 {int(y.sum()):,}，未违约 {int((1 - y).sum()):,}，违约率 {y.mean():.4f}",
    "",
    "## 类别变量的原始取值",
    "",
    "| 变量 | 取值：行数 |",
    "|---|---|",
]
for col in C.CAT_COLS + C.PAY_COLS:
    counts = raw[col].value_counts().sort_index()
    lines.append(f"| {col} | " + "，".join(f"{k}：{v:,}" for k, v in counts.items()) + " |")

dup = raw[raw.duplicated(C.FEATURES, keep=False)]
groups = dup.groupby(C.FEATURES)[C.TARGET_RAW].agg(["size", "nunique"])
zero = (dup[C.BILL_COLS + C.PAY_AMT_COLS].abs().sum(axis=1) == 0).mean()
lines += [
    "",
    "## 特征完全相同的行",
    "",
    f"- 共 {len(dup)} 行，分属 {len(groups)} 组；其中 {int((groups['nunique'] > 1).sum())} 组内标签不一致",
    f"- 这些行中 {zero:.0%} 的账单和还款金额全为 0，即没有交易的账户",
    "- 处理：标签不一致说明它们是不同客户而非重复录入，全部保留；划分时同组的行放在同一分区、同一折",
    "",
    "## 数值变量范围",
    "",
    "| 变量 | 最小 | 中位数 | 最大 |",
    "|---|---:|---:|---:|",
]
for col in ["LIMIT_BAL", "AGE"] + C.BILL_COLS + C.PAY_AMT_COLS:
    s = raw[col]
    lines.append(f"| {col} | {s.min():,.0f} | {s.median():,.0f} | {s.max():,.0f} |")

changed = int((raw["EDUCATION"] != clean["EDUCATION"]).sum()), int((raw["MARRIAGE"] != clean["MARRIAGE"]).sum())
lines += [
    "",
    "## 重编码影响的行数",
    "",
    f"- EDUCATION 的 0、5、6 并入 4：{changed[0]} 行",
    f"- MARRIAGE 的 0 并入 3：{changed[1]} 行",
    "",
]

out = C.ROOT / "docs" / "data_check.md"
out.write_text("\n".join(lines), encoding="utf-8")
print(f"已写入 {out.relative_to(C.ROOT)}")
