"""生成全组共用的 70/30 划分和五折索引，写入 data/splits/splits.csv。

只需运行一次；结果已提交进仓库，其他人直接读取，不要重新生成。
特征完全相同的行视为一组：先对每组的第一行做分层划分，再把结果传给组内其他行，
保证同组的行不会一部分在训练、一部分在验证或测试。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.model_selection import StratifiedKFold, train_test_split

from src import config as C
from src.data.load import load_clean

df = load_clean()
df["group"] = df.groupby(C.FEATURES, sort=False).ngroup()
reps = df.drop_duplicates("group")

dev_groups, test_groups = train_test_split(
    reps["group"], test_size=C.TEST_SIZE, stratify=reps[C.TARGET], random_state=C.SEED
)
df["partition"] = df["group"].isin(set(dev_groups)).map({True: "dev", False: "test"})

dev_reps = reps[reps["group"].isin(set(dev_groups))]
fold_of_group = {}
skf = StratifiedKFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
for k, (_, val) in enumerate(skf.split(dev_reps, dev_reps[C.TARGET])):
    fold_of_group.update(dict.fromkeys(dev_reps["group"].iloc[val], k))
df["fold"] = df["group"].map(fold_of_group).fillna(-1).astype(int)

C.SPLITS_FILE.parent.mkdir(parents=True, exist_ok=True)
df[[C.ID_COL, "partition", "fold", "group"]].to_csv(C.SPLITS_FILE, index=False)

summary = (
    df.assign(part=df["partition"].where(df["fold"] < 0, "dev fold " + df["fold"].astype(str)))
    .groupby("part")[C.TARGET]
    .agg(n="size", defaults="sum", default_rate="mean")
)
summary.loc["dev (all)"] = [
    (df["partition"] == "dev").sum(),
    df.loc[df["partition"] == "dev", C.TARGET].sum(),
    df.loc[df["partition"] == "dev", C.TARGET].mean(),
]
summary = summary.astype({"n": int, "defaults": int}).round({"default_rate": 4})
C.TABLE_DIR.mkdir(parents=True, exist_ok=True)
summary.to_csv(C.TABLE_DIR / "split_summary.csv")
print(summary)
