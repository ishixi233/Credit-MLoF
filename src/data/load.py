"""读取原始数据、应用固定的重编码规则、按共享划分取出开发集或测试集。"""
import hashlib

import pandas as pd

from src import config as C


def load_raw() -> pd.DataFrame:
    """读取原始 Excel。第一行是 X1..X23 的占位表头，真正的列名在第二行。"""
    sha = hashlib.sha256(C.RAW_FILE.read_bytes()).hexdigest()
    if sha != C.RAW_SHA256:
        raise RuntimeError(f"原始数据文件与约定版本不一致：{sha}")
    df = pd.read_excel(C.RAW_FILE, header=1)
    assert df.shape == (30000, 25), df.shape
    assert df[C.ID_COL].is_unique and not df.isna().any().any()
    return df


def load_clean() -> pd.DataFrame:
    """返回 ID、23 个特征和标签 y（1 = 下月违约）。

    重编码只依赖单行自身的取值，不使用任何统计量，所以可以在划分之前做：
    EDUCATION 中文档未说明的 0、5、6 并入 4（others）；MARRIAGE 中的 0 并入 3（others）。
    PAY_* 中文档未说明的 -2 和 0 保留原值。
    """
    df = load_raw().rename(columns={C.TARGET_RAW: C.TARGET})
    df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
    df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})
    assert set(df[C.TARGET].unique()) == {0, 1}
    return df[[C.ID_COL] + C.FEATURES + [C.TARGET]]


def load_splits() -> pd.DataFrame:
    return pd.read_csv(C.SPLITS_FILE)


def _partition(name: str) -> pd.DataFrame:
    df = load_clean().merge(load_splits(), on=C.ID_COL, validate="one_to_one")
    return df[df["partition"] == name].reset_index(drop=True)


def load_dev():
    """70% 训练开发集。返回 X, y, fold, ids；fold 取 0..4。"""
    df = _partition("dev")
    return df[C.FEATURES], df[C.TARGET], df["fold"], df[C.ID_COL]


def load_test():
    """30% 测试集。只有冻结清单存在时才允许读取，防止测试前误用。"""
    if not C.FROZEN_FILE.exists():
        raise RuntimeError(
            f"找不到 {C.FROZEN_FILE.relative_to(C.ROOT)}：模型和阈值冻结并提交之前不能读取测试集。"
        )
    df = _partition("test")
    return df[C.FEATURES], df[C.TARGET], df[C.ID_COL]
