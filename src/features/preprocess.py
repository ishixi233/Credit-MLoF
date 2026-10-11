"""共享预处理。必须作为 Pipeline 的第一步，这样每折只在该折的训练部分上拟合。"""
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config as C


def make_preprocessor(scale: bool, pay_onehot: bool = False) -> ColumnTransformer:
    """SEX、EDUCATION、MARRIAGE 做 one-hot；其余列按数值处理。

    scale=True 时对数值列做标准化，用于 Logistic、KNN、Gaussian NB；树模型用 scale=False。
    pay_onehot=True 时 PAY_* 也做 one-hot。PAY 的 -2、-1、0 并非严格有序，线性模型把它当
    数值会吃亏；树模型能自己切分，保持数值即可。
    """
    cat = C.CAT_COLS + (C.PAY_COLS if pay_onehot else [])
    num = [c for c in C.NUM_COLS if c not in cat]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
            ("num", StandardScaler() if scale else "passthrough", num),
        ],
        verbose_feature_names_out=False,
    )
