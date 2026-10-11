"""模型登记表。每个模型给出 build（返回含预处理的 Pipeline）和 grid（候选参数列表）。

添加自己的模型：照着下面的写法加一个 build 函数，再在 MODELS 里登记一项，然后运行
    python scripts/03_run_cv.py <模型名>
grid 在实验前写定，较简单的配置放前面（平均 AP 并列时取靠前的）。
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

from src import config as C
from src.features.preprocess import make_preprocessor


def baseline_rate():
    """常数分数基准：对所有人输出训练部分的违约率。"""
    return Pipeline([("prep", make_preprocessor(scale=False)), ("clf", DummyClassifier(strategy="prior"))])


def logistic(C_reg: float, pay_onehot: bool):
    return Pipeline(
        [
            ("prep", make_preprocessor(scale=True, pay_onehot=pay_onehot)),
            ("clf", LogisticRegression(C=C_reg, max_iter=5000, random_state=C.SEED)),
        ]
    )


def gaussian_nb(pay_onehot: bool):
    return Pipeline([("prep", make_preprocessor(scale=True, pay_onehot=pay_onehot)), ("clf", GaussianNB())])


MODELS = {
    "baseline_rate": {"build": baseline_rate, "grid": [{}]},
    "logistic": {
        "build": logistic,
        "grid": [{"C_reg": c, "pay_onehot": p} for p in (False, True) for c in (0.1, 1.0, 10.0)],
    },
    # 只用一套默认配置。PAY one-hot 试过：五折 AP 0.491 对 0.487，差距小于折间标准差（约 0.015），
    # 且 0/1 变量不符合高斯假设、会把八成客户的分数压到接近 0，所以不采用。
    "gaussian_nb": {"build": gaussian_nb, "grid": [{"pay_onehot": False}]},
    # Li Erteng: "knn", "decision_tree"
    # Jiang Chengrui: "random_forest", "adaboost"
}
